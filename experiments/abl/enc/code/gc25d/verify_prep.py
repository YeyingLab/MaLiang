#!/usr/bin/env python3
"""prep_v10 产物校验（data_v10.md §8.5）。

    python verify_prep.py --root $M/data/gc25d_v10 [--spot 16] [--device cuda:0] [--workers 32]

分辨率 = geometry（env GC_RES，默认 832x480）：index/prep_args.json 的 "res" 与每行的 "res"（缺失 = 832x480）必须等于 GC_RES，
下面的 [48,21,30,52] / [81,480,832] 是 832×480 的例子（1280×704：[48,21,44,80] / [81,704,1280]）。
A. index：res 一致、clips.jsonl 字段齐、clip_id 唯一、split = 规则、drop = buried 名单、runs.json 一致、每 run 50 条、
   text_table_gc25d.pt 与源文件逐字节相同且 19 个 key = text.all_prompts() + NULL_PROMPT、没有残留 .tmp 文件。
B. 全部 clip（多进程）：文件齐（cond_ext 当且仅当 has_ext）；latent bf16 [48,21,30,52] 有限；actions_block uint8 [21,5]，
   cell 0 = cell 1，[1:] = 原始 actions_block[s//4 : s//4+20]；prompts == text.cell_prompts(actions_block[1:])；cond ids
   uint8 [81,480,832] ⊆ parts 词表（无 255）；cond_ext ids [28,480,832] / blocks [7,5] = 原始块 / prompts；has_ext 用
   frames.jsonl 独立重判；meta 形状 / dtype（F = 81 或 109）、frames 连续、player_px / boss_px == cond 里数出来的像素数。
C. GPU 抽查 N 条：
   * VAE 解码（fp32）PSNR vs resample_rgb(GT)（典型 23–26 dB，简单内容 30+；< 18 dB ⇒ FAIL，[PSNR_WARN_LO, 40] 外 ⇒ WARN；
     PSNR_WARN_LO = 21（832×480 上定的），其它分辨率 20：GT 高频更多，繁杂 clip 低 ≈ 1 dB，704p mini 实测最低 20.26）；
   * 官方 vae.encode([x], tiled=False) 重编整段 == 存的 latent（逐位）；latent cell 0 == encode(帧 0)（到 bf16 舍入）；
   * 源 ids.png ⊆ {0..6,10,11}；collapse_parts(cond ids) == resample_ids(ids.png)（全部帧，含扩展帧）；meta 与 tar json 逐位相同；
     从 meta 重跑 label_parts == cond；
   * 从 meta 用 capsules.render_capsules 重画角色：v10 无遮挡帧 IoU ≥ 0.98（及 GT ⊆ 重画的覆盖率）。
退出码：有 FAIL ⇒ 1。报告写 <root>/index/verify_report.json。
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import prep_v10 as P   # 先 import：设单线程环境变量

import cv2
import numpy as np
import torch

import capsules as CAP
import geometry as GEO
import model as GM
import text as TX

ROW_KEYS = ("clip_id", "subset", "scene", "run", "start", "end", "split", "drop", "drop_reason", "episode",
            "camera_variant", "look_name", "centered", "n_events", "dead_frac", "has_ext", "ortho")
PT_KEYS = ("latent", "prompts", "actions_block", "clip_id", "subset", "scene", "run", "start")


def meta_spec(F: int) -> dict:
    return {
        "world_to_clip": ("f8", (F, 16)), "world_to_view": ("f8", (F, 16)), "projection": ("f8", (F, 16)),
        "orthographic": ("b1", ()), "fov_x_deg": ("f4", (F,)), "ortho_width_cm": ("f4", (F,)),
        "frame_offset_cm": ("f4", (F, 3)), "capsules": ("f4", (F, 2, 12, 7)), "actor_pos": ("f4", (F, 2, 3)),
        "ground_z": ("f4", (F,)), "player_state": ("U", (F,)), "boss_state": ("U", (F,)),
        "player_px": ("i4", (F,)), "boss_px": ("i4", (F,)),
    }


def _dtype_ok(arr, code):
    return arr.dtype.kind == "U" if code == "U" else arr.dtype == np.dtype(code)


def _px(ids):
    """[F,H,W] parts ids → (player_px, boss_px) int [F]。"""
    c = GM.collapse_parts(ids)
    return (c == GEO.PLAYER_ID).sum((1, 2)), (c == GEO.BOSS_ID).sum((1, 2))


# ----------------------------------------------------------------------------
# B. 逐 clip 校验（CPU，多进程）
# ----------------------------------------------------------------------------
_TABLE_KEYS: frozenset = frozenset()
_SRC = ""


def _pool_init(table_keys, src):
    global _TABLE_KEYS, _SRC
    _TABLE_KEYS, _SRC = frozenset(table_keys), src
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def check_clip(args) -> tuple[str, list[str], dict]:
    row, root = args
    cid, s = row["clip_id"], row["start"]
    err = []
    paths = P.clip_paths(root, cid)
    for k in ("latent", "cond", "meta") + (("cond_ext",) if row["has_ext"] else ()):
        if not os.path.isfile(paths[k]):
            err.append(f"缺 {k}")
    if not row["has_ext"] and os.path.isfile(paths["cond_ext"]):
        err.append("has_ext=False 却有 cond_ext")
    if err:
        return cid, err, {}
    run_dir = os.path.join(_SRC, row["subset"], row["scene"], row["run"])
    AB = np.load(os.path.join(run_dir, "index", "actions_block_0.25s.npy"))

    # latent .pt
    d = torch.load(paths["latent"], map_location="cpu", weights_only=True)
    miss = [k for k in PT_KEYS if k not in d]
    if miss:
        err.append(f"latent 缺 key {miss}")
    else:
        lat, ab = d["latent"], d["actions_block"]
        if lat.dtype != torch.bfloat16 or tuple(lat.shape) != P.LAT_SHAPE:
            err.append(f"latent {lat.dtype} {tuple(lat.shape)}")
        elif not torch.isfinite(lat.float()).all():
            err.append("latent 有非有限值")
        ab = ab.numpy() if torch.is_tensor(ab) else np.asarray(ab)
        if ab.dtype != np.uint8 or ab.shape != (21, 5) or not np.isin(ab, (0, 1)).all():
            err.append(f"actions_block {ab.dtype} {ab.shape}")
        else:
            if not (ab[0] == ab[1]).all():
                err.append("actions_block cell 0 ≠ cell 1")
            if not (ab[1:] == P.clip_blocks(AB, s)).all():
                err.append("actions_block[1:] ≠ 原始 actions_block[s//4:s//4+20]")
            if list(d["prompts"]) != TX.cell_prompts(ab[1:]):
                err.append("prompts ≠ text.cell_prompts(actions_block[1:])")
        if not set(d["prompts"]) <= _TABLE_KEYS:
            err.append(f"prompt 不在 text table：{set(d['prompts']) - _TABLE_KEYS}")
        for k in ("clip_id", "subset", "scene", "run", "start"):
            if d[k] != row[k]:
                err.append(f"latent[{k}]={d[k]!r} ≠ row {row[k]!r}")

    # has_ext 独立重判（index/frames.jsonl，不是 prep 用的 tar json）
    with open(os.path.join(run_dir, "index", "frames.jsonl")) as fh:
        fr = [json.loads(line) for line in itertools.islice(fh, s, s + P.F_SPAN)]
    he = P.ext_valid([j["episode"] for j in fr], [j["camera_variant"] for j in fr], s, len(AB))
    if he != row["has_ext"]:
        err.append(f"has_ext={row['has_ext']}，frames.jsonl 判为 {he}")

    # cond / cond_ext
    with np.load(paths["cond"], allow_pickle=False) as z:
        ids = z["ids"]
    if ids.dtype != np.uint8 or ids.shape != (P.F_CLIP, P.DST_H, P.DST_W):
        err.append(f"cond ids {ids.dtype} {ids.shape}")
        return cid, err, {}
    bad = np.nonzero(np.bincount(ids.ravel(), minlength=256) * ~P._VALID_PARTS)[0]
    if len(bad):
        err.append(f"cond ids 有词表外值 {bad.tolist()}")
    if row["has_ext"]:
        with np.load(paths["cond_ext"], allow_pickle=False) as z:
            eids, eb, ep = z["ids"], z["blocks"], z["prompts"]
        if eids.dtype != np.uint8 or eids.shape != (P.F_EXT, P.DST_H, P.DST_W):
            err.append(f"cond_ext ids {eids.dtype} {eids.shape}")
        else:
            bad = np.nonzero(np.bincount(eids.ravel(), minlength=256) * ~P._VALID_PARTS)[0]
            if len(bad):
                err.append(f"cond_ext ids 有词表外值 {bad.tolist()}")
            ids = np.concatenate([ids, eids])
        if eb.dtype != np.uint8 or eb.shape != (P.N_EXT_CELLS, 5) or not (eb == P.ext_blocks(AB, s)).all():
            err.append("cond_ext blocks ≠ 原始 actions_block[s//4+20:s//4+27]")
        elif ep.shape != (P.N_EXT_CELLS,) or ep.tolist() != [TX.block_to_prompt(*r.tolist()) for r in eb]:
            err.append("cond_ext prompts ≠ block_to_prompt(blocks)")

    # meta
    F = P.F_SPAN if row["has_ext"] else P.F_CLIP
    with np.load(paths["meta"], allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    for k, (code, shape) in meta_spec(F).items():
        if k not in m:
            err.append(f"meta 缺 {k}")
        elif not _dtype_ok(m[k], code) or m[k].shape != shape:
            err.append(f"meta {k}: {m[k].dtype} {m[k].shape}，应为 {code} {shape}")
    if err:
        return cid, err, {}
    if bool(m["orthographic"]) != row["ortho"]:
        err.append("meta orthographic ≠ row ortho")
    if "frames" in m and not (m["frames"] == s + np.arange(F)).all():
        err.append("meta frames 不连续")
    for k in ("world_to_clip", "world_to_view", "projection", "fov_x_deg", "ortho_width_cm", "frame_offset_cm"):
        if not np.isfinite(m[k]).all():
            err.append(f"meta {k} 有非有限值")
    pp, bp = _px(ids)
    if not (pp == m["player_px"]).all() or not (bp == m["boss_px"]).all():
        err.append("meta player_px / boss_px ≠ cond 里的像素数")
    has_player_caps = ~np.isnan(m["capsules"][:, 0, 0, 0])
    if not np.isfinite(m["ground_z"][has_player_caps]).all():
        err.append("ground_z 在有玩家胶囊的帧上非有限")
    info = {"subset": row["subset"], "player_px_mean": float(pp[:P.F_CLIP].mean()), "boss_px_mean": float(bp[:P.F_CLIP].mean()),
            "bytes": {k: os.path.getsize(paths[k]) for k in paths if os.path.isfile(paths[k])}}
    return cid, err, info


# ----------------------------------------------------------------------------
# C. GPU 抽查
# ----------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


# 832×480 上定的阈值按分辨率换算（832×480 时与原值相同）
PSNR_WARN_LO = 21.0 if GEO.RES_TAG == GEO.LEGACY_RES else 20.0
VIS_MIN_PX = 50 * GEO.AREA_VS_480                     # 「可见帧」的 GT 像素下限（像素数 ∝ 面积）


def render_stats(cond_ids, meta, frames) -> list[dict]:
    """从 meta 重画胶囊（不看 ids）→ 每帧每角色的 IoU / 覆盖率 / 遮挡比例 / 部位一致率。"""
    out = []
    for i in frames:
        caps = CAP.capsules_from_array(meta["capsules"][i])
        with np.errstate(invalid="ignore"):            # cap_hit 里 inf·0（没命中的射线），无害
            rend = CAP.render_capsules(np.zeros((P.DST_H, P.DST_W), np.uint8), meta["world_to_clip"][i], meta["world_to_view"][i], caps)
        g, r = GM.collapse_parts(cond_ids[i]), GM.collapse_parts(rend)
        for cls in (GEO.PLAYER_ID, GEO.BOSS_ID):
            G, Rm = g == cls, r == cls
            inter = G & Rm
            out.append({"frame": int(i), "cls": cls, "g": int(G.sum()), "r": int(Rm.sum()), "i": int(inter.sum()),
                        "u": int((G | Rm).sum()), "occl": int((Rm & (cond_ids[i] < GEO.PLAYER_ID)).sum()),
                        "part_eq": int((rend[inter] == cond_ids[i][inter]).sum())})
    return out


def spot_check(row, root, src, pipe, dev, log) -> tuple[list[str], dict]:
    cid, s = row["clip_id"], row["start"]
    err = []
    paths = P.clip_paths(root, cid)
    F = P.F_SPAN if row["has_ext"] else P.F_CLIP
    R = P.RunTars(os.path.join(src, row["subset"], row["scene"], row["run"]))
    Js = [R.json(s + i) for i in range(F)]
    gt = np.stack([GEO.resample_rgb(cv2.cvtColor(cv2.imdecode(np.frombuffer(R.get(s + i, "jpg"), np.uint8), cv2.IMREAD_COLOR),
                                                 cv2.COLOR_BGR2RGB)) for i in range(P.F_CLIP)])
    raws = [cv2.imdecode(np.frombuffer(R.get(s + i, "ids.png"), np.uint8), cv2.IMREAD_UNCHANGED) for i in range(F)]
    R.close()
    for i, raw in enumerate(raws):                    # 源 ids ⊆ {0..6,10,11}（全分辨率）
        try:
            P.check_raw_ids(raw, f"帧 {s + i}")
        except AssertionError as e:
            err.append(str(e))
            break
    ids_src = np.stack([GEO.resample_ids(raw) for raw in raws])
    del raws
    with np.load(paths["cond"]) as z:
        cond = z["ids"]
    if row["has_ext"]:
        with np.load(paths["cond_ext"]) as z:
            cond = np.concatenate([cond, z["ids"]])
    with np.load(paths["meta"]) as z:
        meta = {k: z[k] for k in z.files}
    info = {"clip_id": cid, "subset": row["subset"], "drop": row["drop"], "F": F}

    # label_parts 一致性
    n_bad = int((GM.collapse_parts(cond) != ids_src).sum())
    info["collapse_mismatch_px"] = n_bad
    if n_bad:
        err.append(f"collapse_parts(cond) ≠ resample_ids(ids.png)：{n_bad} px")
    for k in ("world_to_clip", "world_to_view", "projection"):
        if not (meta[k] == np.array([J[k] for J in Js], np.float64)).all():
            err.append(f"meta {k} ≠ tar json")
    caps_j = np.stack([CAP.capsules_to_array(CAP.capsules_from_json(J)) for J in Js])
    if not np.array_equal(meta["capsules"], caps_j, equal_nan=True):
        err.append("meta capsules ≠ capsules_to_array(tar json)")
    sub = list(range(0, F, 4))
    eq_json = eq_meta = tot = 0
    for i in sub:
        J = Js[i]
        a = CAP.label_parts(ids_src[i], J["world_to_clip"], J["world_to_view"], CAP.capsules_from_json(J))
        b = CAP.label_parts(ids_src[i], meta["world_to_clip"][i], meta["world_to_view"][i], CAP.capsules_from_array(meta["capsules"][i]))
        ch = ids_src[i] >= GEO.PLAYER_ID
        eq_json += int((a == cond[i]).all())
        eq_meta += int((b[ch] == cond[i][ch]).sum())
        tot += int(ch.sum())
    info["relabel_json_exact_frames"] = f"{eq_json}/{len(sub)}"
    info["relabel_meta_char_px_agree"] = eq_meta / max(tot, 1)
    if eq_json != len(sub):
        err.append(f"label_parts(tar json) ≠ cond 在 {len(sub) - eq_json}/{len(sub)} 帧")
    if info["relabel_meta_char_px_agree"] < 0.999:
        err.append(f"label_parts(meta) 与 cond 角色像素一致率 {info['relabel_meta_char_px_agree']:.4f} < 0.999")

    # VAE
    d = torch.load(paths["latent"], map_location="cpu", weights_only=True)
    lat = d["latent"]
    with torch.no_grad():
        rec = pipe.vae.decode([lat.float()], device=dev, tiled=False)[0]          # [3,81,H,W] ∈ [-1,1]
        rec = rec.float().add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        info["psnr_db"] = psnr(rec, gt)
        info["psnr_frame0_db"] = psnr(rec[:1], gt[:1])
        x = torch.from_numpy(gt).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)  # infer.encode_video 原样（CPU）
        z = pipe.vae.encode([x], device=dev, tiled=False)[0].to(torch.bfloat16).cpu()
        z0 = pipe.vae.encode([x[:, :1]], device=dev, tiled=False)[0].float().cpu()   # [48,1,H/16,W/16] fp32
    info["reencode_exact_frac"] = float((z == lat).float().mean())
    info["reencode_max_abs"] = float((z.float() - lat.float()).abs().max())
    c0 = lat[:, :1].float()
    info["cell0_bf16_exact_frac"] = float((z0.to(torch.bfloat16).float() == c0).float().mean())
    info["cell0_rel_l2"] = float((z0 - c0).norm() / z0.norm())
    if info["psnr_db"] < 18.0:                        # 错位 / 串 clip 才会这么低；内容简单的 clip 可到 35 dB 以上
        err.append(f"VAE 重建 PSNR {info['psnr_db']:.2f} dB < 18")
    elif not PSNR_WARN_LO <= info["psnr_db"] <= 40.0:
        info["warn"] = f"VAE 重建 PSNR {info['psnr_db']:.2f} dB 不在 [{PSNR_WARN_LO:g}, 40]"
    if info["reencode_exact_frac"] < 0.999:
        err.append(f"官方 encode 重编 latent 一致率 {info['reencode_exact_frac']:.5f}")
    if info["cell0_bf16_exact_frac"] < 0.999 or info["cell0_rel_l2"] > 4e-3:
        err.append(f"latent cell 0 ≠ encode(帧 0)：bf16 一致率 {info['cell0_bf16_exact_frac']:.5f}，rel L2 {info['cell0_rel_l2']:.2e}")

    # 胶囊重画（只用 meta）
    info["render"] = render_stats(cond, meta, list(range(0, F, 4)))
    log(f"  {cid} F={F}: PSNR {info['psnr_db']:.2f} dB (f0 {info['psnr_frame0_db']:.2f}); re-encode exact {info['reencode_exact_frac']:.5f} "
        f"(max {info['reencode_max_abs']:.2e}); cell0 bf16 exact {info['cell0_bf16_exact_frac']:.5f} rel {info['cell0_rel_l2']:.1e}; "
        f"collapse mismatch {n_bad}; relabel json {info['relabel_json_exact_frames']} meta {info['relabel_meta_char_px_agree']:.5f}"
        + (f"  ERR {err}" if err else ""))
    return err, info


def summarize_render(spots) -> dict:
    """v10 未 drop 的抽查帧：pooled IoU / 覆盖率；无遮挡帧（occl < 1 %，GT ≥ 50 px · AREA_VS_480）的逐帧 IoU。"""
    out = {}
    for subset in P.SUBSETS:
        recs = [r for sp in spots if sp["subset"] == subset and not sp["drop"] for r in sp.get("render", [])]
        if not recs:
            continue
        o = {}
        for cls, name in ((GEO.PLAYER_ID, "player"), (GEO.BOSS_ID, "boss")):
            rs = [r for r in recs if r["cls"] == cls]
            vis = [r for r in rs if r["g"] >= VIS_MIN_PX]
            un = [r for r in vis if r["occl"] < 0.01 * max(r["r"], 1)]
            ious = [r["i"] / r["u"] for r in un]
            o[name] = {
                "frames": len(rs), "visible_frames": len(vis), "unoccluded_frames": len(un),
                "pooled_iou": sum(r["i"] for r in rs) / max(1, sum(r["u"] for r in rs)),
                "coverage_gt_in_render": sum(r["i"] for r in rs) / max(1, sum(r["g"] for r in rs)),
                "occluded_frac_of_render": sum(r["occl"] for r in rs) / max(1, sum(r["r"] for r in rs)),
                "unoccluded_iou_median": float(np.median(ious)) if ious else None,
                "unoccluded_iou_p10": float(np.percentile(ious, 10)) if ious else None,
                "unoccluded_pooled_iou": sum(r["i"] for r in un) / max(1, sum(r["u"] for r in un)) if un else None,
                "part_agree_in_intersection": sum(r["part_eq"] for r in rs) / max(1, sum(r["i"] for r in rs)),
            }
        out[subset] = o
    return out


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--src", default=P.SRC_DEFAULT)
    ap.add_argument("--spot", type=int, default=8, help="GPU 抽查的 clip 数（0 = 不抽查）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--buried", default=P.BURIED_LIST)
    ap.add_argument("--text_table", default=P.TEXT_TABLE_SRC)
    ap.add_argument("--clips_per_run", type=int, default=50, help="每 run 应有的 clip 数（0 = 不查）")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    T0 = time.time()
    fails, warns = [], []
    log = lambda *x: print(*x, flush=True)        # noqa: E731
    GEO.check_root_res(root, "--root")            # prep_args.json 的 res ≠ GC_RES ⇒ 直接报错（形状全对不上，没法往下验）

    # ---------------- A. index
    rows = [json.loads(line) for line in open(os.path.join(root, "index", "clips.jsonl")) if line.strip()]
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        fails.append("clips.jsonl 有重复 clip_id")
    buried = {x.strip() for x in open(args.buried) if x.strip()}
    for r in rows:
        miss = [k for k in ROW_KEYS if k not in r]
        if miss:
            fails.append(f"{r.get('clip_id')}: 行缺字段 {miss}")
            continue
        if r["clip_id"] != P.clip_id_of(r["scene"], r["run"], r["start"]) or r["end"] != r["start"] + P.F_CLIP:
            fails.append(f"{r['clip_id']}: clip_id / end 不对")
        if r["split"] != P.split_of(r["run"]):
            fails.append(f"{r['clip_id']}: split {r['split']} ≠ 规则 {P.split_of(r['run'])}")
        if r["drop"] != (r["clip_id"] in buried) or (r["drop_reason"] == P.DROP_REASON) != r["drop"]:
            fails.append(f"{r['clip_id']}: drop / drop_reason 与 buried 名单不符")
    bad_res = collections.Counter(r.get("res", GEO.LEGACY_RES) for r in rows if r.get("res", GEO.LEGACY_RES) != GEO.RES_TAG)
    if bad_res:
        fails.append(f"index 行的 res {dict(bad_res)} ≠ {GEO.RES_ENV}={GEO.RES_TAG}")
    per_run = collections.Counter(r["run"] for r in rows)
    if args.clips_per_run:
        badrun = {k: v for k, v in per_run.items() if v != args.clips_per_run}
        if badrun:
            fails.append(f"这些 run 的 clip 数 ≠ {args.clips_per_run}：{badrun}")
    runs = json.load(open(os.path.join(root, "index", "runs.json")))
    if "_rule" not in runs or {k: v for k, v in runs.items() if k != "_rule"} != {r["run"]: r["split"] for r in rows}:
        fails.append("runs.json 与 clips.jsonl 不一致")
    tt = os.path.join(root, "text_table_gc25d.pt")
    table_keys = []
    if not os.path.isfile(tt):
        fails.append("缺 text_table_gc25d.pt")
    else:
        h = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (tt, args.text_table)]
        if h[0] != h[1]:
            fails.append("text_table_gc25d.pt 与源文件不同")
        table = torch.load(tt, map_location="cpu", weights_only=True)
        table_keys = sorted(table)
        if set(table_keys) != set(TX.all_prompts()) | {TX.NULL_PROMPT}:
            fails.append("text table 的 key ≠ 18 条 prompt + NULL_PROMPT")
        for k, v in table.items():
            if v.dtype != torch.bfloat16 or tuple(v.shape) != (512, 4096):
                fails.append(f"text table[{k!r}] {v.dtype} {tuple(v.shape)}")
    stray = [p for d in ("latents", "cond", "cond_ext", "meta") for p in glob.glob(os.path.join(root, d, "*.tmp*"))]
    if stray:
        warns.append(f"{len(stray)} 个残留 .tmp 文件（中断的运行），例如 {stray[:3]}")
    on_disk = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(root, "latents", "*.pt"))}
    orphan = on_disk - set(ids)
    if orphan:
        warns.append(f"{len(orphan)} 个 latent 不在 clips.jsonl 里")
    sp = collections.Counter((r["subset"], r["split"], r["drop"], r["has_ext"]) for r in rows)
    log(f"A. index: res {GEO.RES_TAG}, {len(rows)} clips, {len(per_run)} runs; " + "; ".join(f"{k}: {v}" for k, v in sorted(sp.items())))
    # has_ext 里有多少在 cond_ext 内换 look
    ext = {s: {"has_ext": sum(r["has_ext"] for r in rows if r["subset"] == s),
               "ext_look_change": sum(bool(r.get("ext_look_change")) for r in rows if r["subset"] == s),
               "dmd_pool_train": sum(r["has_ext"] and not r["drop"] and r["split"] == "train" for r in rows if r["subset"] == s),
               "dmd_pool_train_same_look": sum(r["has_ext"] and not r["drop"] and r["split"] == "train" and not r.get("ext_look_change")
                                               for r in rows if r["subset"] == s)}
           for s in P.SUBSETS}
    log("   ext: " + "; ".join(f"{s} " + " ".join(f"{k}={v}" for k, v in o.items()) for s, o in ext.items()))

    # ---------------- B. 逐 clip
    t = time.time()
    nbad = 0
    infos = []
    with Pool(min(args.workers, max(1, len(rows))), initializer=_pool_init, initargs=(table_keys, args.src)) as pool:
        for cid, err, info in pool.imap_unordered(check_clip, [(r, root) for r in rows], chunksize=1):
            if err:
                nbad += 1
                if nbad <= 50:
                    fails.append(f"{cid}: {'; '.join(err)}")
            if info:
                infos.append(info)
    if nbad > 50:
        fails.append(f"…… 共 {nbad} 个 clip 有错")
    pxs = {s: (np.mean([i["player_px_mean"] for i in infos if i["subset"] == s]),
               np.mean([i["boss_px_mean"] for i in infos if i["subset"] == s])) for s in P.SUBSETS if any(i["subset"] == s for i in infos)}
    size = collections.defaultdict(list)
    for i in infos:
        for k, v in i["bytes"].items():
            size[k].append(v)
    sizes = {k: {"mean_MB": round(float(np.mean(v)) / 1e6, 3), "total_GB": round(sum(v) / 1e9, 3)} for k, v in size.items()}
    log(f"B. per-clip: {len(rows) - nbad}/{len(rows)} ok ({time.time() - t:.0f}s); player/boss px per frame ({GEO.RES_TAG}) "
        + ", ".join(f"{s} {a:.0f}/{b:.0f}" for s, (a, b) in pxs.items()) + f"; sizes {sizes}")

    # ---------------- C. GPU 抽查
    spots = []
    if args.spot and rows:
        order = {s: i for i, s in enumerate(P.SUBSETS)}
        cand = sorted(rows, key=lambda r: (order[r["subset"]], r["scene"], r["run"], r["start"]))
        pick = sorted(set(np.linspace(0, len(cand) - 1, min(args.spot, len(cand))).round().astype(int).tolist()))
        log(f"C. spot check {len(pick)} clips on {args.device}")
        pipe = P.load_vae(args.device)
        for k in pick:
            try:
                err, info = spot_check(cand[k], root, args.src, pipe, args.device, log)
            except Exception as e:                 # noqa: BLE001
                err, info = [f"抽查异常 {type(e).__name__}: {e}"], {"clip_id": cand[k]["clip_id"], "subset": cand[k]["subset"], "drop": True}
            spots.append(info)
            if info.get("warn"):
                warns.append(f"spot {cand[k]['clip_id']}: {info['warn']}")
            fails += [f"spot {cand[k]['clip_id']}: {e}" for e in err]
        rs = summarize_render(spots)
        for subset, o in rs.items():
            for name, v in o.items():
                log(f"   render {subset:<11} {name:<6} " + " ".join(f"{k}={v[k]:.4f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in v))
        v10 = rs.get("v10_real")
        if v10 is None:
            warns.append("抽查里没有 v10_real 的未 drop clip，跳过胶囊重画 IoU 判据")
        else:
            for name, v in v10.items():
                if v["visible_frames"] and v["unoccluded_frames"] < 0.5 * v["visible_frames"]:
                    fails.append(f"render v10 {name}: 无遮挡帧只占 {v['unoccluded_frames']}/{v['visible_frames']}（meta 几何可能错位）")
                if v["coverage_gt_in_render"] < 0.995:
                    fails.append(f"render v10 {name}: GT ⊆ 重画覆盖率 {v['coverage_gt_in_render']:.4f} < 0.995")
                if v["unoccluded_iou_median"] is not None and (v["unoccluded_iou_median"] < 0.98 or v["unoccluded_pooled_iou"] < 0.98):
                    fails.append(f"render v10 {name}: 无遮挡帧 IoU median {v['unoccluded_iou_median']:.4f} / pooled {v['unoccluded_pooled_iou']:.4f} < 0.98")
        ps = [s["psnr_db"] for s in spots if "psnr_db" in s]
        if ps:
            log(f"   VAE recon PSNR: mean {np.mean(ps):.2f} dB, min {np.min(ps):.2f}, max {np.max(ps):.2f}")
    else:
        rs = {}

    report = {"root": root, "res": GEO.RES_TAG, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - T0, 1),
              "n_clips": len(rows), "n_runs": len(per_run), "fails": fails, "warns": warns, "ext": ext,
              "px_per_frame": {s: {"player": a, "boss": b} for s, (a, b) in pxs.items()}, "sizes": sizes,
              "spots": [{k: v for k, v in s.items() if k != "render"} for s in spots], "render": rs}
    with open(os.path.join(root, "index", "verify_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=float)
    for w in warns:
        log("WARN", w)
    for f in fails[:80]:
        log("FAIL", f)
    log(f"verify_prep: {'FAIL' if fails else 'OK'} — {len(rows)} clips, {len(fails)} fails, {len(warns)} warns, {time.time() - T0:.0f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
