#!/usr/bin/env python3
"""stage3 rollout 延长段的「代码世界式」重渲染：cond_ext_canvas/<clip>.npz。

为什么
------
v10_real 每个 clip 一个 look（biome / 光照 / 布景重新散布），切换恰好落在帧 s+81：prep 写的 cond_ext（UE 的 ids）在
cell 21 处静态布景整体换掉（中位直方图跳变 0.088，是片内相邻帧的 34 倍）。DMD rollout 要 4N+1 ≤ 109 帧条件，
直接用会在 rollout 末段训一个「世界突然重排」的输入。

做法（= 推理时代码世界的语义：画布 + 代码画的胶囊 + 没见过的地方 255）
---------------------------------------------------------------------
1. 地面画布：帧 0..80（同一 look）的静态像素（raw 0..6）逐像素射线 ∩ 水平面 Z = median(ground_z)，落到世界 XY 的
   CELL_CM 网格，后来的帧覆盖先前的（latest wins，帧 80 的视角优先）；角色像素与 255 不写。
2. 延长帧 81..108：每个目标像素射线 ∩ 同一平面 → 查画布；没写过的格子 ⇒ 255；再用 meta 里该帧的胶囊
   capsules.render_capsules 画角色（parts raw id，不做场景遮挡；v10 的遮挡 ≈1 %）。
3. 只给 ext_look_change 的 clip 做（v9 的 cond_ext 本身连续，照用）。

正交相机 + 平移时这一步是精确的（所有射线平行）；v10 的透视长焦下，直立物体（植被 / 石头）有与相机位移成正比的
视差误差，量级见 --report。输出 ids uint8 [28,H,W]（训练分辨率 = geometry / GC_RES，--root 的 res 必须与之相同；
832×480 ⇒ [28,480,832]），键与 cond_ext 相同（blocks / prompts 原样拷贝）。画布格 CELL_CM 与分辨率无关（世界 cm）。

usage: ext_canvas.py --root data/gc25d_v10 [--workers 64] [--clips id,id] [--report]
"""
from __future__ import annotations

import os

# 整个进程树单线程 BLAS（同 prep_v10.py）：要在 import numpy 之前设，并且盖掉 $ENV 里的 OMP_NUM_THREADS=8 ——
# 否则 64–160 个 fork worker 各开一个线程池（704p 实测 load 1850）。本脚本没有吃线程的大算子（逐 clip 的 numpy）。
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import capsules as CAP  # noqa: E402
import geometry as GEO  # noqa: E402

H, W = GEO.DST_H, GEO.DST_W
F_CLIP, F_EXT = 81, 28
CELL_CM = 6.0
CANVAS_STRIDE = 2            # 建画布用的帧步长（帧 80 一定包含）
STATIC_MAX = 6


def _ground_xy(m2c, w2v, z):
    ys, xs = np.mgrid[0:H, 0:W]
    return CAP.ground_points(m2c, w2v, xs.ravel(), ys.ravel(), z).reshape(H, W, 2)


def build_canvas(ids81: np.ndarray, meta, z: float):
    frames = sorted(set(range(0, F_CLIP, CANVAS_STRIDE)) | {F_CLIP - 1})
    pts, cls = [], []
    for f in frames:
        g = _ground_xy(meta["world_to_clip"][f], meta["world_to_view"][f], z)
        ok = (ids81[f] <= STATIC_MAX) & np.isfinite(g[..., 0])
        pts.append(g[ok]); cls.append(ids81[f][ok])
    allp = np.concatenate(pts)
    x0, y0 = np.floor(allp.min(0) / CELL_CM) * CELL_CM - CELL_CM
    nx = int(np.ceil((allp[:, 0].max() - x0) / CELL_CM)) + 2
    ny = int(np.ceil((allp[:, 1].max() - y0) / CELL_CM)) + 2
    canvas = np.full((ny, nx), 255, np.uint8)
    for p, c in zip(pts, cls):                      # 时间顺序写入 ⇒ latest wins
        ix = ((p[:, 0] - x0) / CELL_CM).astype(np.int64)
        iy = ((p[:, 1] - y0) / CELL_CM).astype(np.int64)
        canvas[iy, ix] = c
    # 1 格的洞（远处像素足迹 > 格子）用 3×3 内已知值补：只补四周至少 5 个已知的未知格
    unk = canvas == 255
    if unk.any():
        pad = np.pad(canvas, 1, constant_values=255)
        nb = np.stack([pad[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)])
        known = nb != 255
        fill = unk & (known.sum(0) >= 5)
        if fill.any():
            # 众数：对每个候选类计数
            best = np.full((ny, nx), 255, np.uint8); bestn = np.zeros((ny, nx), np.int16)
            for k in range(STATIC_MAX + 1):
                n = (nb == k).sum(0).astype(np.int16)
                w = n > bestn
                best[w], bestn[w] = k, n[w]
            canvas[fill] = best[fill]
    return canvas, (float(x0), float(y0))


def render_frame(canvas, origin, meta, f: int, z: float) -> np.ndarray:
    g = _ground_xy(meta["world_to_clip"][f], meta["world_to_view"][f], z)
    ix = np.floor((g[..., 0] - origin[0]) / CELL_CM)
    iy = np.floor((g[..., 1] - origin[1]) / CELL_CM)
    ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (iy >= 0) & (ix < canvas.shape[1]) & (iy < canvas.shape[0])
    out = np.full((H, W), 255, np.uint8)
    out[ok] = canvas[iy[ok].astype(np.int64), ix[ok].astype(np.int64)]
    caps = CAP.capsules_from_array(meta["capsules"][f])
    return CAP.render_capsules(out, meta["world_to_clip"][f], meta["world_to_view"][f], caps)


def render_clip(root: str, clip_id: str, report: bool = False) -> dict:
    t0 = time.time()
    ids81 = np.load(os.path.join(root, "cond", f"{clip_id}.npz"))["ids"]
    ext = np.load(os.path.join(root, "cond_ext", f"{clip_id}.npz"))
    meta = dict(np.load(os.path.join(root, "meta", f"{clip_id}.npz")))
    assert meta["world_to_clip"].shape[0] == F_CLIP + F_EXT, (clip_id, meta["world_to_clip"].shape)
    z = float(np.median(meta["ground_z"][:F_CLIP]))
    canvas, origin = build_canvas(ids81, meta, z)
    out = np.stack([render_frame(canvas, origin, meta, F_CLIP + j, z) for j in range(F_EXT)])
    res = {"clip_id": clip_id, "unknown_frac": float((out == 255).mean()), "sec": round(time.time() - t0, 2)}
    if report:
        # 自检：用同一画布重渲染帧 80（建画布的帧之一）与真 ids 的静态一致率；帧 81 与 UE ext 帧 0 的一致率（look 已换，仅参考）
        r80 = render_frame(canvas, origin, meta, F_CLIP - 1, z)
        st = ids81[F_CLIP - 1] <= STATIC_MAX
        res["self_agree_f80"] = float((r80[st] == ids81[F_CLIP - 1][st]).mean())
        r76 = render_frame(canvas, origin, meta, F_CLIP - 5, z)
        st76 = ids81[F_CLIP - 5] <= STATIC_MAX
        res["agree_f76"] = float((r76[st76] == ids81[F_CLIP - 5][st76]).mean())
    dst = os.path.join(root, "cond_ext_canvas", f"{clip_id}.npz")
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, ids=out, blocks=ext["blocks"], prompts=ext["prompts"])
    os.replace(tmp, dst)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--clips", default="")
    ap.add_argument("--all_ext", action="store_true", help="也给没有 look 切换的 clip 渲染（对拍用）")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    GEO.check_root_res(a.root, "--root")             # 数据根的 res 必须 = GC_RES（画布按 H×W 渲染）
    os.makedirs(os.path.join(a.root, "cond_ext_canvas"), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(a.root, "index", "clips.jsonl"))]
    if a.clips:
        want = set(a.clips.split(","))
        todo = [r["clip_id"] for r in rows if r["clip_id"] in want]
    else:
        todo = [r["clip_id"] for r in rows if r.get("has_ext") and (a.all_ext or r.get("ext_look_change"))
                and not os.path.exists(os.path.join(a.root, "cond_ext_canvas", f"{r['clip_id']}.npz"))]
    print(f"ext_canvas: {len(todo)} clips → {a.root}/cond_ext_canvas（{GEO.RES_TAG}）", flush=True)
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(render_clip, [a.root] * len(todo), todo, [a.report] * len(todo), chunksize=1)):
            out.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}  unknown {np.mean([x['unknown_frac'] for x in out]):.3f}", flush=True)
    rep = {"n": len(out), "res": GEO.RES_TAG, "unknown_frac_mean": float(np.mean([x["unknown_frac"] for x in out])) if out else None}
    if a.report and out:
        rep["self_agree_f80_mean"] = float(np.mean([x["self_agree_f80"] for x in out]))
        rep["agree_f76_mean"] = float(np.mean([x["agree_f76"] for x in out]))
    print(json.dumps(rep, indent=1))
    with open(os.path.join(a.root, "index", "ext_canvas_report.json"), "w") as fh:
        json.dump({"summary": rep, "clips": out}, fh)


if __name__ == "__main__":
    sys.exit(main())
