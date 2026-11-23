#!/usr/bin/env python3
"""gc25d 离线采样 · 用户模式：自带首帧 + parts id 视频 + 逐 cell 文本 ⇒ 81 帧 W×H（无 GT，不出 PSNR）。泛化测试（genbench）用。
W×H = geometry.DST_W × DST_H（env GC_RES，默认 832×480）；--ckpt 的 metadata
"res"（缺失 ⇒ 832x480）必须等于 GC_RES，--export_clip 的 --data_root 同理。

薄封装，复用 sample.py（load_dit / TextTable / model_ids / generate / PartsSideBySide）与 infer.py（load_image_rgb /
parse_actions / encode_first_frame / decode_video / prompt_runs）。采样核心 = sample.generate，与 clip 模式是同一段代码：
同首帧 latent + 同 ids + 同 prompts + 同种子 ⇒ latents 逐位相同。首帧用 fp32 VAE 单帧编码再转 bf16（infer.encode_first_frame），
与 prep 存的 latent cell 0 逐位相同（verify_prep 抽查 cell0_bf16_exact_frac = 1.0），所以 clip 自己的帧 0 png 能逐位复现 clip 模式。
注意顺序：先 sample.warmup（真实形状空跑一次 DiT 前向）再编码首帧 —— cuDNN attention 的图在首次调用时建，建图结果受之前
跑过的 VAE 运算影响（不 warmup 时实测 latents 最大差 0.035），warmup 后与 clip 模式逐位相同（outputs/genbench/usermode_test）。

输入
  --first_frame  png/jpg：W×H 原样；1280×704（源分辨率）→ geometry.resample_rgb；其它尺寸居中裁成 W:H 再 INTER_AREA。
  --ids          npz（'ids' 键或唯一数组）/ npy，uint8 [81,H,W] parts raw id，或源分辨率 [81,704,1280]（geometry.resample_ids）。
                 取值必须 ⊆ model.valid_raw_ids(ckpt 词表)（ids 词表的 ckpt 先 collapse_parts，与 clip 模式相同），否则报错。
  --prompts      json：21 条串（cell 0..20，须都在文本表里）；或
  --actions      "<spec>"（语法见 infer.py 模块文档，20 块，如 "D:10,A:10"）/ [20,5] 的 .npy → text.cell_prompts。
  --export_clip  <clip_id>：只导出该 clip 的 帧 0（W×H + 源 1280×704 png）/ ids npz / prompts json / 动作块 npy 到 --out，
                 不加载模型（L2 重绘首帧、复现测试的素材）。

输出（--out 目录，前缀 --name）：<name>.mp4（generated | cond 色图，--thumb 在生成面板右上角嵌输入首帧缩略图）、
<name>_gen.mp4（纯生成）、<name>_sheet.jpg（帧 0/20/40/60/80）、<name>.json（设置 / 输入 / 条件类别占比 / 耗时）、
--save_latents ⇒ <name>_latents.pt（bf16 [1,48,21,H/16,W/16]）。

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python code/gc25d/sample_user.py --ckpt outputs/sft_v10/step-8000.safetensors \
        --first_frame f0.png --ids ids.npz --actions "D:10,A:10" --out outputs/genbench/x --name d10a10 --thumb
    .venv/bin/python code/gc25d/sample_user.py --export_clip <clip_id> --out outputs/genbench/x
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time

import sample as S  # noqa: E402  （先 import：设好 sys.path / 环境变量）

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import geometry as GEO  # noqa: E402
import infer as INF  # noqa: E402
import model as GM  # noqa: E402
import text as TXT  # noqa: E402
import viz as VIZ  # noqa: E402

F_CLIP, N_CELLS, H, W = S.F_CLIP, S.N_CELLS, GEO.DST_H, GEO.DST_W
STATIC_NAMES = ("sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop")
CLASS_NAMES = {**dict(enumerate(STATIC_NAMES)),
               **{r: f"player_{n}" for r, n in zip(GM.PLAYER_PART_IDS, GM.PART_NAMES)},
               **{r: f"boss_{n}" for r, n in zip(GM.BOSS_PART_IDS, GM.PART_NAMES)}, GM.UNKNOWN_ID: "unknown"}
THUMB_W, THUMB_H = W // 4, H // 4


# ------------------------------------------------------------------ 输入 ----
def ckpt_vocab(path: str) -> str:
    """只读 safetensors 头（不碰权重数据）→ gc_sem_cond 的词表；加载模型前就能校验 ids。"""
    with open(path, "rb") as fh:
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    v = GM.vocab_from_state(hdr)
    if v is None:
        raise ValueError(f"{path} 里没有 gc_sem_cond（不是 gc25d 导出？）")
    return v


def load_first_frame(path: str) -> tuple[np.ndarray, dict]:
    with Image.open(path) as im:
        w, h = im.size
    mode = "as_is" if (h, w) == (H, W) else "resample_rgb" if (h, w) == (GEO.SRC_H, GEO.SRC_W) else "center_crop_area"
    return INF.load_image_rgb(path), {"path": os.path.abspath(path), "src_size": [w, h], "resize": mode}


def load_ids(path: str, vocab: str) -> tuple[np.ndarray, dict]:
    """→ (模型词表下的 raw ids uint8 [81,H,W], info)。取值严格 ⊆ valid_raw_ids(vocab)，不静默映射。"""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            if "ids" not in keys and len(keys) != 1:
                raise ValueError(f"{path}：npz 里没有 'ids' 键且不止一个数组：{keys}")
            arr = z["ids" if "ids" in keys else keys[0]]
    elif path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"--ids 应为 .npz / .npy，得到 {path}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{path}：ids 应为 uint8，得到 {arr.dtype}")
    info = {"path": os.path.abspath(path), "src_shape": list(arr.shape), "resampled": False}
    if arr.shape == (F_CLIP, GEO.SRC_H, GEO.SRC_W) and not GEO.IDENTITY:   # 1280×704：源 = 训练分辨率，不重采样
        arr, info["resampled"] = GEO.resample_ids(arr), True
    if arr.shape != (F_CLIP, H, W):
        raise ValueError(f"{path}：ids 应为 [81,{H},{W}]（或源分辨率 [81,704,1280]；{GEO.RES_ENV}={GEO.RES_TAG}），得到 {arr.shape}")
    if vocab == "ids":
        arr = GM.collapse_parts(arr)
    cnt = np.bincount(arr.ravel(), minlength=256)
    bad = {int(k): int(cnt[k]) for k in np.nonzero(cnt)[0] if int(k) not in GM.valid_raw_ids(vocab)}
    if bad:
        raise ValueError(f"{path}：ids 含 {vocab!r} 词表外的值 {{值: 像素数}} = {bad}"
                         f"（允许 {sorted(GM.valid_raw_ids(vocab))}；未知区域请写 255）")
    return np.ascontiguousarray(arr), info


def load_prompts(args, tt: S.TextTable) -> tuple[list[str], np.ndarray | None, dict]:
    """--prompts json（21 条）或 --actions → (prompts, cell 顺序动作块 [21,5] 或 None, info)。"""
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as fh:
            prompts = json.load(fh)
        if not (isinstance(prompts, list) and len(prompts) == N_CELLS and all(isinstance(p, str) for p in prompts)):
            raise ValueError(f"{args.prompts}：应为 {N_CELLS} 条字符串的 json 列表")
        blocks, info = None, {"source": "prompts_json", "path": os.path.abspath(args.prompts)}
    else:
        blocks20 = INF.parse_actions(args.actions)
        blocks, prompts = TXT.cell_blocks(blocks20), TXT.cell_prompts(blocks20)
        info = {"source": "actions", "spec": args.actions}
    missing = sorted({p for p in prompts if p not in tt.table})
    if missing:
        raise KeyError(f"文本表里没有 {missing[:3]}（只能用 text.py 模板的 18 条 + NULL_PROMPT）")
    return prompts, blocks, info


def cond_stats(ids: np.ndarray) -> dict:
    """条件（模型收到的）类别占比：全部帧 / 帧 0；UNKNOWN 逐帧占比的 min / mean / max。"""
    def frac(a):
        cnt = np.bincount(a.ravel(), minlength=256)
        return {CLASS_NAMES.get(int(k), str(int(k))): float(cnt[k]) / a.size for k in np.nonzero(cnt)[0]}
    unk = (ids == GM.UNKNOWN_ID).mean(axis=(1, 2))
    return {"class_frac": frac(ids), "class_frac_frame0": frac(ids[0]),
            "unknown_frac_per_frame": {"min": float(unk.min()), "mean": float(unk.mean()), "max": float(unk.max())}}


# ------------------------------------------------------------------ 导出 clip 素材 ----
def export_clip(args) -> dict:
    GEO.check_root_res(args.data_root, "--data_root")
    row = S.clip_index(args.data_root).get(args.export_clip)
    if row is None:
        raise KeyError(f"{args.export_clip} 不在 {args.data_root}/index/clips.jsonl")
    c = S.load_clip(args.data_root, args.export_clip)
    src0 = INF.read_raw_frames(row["scene"], row["run"], int(row["start"]), 1, "jpg",
                               raw_root=os.path.join(args.raw_root, row["subset"]))[0]
    os.makedirs(args.out, exist_ok=True)
    p = {k: os.path.join(args.out, f"{args.export_clip}_{k}") for k in
         ("frame0.png", "frame0_src.png", "ids.npz", "prompts.json", "actions.npy")}
    for key, img in (("frame0.png", GEO.resample_rgb(src0)), ("frame0_src.png", src0)):
        assert cv2.imwrite(p[key], cv2.cvtColor(img, cv2.COLOR_RGB2BGR)), p[key]
    np.savez_compressed(p["ids.npz"], ids=c["ids"])
    with open(p["prompts.json"], "w", encoding="utf-8") as fh:
        json.dump(c["prompts"], fh, ensure_ascii=False, indent=1)
    np.save(p["actions.npy"], c["blocks"][1:])                     # [20,5]，--actions 可直接吃
    print(f"导出 {args.export_clip}（split={row['split']}）→ " + ", ".join(os.path.basename(v) for v in p.values()))
    return p


# ------------------------------------------------------------------ 可视化 ----
class UserSideBySide(S.PartsSideBySide):
    """generated | cond（parts 着色）；thumb ≠ None ⇒ 生成面板右上角嵌输入首帧缩略图。"""

    def __init__(self, *a, thumb: np.ndarray | None = None, **kw):
        super().__init__(*a, **kw)
        self.thumb = thumb

    def frame(self, i: int) -> np.ndarray:
        out = super().frame(i)
        if self.thumb is not None:
            th, tw = self.thumb.shape[:2]
            x0, y0 = (self.gen_index + 1) * VIZ.PANEL_W - tw - 6, VIZ.HEADER_H + 6
            out[y0:y0 + th, x0:x0 + tw] = self.thumb
        return out


def make_thumb(img: np.ndarray) -> np.ndarray:
    t = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (THUMB_W - 1, THUMB_H - 1), VIZ.ACCENT, 1)
    VIZ._put(t, "input frame 0", 4, THUMB_H - 6, VIZ.ACCENT, 0.4)
    return t


# ------------------------------------------------------------------ CLI ----
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="gc25d 离线采样 · 用户模式（自带首帧 / ids / 文本）")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--vocab", choices=tuple(GM.VOCABS), default="parts", help="只对 --base 有意义（--ckpt 按 ckpt 定）")
    ap.add_argument("--model_dir", default=S.MODEL_DIR)
    ap.add_argument("--data_root", default=S.DATA_ROOT)
    ap.add_argument("--raw_root", default=S.RAW_ROOT, help="只有 --export_clip 用")
    ap.add_argument("--text_table", default=None, help="默认 <data_root>/text_table_gc25d.pt")
    ap.add_argument("--first_frame", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--prompts", default=None, help="json：21 条 cell prompt")
    ap.add_argument("--actions", default=None, help='动作脚本（如 "D:10,A:10"）或 [20,5] .npy')
    ap.add_argument("--export_clip", default=None, help="只导出某 clip 的帧 0 / ids / prompts / 动作块，然后退出")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--name", default=None, help="输出文件前缀（默认 = 首帧文件名）")
    ap.add_argument("--title", default=None, help="视频标题栏右侧文字（默认 = name）")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--cfg", type=float, default=1.0, help="文本 CFG（负支 NULL_PROMPT × 21）")
    ap.add_argument("--sem_cfg", type=float, default=1.0, help="语义 CFG（负支 ids 全 255）")
    ap.add_argument("--sem_mode", choices=("cond", "unknown"), default="cond")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thumb", action="store_true", help="生成面板右上角嵌输入首帧缩略图")
    ap.add_argument("--save_latents", action="store_true")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None) -> dict:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.export_clip:
        return export_clip(args)
    if bool(args.ckpt) == bool(args.base):
        ap.error("必须二选一：--ckpt 或 --base")
    if not (args.first_frame and args.ids):
        ap.error("用户模式需要 --first_frame 和 --ids")
    if bool(args.prompts) == bool(args.actions):
        ap.error("必须二选一：--prompts <json> 或 --actions <spec>")
    if args.sem_mode != "cond" and args.sem_cfg != 1.0:
        ap.error("--sem_cfg 只在 --sem_mode cond 下有意义")
    t_all = time.time()
    name = args.name or os.path.splitext(os.path.basename(args.first_frame))[0]
    stem = os.path.join(args.out, name)

    # 加载模型（~1 min）之前把输入全部校验完
    if args.ckpt:
        GEO.check_ckpt_res(args.ckpt, "--ckpt")             # metadata res 必须 = GC_RES（缺失 ⇒ 832x480）
    vocab = ckpt_vocab(args.ckpt) if args.ckpt else args.vocab
    tt = S.TextTable(args.text_table or os.path.join(args.data_root, "text_table_gc25d.pt"))
    img, ff_info = load_first_frame(args.first_frame)
    ids, ids_info = load_ids(args.ids, vocab)
    prompts, blocks, p_info = load_prompts(args, tt)
    ids_model = S.model_ids(ids, vocab, args.sem_mode)        # vocab=ids 时 load_ids 已折叠，这里只处理 unknown
    os.makedirs(args.out, exist_ok=True)
    print(f"[{name}] 首帧 {ff_info['src_size']} ({ff_info['resize']}) · ids {ids_info['src_shape']} · vocab={vocab} · "
          f"文本（{p_info['source']}）：\n   " + "\n   ".join(INF.prompt_runs(prompts)), flush=True)

    timing = {}
    dit, model_info = S.load_dit(args.ckpt, args.base, vocab, args.model_dir, args.device)
    assert model_info["vocab"] == vocab, (model_info["vocab"], vocab)
    t0 = time.time()
    vae_pipe = INF.load_vae(args.device, args.model_dir)
    timing.update(dit_load_sec=model_info["dit_load_sec"], vae_load_sec=time.time() - t0)
    print(f"[model] {args.ckpt or 'base'}：vocab={vocab}（DiT {timing['dit_load_sec']:.0f}s · VAE {timing['vae_load_sec']:.0f}s）",
          flush=True)

    timing["warmup_sec"] = S.warmup(args, dit, tt, ids_model, prompts)   # 必须在 VAE 编码之前（见 sample.warmup）
    t0 = time.time()
    z0 = INF.encode_first_frame(vae_pipe, img)                 # fp32 VAE 单帧 → bf16 [1,48,1,H/16,W/16]
    torch.cuda.synchronize()
    timing["encode_sec"] = time.time() - t0
    latents, timing["sample_sec"] = S.generate(args, dit, tt, z0, ids_model, prompts)
    if args.save_latents:
        torch.save(latents.cpu(), stem + "_latents.pt")
    t0 = time.time()
    frames = INF.decode_video(vae_pipe, latents)
    timing["decode_sec"] = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    t0 = time.time()
    gen_name, ids_name = S.panel_names(args)
    sbs = UserSideBySide(frames, ids_model, prompts=prompts, blocks=blocks, title=args.title or name,
                         gen_name=gen_name, ids_name=ids_name, thumb=make_thumb(img) if args.thumb else None)
    VIZ.write_mp4((sbs.frame(i) for i in range(sbs.n_frames)), stem + ".mp4")
    VIZ.write_video(stem + "_gen.mp4", frames)
    VIZ.contact_sheet(stem + "_sheet.jpg", sbs)
    timing["write_sec"] = time.time() - t0
    timing["total_sec"] = time.time() - t_all

    info = {"name": name, "mode": "user", "res": GEO.RES_TAG, "vocab": vocab, "sem_mode": args.sem_mode, "steps": args.steps, "shift": args.shift,
            "cfg": args.cfg, "sem_cfg": args.sem_cfg, "seed": args.seed,
            "inputs": {"first_frame": ff_info, "ids": ids_info, "text": p_info},
            "prompt_runs": INF.prompt_runs(prompts), "prompts": prompts,
            "actions_block": None if blocks is None else blocks.tolist(),
            "cond": cond_stats(ids_model), "ids_supplied": cond_stats(ids) if args.sem_mode != "cond" else "= cond",
            "timing": timing, "peak_mem_gb": peak, "model": model_info,
            "out": {"mp4": stem + ".mp4", "gen_mp4": stem + "_gen.mp4", "sheet": stem + "_sheet.jpg",
                    "latents": stem + "_latents.pt" if args.save_latents else None}}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, default=INF._json_default)
    print(f"DONE {stem}.mp4 · warmup {timing['warmup_sec']:.1f}s · 编码 {timing['encode_sec']:.1f}s · 采样 {timing['sample_sec']:.1f}s · 解码 {timing['decode_sec']:.1f}s"
          f" · 写 {timing['write_sec']:.1f}s · 总 {timing['total_sec']:.0f}s · 峰值显存 {peak:.1f} GB", flush=True)
    return info


if __name__ == "__main__":
    main()
