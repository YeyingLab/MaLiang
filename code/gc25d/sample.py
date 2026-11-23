#!/usr/bin/env python3
"""gc25d 离线评测（SFT 双向模型 / 基座）：val clip 的首帧 latent + parts id 视频 + 逐 cell 文本 ⇒ 81 帧，对原始 GT 算 PSNR。

改自 infer.py（演示版推理）的 clip 评测路径，适配 maliang 的数据布局：
  * 采样直接 import infer.sample（TI2V：FlowMatchScheduler("Wan")、帧 0 每步写回 z0、组合式 CFG），前向 = model.gc_forward；
  * z0   = latents/<clip>.pt 的 cell 0（因果 VAE ⇒ 等于单独编码帧 0）；
  * ids  = cond/<clip>.npz 的 parts raw id；模型是 ids 词表 ⇒ 先 model.collapse_parts；--sem_mode unknown ⇒ 全 255；
  * GT   = 原始 tar（<raw_root>/<subset>/<scene>/<run>/frames-*.tar）里帧 [start, start+81) 的 jpg → geometry.resample_rgb
           （不是 VAE 重建；infer.read_raw_frames 读，按 index/clips.jsonl 的 subset / scene / run / start 定位）；
           另把缓存 latent 解码一遍当"VAE 重建上限"（recon_ceiling，--no_recon_ceiling 关掉）；
  * 文本 = <data_root>/text_table_gc25d.pt 查表，context_ids = 排序下标（与 dataset.key_to_id / infer._KEY_TO_ID 一致）；
  * 模型 = --ckpt（model.load_gc_weights，词表按 ckpt 自动定）或 --base（基座 + 零初始化 gc_sem_cond ⇒ 等于基座 TI2V）。
infer.load_models 不能直接用：它先按默认 "ids" 词表 install_sem_cond，parts ckpt 会报词表不符。

分辨率 = geometry（env GC_RES，默认 832x480）：--data_root 的 index/prep_args.json "res" 与 --ckpt metadata 的 "res"
（缺失 ⇒ 832x480）都必须等于 GC_RES，否则加载前报错。

输出（--out_dir）：<clip>.mp4（GT | generated | cond 色图：静态类沿用 geometry.PALETTE，玩家 / Boss 的 6 个部位各一色，
255 灰色斜纹）、<clip>.json（逐帧 PSNR / SSIM、设置、耗时）、metrics.json（汇总：帧 1..80 的 PSNR 均值按 clip 平均）。

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python code/gc25d/sample.py --ckpt outputs/sft_v10/step-20000.safetensors \
        --val_n 8 --out_dir outputs/sft_v10/eval_step-20000
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python code/gc25d/sample.py --base --clip_ids <id> --steps 8 --out_dir outputs/...
用户模式（自带首帧 / ids / 文本，无 GT）见 sample_user.py：与这里共用 model_ids / generate（同输入同种子逐位相同；
进程里先碰 VAE 再采样的调用方须先 warmup，见该函数）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                    # maliang 根目录
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
os.environ.setdefault("AROPE_ROOT", os.path.join(ROOT, "code", "arope"))   # infer.py 的默认值指向旧机器路径
os.environ.setdefault("TMPDIR", os.path.join(ROOT, ".tmp"))
for _p in (HERE, os.environ["AROPE_ROOT"]):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import geometry as GEO  # noqa: E402
import infer as INF  # noqa: E402
import metrics as MET  # noqa: E402
import model as GM  # noqa: E402
import text as TXT  # noqa: E402
import viz as VIZ  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "Wan2.2-TI2V-5B")
DATA_ROOT = os.path.join(ROOT, "data", "gc25d_v10_fxcond")
RAW_ROOT = os.path.join(ROOT, "data", "recordings", "data")
F_CLIP, N_CELLS = 81, 21

# parts 词表的部位色（core 10 / 11 沿用 geometry.PALETTE 的品红 / 红）：玩家偏紫粉、Boss 偏红橙，武器各给一个冷色
PART_COLORS = {
    12: (255, 150, 255), 13: (150, 60, 255), 14: (255, 70, 160), 15: (110, 40, 190), 16: (0, 255, 255),
    17: (255, 150, 110), 18: (255, 120, 0), 19: (200, 0, 70), 20: (130, 20, 20), 21: (0, 90, 255),
    22: (255, 90, 30), 23: (255, 0, 0), 24: (255, 230, 80), 25: (120, 255, 60),
}
PARTS_PALETTE = GEO.PALETTE.copy()
for _raw, _c in PART_COLORS.items():
    PARTS_PALETTE[_raw] = _c


def colorize_parts(ids: np.ndarray) -> np.ndarray:
    """uint8 [..., H, W] → RGB：geometry.colorize_ids（静态类 + 255 斜纹）再把 12..21 的部位换成 PART_COLORS。"""
    out = GEO.colorize_ids(ids)
    m = (ids >= 12) & (ids <= 25)
    if m.any():
        out[m] = PARTS_PALETTE[ids[m]]
    return out


class PartsSideBySide(VIZ.SideBySide):
    """viz.SideBySide，只把 ids 面板换成 colorize_parts（保留面板分隔线）。"""

    def frame(self, i: int) -> np.ndarray:
        out = super().frame(i)
        x0 = (self.gen_index + 1) * VIZ.PANEL_W
        out[VIZ.HEADER_H:, x0 + 1:x0 + VIZ.PANEL_W] = colorize_parts(self.ids[i])[:, 1:]
        return out


# ------------------------------------------------------------------ 数据 ----
def clip_index(data_root: str) -> dict[str, dict]:
    return {r["clip_id"]: r for r in INF.read_jsonl(os.path.join(data_root, "index", "clips.jsonl"))}


def select_val_clips(index: dict[str, dict], n: int) -> list[str]:
    """val 且 drop=false 的 clip 里确定性地挑 n 条：按 run 轮流分配名额，run 内等距取。"""
    by_run: dict[str, list[str]] = {}
    for cid, r in index.items():
        if r["split"] == "val" and not r.get("drop"):
            by_run.setdefault(r["run"], []).append(cid)
    if not by_run:
        raise RuntimeError("index/clips.jsonl 里没有可用的 val clip")
    runs = sorted(by_run)
    for r in runs:
        by_run[r].sort()
    n = min(n, sum(len(v) for v in by_run.values()))
    quota, left, k = {r: 0 for r in runs}, n, 0
    while left > 0:
        r = runs[k % len(runs)]
        if quota[r] < len(by_run[r]):
            quota[r] += 1
            left -= 1
        k += 1
    out = []
    for r in runs:
        L, m = by_run[r], quota[r]
        out += [L[int((j + 0.5) * len(L) / m)] for j in range(m)]
    return out


def load_clip(data_root: str, cid: str) -> dict:
    d = torch.load(os.path.join(data_root, "latents", cid + ".pt"), map_location="cpu", weights_only=False)
    lat = d["latent"]
    assert lat.dtype == torch.bfloat16 and tuple(lat.shape) == (48, N_CELLS, GEO.LAT_H, GEO.LAT_W), (lat.dtype, tuple(lat.shape))
    prompts = list(d["prompts"])
    assert len(prompts) == N_CELLS, len(prompts)
    blocks = d["actions_block"]
    blocks = np.asarray(blocks.numpy() if torch.is_tensor(blocks) else blocks).astype(np.uint8)
    with np.load(os.path.join(data_root, "cond", cid + ".npz"), allow_pickle=False) as z:
        ids = z["ids"]
    assert ids.dtype == np.uint8 and ids.shape == (F_CLIP, GEO.DST_H, GEO.DST_W), (ids.dtype, ids.shape)
    bad = sorted(set(np.unique(ids).tolist()) - GM.valid_raw_ids("parts_fx"))
    assert not bad, f"{cid}: ids 含 parts_fx 词表外的值 {bad}"
    return {"latent": lat, "prompts": prompts, "blocks": blocks, "ids": ids}


def gt_frames(raw_root: str, row: dict) -> np.ndarray:
    raw = INF.read_raw_frames(row["scene"], row["run"], int(row["start"]), F_CLIP, "jpg",
                              raw_root=os.path.join(raw_root, row["subset"]))
    return np.stack([GEO.resample_rgb(f) for f in raw], 0)


class TextTable:
    """text_table_gc25d.pt 查表；context_ids = 排序下标（与训练一致）。"""

    def __init__(self, path: str):
        self.table = torch.load(path, map_location="cpu", weights_only=False)
        self.key_to_id = {k: i for i, k in enumerate(sorted(self.table))}

    def context(self, prompts: list[str], device):
        missing = sorted({p for p in prompts if p not in self.table})
        if missing:
            raise KeyError(f"文本表里没有 {missing[:3]}（gc25d 的 19 条串应该都在）")
        ctx = torch.stack([self.table[p] for p in prompts], 0).to(torch.bfloat16)
        ids = torch.tensor([self.key_to_id[p] for p in prompts], dtype=torch.int64)
        return ctx.unsqueeze(0).to(device), ids.unsqueeze(0).to(device)


# ------------------------------------------------------------------ 模型 ----
def load_dit(ckpt: str | None, base: bool, vocab: str, model_dir: str, device: str):
    if bool(ckpt) == bool(base):
        raise ValueError("必须二选一：--ckpt <safetensors> 或 --base")
    if ckpt:
        GEO.check_ckpt_res(ckpt, "--ckpt")                  # metadata res 必须 = GC_RES（缺失 ⇒ 832x480）
    t0 = time.time()
    dit = GM.load_dit(model_dir, device=device)
    if ckpt:
        info = GM.load_gc_weights(dit, ckpt)
        info = {k: v for k, v in info.items() if k != "missing"} | {"n_missing": len(info["missing"])}
    else:
        GM.install_sem_cond(dit, vocab)
        info = {"vocab": vocab, "has_sem_cond": False}
    dit = dit.to(device).eval().requires_grad_(False)
    dit.freqs = tuple(t.to(device) for t in dit.freqs)
    info.update(ckpt=ckpt, base=bool(base), model_dir=model_dir, res=GEO.RES_TAG, dit_load_sec=time.time() - t0,
                sem_cond_zero=getattr(dit, GM.SEM_COND_NAME).is_zero_output())
    return dit, info


# ------------------------------------------------------------------ 采样核心（clip 模式 / sample_user.py 共用）----
def model_ids(ids: np.ndarray, vocab: str, sem_mode: str) -> np.ndarray:
    """parts raw ids → 模型实际收到的条件：ids 词表的模型先 collapse_parts；sem_mode=unknown ⇒ 全 255。"""
    out = GM.collapse_parts(ids) if vocab == "ids" else ids
    return np.full_like(out, GM.UNKNOWN_ID) if sem_mode == "unknown" else out


def generate(args, dit, tt: TextTable, z0: torch.Tensor, ids_model: np.ndarray, prompts: list[str]):
    """z0 [1,48,1,H/16,W/16] + 条件 + 21 条 prompt → (latents [1,48,21,H/16,W/16], 采样秒数)。两种模式走同一段代码 ⇒ 同输入同种子逐位相同。
    用 args.device / steps / shift / cfg / sem_cfg / seed。"""
    dev = args.device
    context, context_ids = tt.context(prompts, dev)
    neg_context = neg_ids = None
    if args.cfg != 1.0:
        neg_context, neg_ids = tt.context([TXT.NULL_PROMPT] * N_CELLS, dev)
    sem = torch.from_numpy(np.ascontiguousarray(ids_model)).unsqueeze(0).to(dev)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    latents = INF.sample(dit, z0, context, context_ids, sem, steps=args.steps, shift=args.shift, cfg=args.cfg,
                         sem_cfg=args.sem_cfg, seed=args.seed, neg_context=neg_context, neg_context_ids=neg_ids,
                         forward=GM.gc_forward)
    torch.cuda.synchronize()
    return latents, time.time() - t0


@torch.no_grad()
def warmup(args, dit, tt: TextTable, ids_model: np.ndarray, prompts: list[str]) -> float:
    """在本进程跑任何 VAE 运算之前，用真实形状空跑一次 DiT 前向，返回秒数（clip 模式不调用）。

    默认 SDPA 后端首次调用时建图并缓存，建出的图与进程里之前跑过的 cuDNN 运算有关：
    实测先做一次 fp32 VAE 编码再采样，同输入下第 0 个 block 的自注意力输出就不同（flash / mem-efficient 后端不受影响），
    50 步后 latents 最大差 0.035。clip 模式第一次 DiT 前向之前没跑过 VAE；要先编码首帧的调用方（sample_user.py）
    先调这个，之后的 generate 才与 clip 模式逐位相同。交叉注意力的形状恒为 [21, 512, ·]（逐 cell gather），一次就够。"""
    t0 = time.time()
    context, context_ids = tt.context(prompts, args.device)
    sem = torch.from_numpy(np.ascontiguousarray(ids_model)).unsqueeze(0).to(args.device)
    lat = torch.zeros((1, 48, N_CELLS, GEO.LAT_H, GEO.LAT_W), dtype=torch.bfloat16, device=args.device)
    GM.gc_forward(dit, lat, torch.tensor([1000.0], dtype=torch.bfloat16, device=args.device), context, sem,
                  first_frame_cond=True, context_ids=context_ids)
    torch.cuda.synchronize()
    return time.time() - t0


def panel_names(args) -> tuple[str, str]:
    tag = os.path.basename(args.ckpt) if args.ckpt else "base"
    gen_name = f"generated [{tag}  sem={args.sem_mode} cfg={args.cfg:g} sem_cfg={args.sem_cfg:g} {args.steps} steps seed={args.seed}]"
    ids_name = "cond = model input (parts coloured)" if args.sem_mode == "cond" else "cond = all 255 (sem_mode=unknown)"
    return gen_name, ids_name


# ------------------------------------------------------------------ 一条 clip ----
def run_clip(args, dit, vae_pipe, tt: TextTable, model_info: dict, cid: str, row: dict) -> dict:
    dev = args.device
    c = load_clip(args.data_root, cid)
    gt = gt_frames(args.raw_root, row)
    vocab = model_info["vocab"]
    ids_model = model_ids(c["ids"], vocab, args.sem_mode)
    z0 = c["latent"][:, 0:1].unsqueeze(0).to(dev)
    latents, sample_sec = generate(args, dit, tt, z0, ids_model, c["prompts"])
    if getattr(args, "save_latents", False):
        torch.save(latents.cpu(), os.path.join(args.out_dir, cid + "_latents.pt"))
    frames = INF.decode_video(vae_pipe, latents)
    info = {"clip_id": cid, "split": row.get("split"), "subset": row.get("subset"), "scene": row.get("scene"),
            "run": row.get("run"), "start": row.get("start"), "vocab": vocab, "sem_mode": args.sem_mode,
            "steps": args.steps, "shift": args.shift, "cfg": args.cfg, "sem_cfg": args.sem_cfg, "seed": args.seed,
            "prompt_runs": INF.prompt_runs(c["prompts"]), "sample_sec": sample_sec,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "model": model_info}
    info["metrics"] = MET.clip_metrics(frames, gt, with_ssim=not args.no_ssim)
    if args.recon_ceiling:
        recon = INF.decode_video(vae_pipe, c["latent"].unsqueeze(0).to(dev))
        rc = MET.clip_metrics(recon, gt, with_ssim=not args.no_ssim)
        info["recon_ceiling"] = {k: v for k, v in rc.items() if k not in ("psnr", "ssim")}
    m = info["metrics"]
    print(f"   {cid}: PSNR 帧0 {m['psnr_frame0']:.2f} · 帧1–80 均值 {m['psnr_mean']:.2f} dB"
          + (f" · SSIM {m['ssim_mean']:.4f}" if "ssim_mean" in m else "")
          + (f"（VAE 上限 {info['recon_ceiling']['psnr_mean']:.2f}）" if "recon_ceiling" in info else "")
          + f" · 采样 {sample_sec:.1f}s", flush=True)

    out_mp4 = os.path.join(args.out_dir, cid + ".mp4")
    gen_name, ids_name = panel_names(args)
    # ids 面板画模型实际收到的条件（ids 词表的模型 ⇒ 折叠后的 10 / 11，unknown ⇒ 全 255 斜纹）
    sbs = PartsSideBySide(frames, ids_model, gt=gt, prompts=c["prompts"], blocks=c["blocks"], title=cid,
                          gen_name=gen_name, ids_name=ids_name)
    VIZ.write_mp4((sbs.frame(i) for i in range(sbs.n_frames)), out_mp4)
    info["out"] = out_mp4
    with open(os.path.join(args.out_dir, cid + ".json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, default=INF._json_default)
    return info


# ------------------------------------------------------------------ CLI ----
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="gc25d 离线评测（SFT / 基座，val clip vs 原始 GT）")
    ap.add_argument("--ckpt", default=None, help="train.py 导出的 step-N.safetensors（831 键，词表自动识别）")
    ap.add_argument("--base", action="store_true", help="基座 Wan2.2-TI2V-5B + 零初始化 gc_sem_cond")
    ap.add_argument("--vocab", choices=tuple(GM.VOCABS), default="parts", help="只对 --base 有意义（--ckpt 按 ckpt 定）")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--raw_root", default=RAW_ROOT, help="recordings/data（下面是 <subset>/<scene>/<run>/frames-*.tar）")
    ap.add_argument("--text_table", default=None, help="默认 <data_root>/text_table_gc25d.pt")
    ap.add_argument("--clip_ids", default=None, help="逗号分隔，或 @文件（每行一个）")
    ap.add_argument("--val_n", type=int, default=0, help="确定性地挑 N 条 val clip（drop=false，按 run 均分）")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--cfg", type=float, default=1.0, help="文本 CFG（负支 NULL_PROMPT × 21）")
    ap.add_argument("--sem_cfg", type=float, default=1.0, help="语义 CFG（负支 ids 全 255）")
    ap.add_argument("--sem_mode", choices=("cond", "unknown"), default="cond")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_ssim", action="store_true")
    ap.add_argument("--no_recon_ceiling", dest="recon_ceiling", action="store_false")
    ap.add_argument("--save_latents", action="store_true", help="另存 <clip>_latents.pt（采样结果 bf16 [1,48,21,H/16,W/16]，复现核对用）")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv=None) -> dict:
    ap = build_parser()
    args = ap.parse_args(argv)
    if bool(args.ckpt) == bool(args.base):
        ap.error("必须二选一：--ckpt 或 --base")
    if bool(args.clip_ids) == bool(args.val_n):
        ap.error("必须二选一：--clip_ids 或 --val_n")
    if args.sem_mode != "cond" and args.sem_cfg != 1.0:
        ap.error("--sem_cfg 只在 --sem_mode cond 下有意义")
    t_all = time.time()
    GEO.check_root_res(args.data_root, "--data_root")       # 数据根的 res 必须 = GC_RES（加载模型之前）
    index = clip_index(args.data_root)
    if args.clip_ids:
        if args.clip_ids.startswith("@"):
            with open(args.clip_ids[1:], encoding="utf-8") as fh:
                clips = [ln.strip() for ln in fh if ln.strip()]
        else:
            clips = [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    else:
        clips = select_val_clips(index, args.val_n)
    unknown = [c for c in clips if c not in index]
    if unknown:
        raise KeyError(f"{len(unknown)} 条 clip 不在 {args.data_root}/index/clips.jsonl：{unknown[:3]}")
    # 加载模型（~1 min）之前先查齐文件：latent / cond / 原始 run 目录
    for cid in clips:
        r = index[cid]
        for p in (os.path.join(args.data_root, "latents", cid + ".pt"), os.path.join(args.data_root, "cond", cid + ".npz"),
                  os.path.join(args.raw_root, r["subset"], r["scene"], r["run"])):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{cid}: 缺 {p}")
    not_val = [c for c in clips if index[c]["split"] != "val"]
    if not_val:
        print(f"   注意：{len(not_val)} / {len(clips)} 条不是 val clip：{not_val[:3]}")
    os.makedirs(args.out_dir, exist_ok=True)
    tt = TextTable(args.text_table or os.path.join(args.data_root, "text_table_gc25d.pt"))

    dit, model_info = load_dit(args.ckpt, args.base, args.vocab, args.model_dir, args.device)
    print(f"[model] {'ckpt ' + args.ckpt if args.ckpt else 'base'}：vocab={model_info['vocab']} "
          f"gc_sem_cond 零输出={model_info['sem_cond_zero']}（{model_info['dit_load_sec']:.0f}s）", flush=True)
    vae_pipe = INF.load_vae(args.device, args.model_dir)

    per_clip = []
    for n, cid in enumerate(clips):
        print(f"[{n + 1}/{len(clips)}] {cid}", flush=True)
        info = run_clip(args, dit, vae_pipe, tt, model_info, cid, index[cid])
        per_clip.append({"clip_id": cid, "split": info["split"], "run": info["run"], "scene": info["scene"],
                         **info["metrics"], "sample_sec": info["sample_sec"],
                         "recon_psnr_mean": info.get("recon_ceiling", {}).get("psnr_mean")})
    summary = {
        "config": {k: getattr(args, k) for k in ("ckpt", "base", "steps", "shift", "cfg", "sem_cfg", "sem_mode", "seed",
                                                  "data_root", "raw_root", "val_n")} | {"res": GEO.RES_TAG},
        "model": {k: v for k, v in model_info.items() if k != "metadata"},
        "ckpt_metadata": model_info.get("metadata"),
        "clip_ids": clips,
        "aggregate": MET.aggregate(per_clip),
        "recon_psnr_mean": (float(np.mean([c["recon_psnr_mean"] for c in per_clip]))
                            if all(c["recon_psnr_mean"] is not None for c in per_clip) else None),
        "per_clip": per_clip,
        "total_sec": time.time() - t_all,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=INF._json_default)
    agg = summary["aggregate"]
    print(f"DONE {len(clips)} 条：PSNR(帧1–80) {agg['psnr_mean']:.2f} dB"
          + (f" · SSIM {agg['ssim_mean']:.4f}" if "ssim_mean" in agg else "")
          + (f"（VAE 上限 {summary['recon_psnr_mean']:.2f}）" if summary["recon_psnr_mean"] is not None else "")
          + f" → {args.out_dir}/metrics.json · {summary['total_sec']:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    main()
