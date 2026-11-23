#!/usr/bin/env python3
"""蒸馏学生（stage2 / stage3 的 EMA）的部署孪生评测：长 rollout → mp4（GT | 生成 | 条件）+ PSNR / ZNCC / 亮度漂移 json。

采样路径与 codeworld/live.py（LiveGame.reset_ids + step_ids）逐行相同 —— 同一个 `_call`（逐 cell 文本 + 本 cell 语义帧
mask_in_k、first_frame_cond=False）、同一张 4 步 renoise 表（1000/750/500/250 吸附到训练网格）、同一个 CPU 生成器喂
初始噪声与重新加噪、sink 3 / window 17 / cap 20 的滑窗 KV、每个 cell 在 t=0 把 x0 提交一次。所以这里量到的就是部署时
会看到的东西（冒烟脚本 scripts/distill_smoke/live_path_check.py 核过：同种子下与 live.py 逐位相同）。
条件不是代码世界现算的，而是 val 的**真实**逐 cell 语义帧 + 动作 prompt，所以生成可以和 GT 比。两种取法：
  * 单 clip（默认）：--cells ≤ 28（> 21 时接 cond_ext 的 7 个 cell）。只到第一次驱逐之后 7 个 cell —— 短程。
  * 长链（--chain_cells C，如 241 = 60 s）：同一 run / episode / camera 的相邻 clip（start 间隔 96 帧 = 24 cell）
    首尾相接 —— clip i 的帧 0..80 + cond_ext 的帧 81..95，动作块同理（cond_ext 与下一条 clip 的重叠部分逐位相同）。
    这才量得到长程漂移（AR 的主要失效方式；gc25d_v8 的参考数是 60 s 链）。GT 指标只在与链首**同一个 look** 的
    clip 上算（v10_real 的 look 每条 clip 都换，链上只有第一条可比；drift 不依赖 GT）。
选 clip / 链默认**按 run 分层**：val run 按 subset 交错排，轮转取每个 run 的下一条，--n_clips ≤ run 数时每条来自不同 run；
summary.json 里另有逐 run / 逐 subset 的均值。

    CUDA_VISIBLE_DEVICES=6 $PY code/gc25d/gcdistill/eval_rollout.py \\
        --ckpt outputs/distill/gc25d_v10/stage3_dmd/ema/ema_step500.safetensors \\
        --chain_cells 241 --n_clips 10 --vae tae --out outputs/distill/eval/ema500

    --ckpt 拒收 /dits/ 路径（stage2 / stage3 的部署权重是 ema/；dits/ 只为续训存在）。
    分辨率 = geometry（env GC_RES，默认 832x480）：--ckpt metadata 的 res 与 --data_root 的 res 必须都等于 GC_RES。
指标（与 GT 共有的帧上，帧 0 是同一个 z0 的解码，不计）：
    psnr       逐帧 PSNR（dB）的均值；zncc 逐帧亮度图的零均值归一化互相关的均值；
    drift      生成的平均亮度（BT.601 luma，0..255）：最后一个 cell − cell 0（整条 rollout，含 GT 之外的 cell）；
    gt_luma_gap 生成 − GT 的平均亮度差（有 GT 的帧上）。逐 cell 的序列都写进 metrics.json。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GC_ROOT not in sys.path:
    sys.path.insert(0, GC_ROOT)

from gcdistill import data_gc as DG  # noqa: E402
from gcdistill import gc_dit as GD  # noqa: E402
from distill.arope_dit import MODEL_DIR, unwrap  # noqa: E402
from distill.flow_pf import DEPLOY_DENOISE_LIST, deploy_schedule  # noqa: E402

M_ROOT = DG.M_ROOT
TAEHV_DIR = os.path.join(M_ROOT, "code", "taehv")
TAEHV_W = os.path.join(M_ROOT, "assets", "taew2_2_super.pth")
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# ----------------------------------------------------------------------------
# 模型：live.py 的模型路径（GcDiT 因果 + load_gc_ckpt），词表按 ckpt 自动判断
# ----------------------------------------------------------------------------

def refuse_dits(path: str) -> None:
    if "/dits/" in os.path.abspath(path).replace(os.sep, "/"):
        raise SystemExit(f"--ckpt 指向 dits/：{path}\n  部署 / 评测用 ema/ema_step{{N}}.safetensors；dits/ 只为续训存在。")


def load_student(ckpt: str, device, *, base_init: str = "skeleton", model_dir: str = MODEL_DIR) -> GD.GcDiT:
    """= LiveGame.__init__ 的模型部分：GcDiT(causal=True) → load_gc_ckpt（strict 831）→ bf16，外加 vocab=ckpt_vocab(ckpt)。"""
    refuse_dits(ckpt)
    DG.check_res(ckpts=[("--ckpt", ckpt)])            # metadata res 必须 = GC_RES（缺失 ⇒ 832x480）
    vocab = GD.ckpt_vocab(ckpt)
    if vocab is None:
        raise SystemExit(f"{ckpt} 没有 gc_sem_cond 键：不是 gc25d 导出")
    dit = GD.empty_wan_dit() if base_init == "skeleton" else None
    m = GD.GcDiT(dit=dit, causal=True, model_dir=model_dir, vocab=vocab).to(device).eval().requires_grad_(False)
    m.move_rope_table(device)
    GD.load_gc_ckpt(m, ckpt)
    return m.to(torch.bfloat16)


# ----------------------------------------------------------------------------
# 部署 rollout（live.py 的逐 cell 回路）
# ----------------------------------------------------------------------------

@torch.no_grad()
def deploy_rollout(model, z0: torch.Tensor, ctx: torch.Tensor, ids: torch.Tensor, sem: torch.Tensor, n_gen: int, *,
                   denoise_list=DEPLOY_DENOISE_LIST, kv_window: int = 17, sink_size: int = 3, rope_cap: int = 20,
                   seed: int = 0, timing: list | None = None) -> torch.Tensor:
    """z0 bf16 [1,48,1,h,w]；ctx [1,1+N,L,4096]；ids [1,1+N]；sem uint8 [1,4N+1,H,W] → latents [1,48,1+N,h,w]。

    逐行对应 LiveGame：reset_ids（cell 0：z0 在 t=0 提交）→ step_ids × N（randn 初始噪声 → 4 步 x0 估计 + 重新加噪
    → t=0 提交 den）。噪声全部来自一个 CPU 生成器（seed），与 live.py 的 self.g 同一条随机流。"""
    dev = z0.device
    bare = unwrap(model)
    _, sig, t_grid = deploy_schedule([float(x) for x in denoise_list])
    DG.assert_rollout_aligned_gc(sem, ctx, ids, n_gen)
    g = torch.Generator("cpu").manual_seed(seed)
    cache = bare.init_kv_cache(kv_window=kv_window, sink_size=sink_size, rope_cap=rope_cap)
    ac = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda"))

    def _call(x, t, k, commit):
        return model(x, t, ctx[:, k:k + 1], context_ids=ids[:, k:k + 1], first_frame_cond=False, frame_index=k,
                     kv_cache=cache, commit=commit, mask_in_k=GD.cell_frames(sem, k))

    z0 = z0.to(dev, torch.bfloat16)
    b, c, _, lh, lw = z0.shape
    with ac:
        _call(z0, torch.zeros(b, device=dev), 0, True)
    out = [z0]
    for k in range(1, n_gen + 1):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        with ac:
            noisy = torch.randn(b, c, 1, lh, lw, generator=g).to(dtype=torch.bfloat16, device=dev)
            for i in range(len(t_grid)):
                v = _call(noisy, torch.full((b,), float(t_grid[i]), device=dev), k, False)
                den = noisy - float(sig[i]) * v
                if i + 1 < len(t_grid):
                    ns = float(sig[i + 1])
                    noisy = (1.0 - ns) * den + ns * torch.randn(den.shape, generator=g).to(den)
            _call(den, torch.zeros(b, device=dev), k, True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if timing is not None:
            timing.append((time.time() - t1) * 1e3)
        out.append(den)
    assert cache["frames_cached"] == n_gen + 1, cache["frames_cached"]
    bare.release_kv_cache(cache)
    return torch.cat(out, dim=2)


# ----------------------------------------------------------------------------
# 选哪些 clip / 链：按 run 分层
# ----------------------------------------------------------------------------

CLIP_STRIDE = 96                       # clip 起点间隔（start ∈ 96ℤ）= 24 个 cell
STRIDE_CELLS = CLIP_STRIDE // DG.CELL_FRAMES


def _run_key(r: dict) -> tuple:
    return r["subset"], r["scene"], r["run"]


def _run_order(rows: list[dict]) -> list[tuple]:
    """val run 的轮转顺序：subset 交错（v10_real / v9_stylized 交替），subset 内按名字。"""
    keys = sorted({_run_key(r) for r in rows})
    per_sub = [[k for k in keys if k[0] == s] for s in sorted({k[0] for k in keys})]
    return [k for grp in itertools.zip_longest(*per_sub) for k in grp if k is not None]


def stratified(items_by_run: dict, n: int) -> list:
    """{run_key: [候选（按 start 升序）]} → 轮转取 n 个：每轮每个 run 取它的下一个 ⇒ n ≤ run 数时来自 n 个不同 run。"""
    order = [k for k in _run_order([{"subset": k[0], "scene": k[1], "run": k[2]} for k in items_by_run])
             if items_by_run[k]]
    out, depth = [], 0
    while len(out) < n and any(depth < len(items_by_run[k]) for k in order):
        for k in order:
            if depth < len(items_by_run[k]) and len(out) < n:
                out.append(items_by_run[k][depth])
        depth += 1
    return out


def pick_clips(rows: list[dict], n: int) -> list[str]:
    """单 clip 模式：rows（数据集已按 has_ext / drop 过滤）按 run 分层取 n 条 clip_id。"""
    by_run: dict = {}
    for r in sorted(rows, key=lambda r: r["start"]):
        by_run.setdefault(_run_key(r), []).append(r["clip_id"])
    return stratified(by_run, n)


def find_chain(by_start: dict, s0: int, n_cells: int) -> list[dict] | None:
    """从 start=s0 起把同一 run / episode / camera 的相邻 clip 串成 n_cells 个 cell 的真实条件；凑不够 ⇒ None。
    clip i 管相对帧 96i … 96i+80（cond）与 96i+81 … 96i+95（cond_ext），相对动作块 24i … 24i+19（clip 的 cell 1..20）
    与 24i+20 … 24i+23（cond_ext 的前 4 个）。by_start：{start: row}，同一个 run，含 has_ext=False 的行。"""
    last_f = DG.CELL_FRAMES * (n_cells - 1)                      # 最后一帧（相对链首）
    last_b = n_cells - 2                                          # 最后一个 cell 的动作块
    n_clip = max(last_f // CLIP_STRIDE, last_b // STRIDE_CELLS) + 1
    first = by_start.get(s0)
    if first is None:
        return None
    chain = []
    for i in range(n_clip):
        r = by_start.get(s0 + CLIP_STRIDE * i)
        if (r is None or r.get("drop") or r["episode"] != first["episode"]
                or r["camera_variant"] != first["camera_variant"]):
            return None
        need_ext = (i < n_clip - 1 or last_f - CLIP_STRIDE * i > DG.CLIP_FRAMES - 1
                    or last_b - STRIDE_CELLS * i >= DG.CLIP_CELLS - 1)
        if need_ext and not r.get("has_ext"):
            return None
        chain.append(r)
    return chain


def pick_chains(rows_all: list[dict], n: int, n_cells: int) -> list[list[dict]]:
    """长链模式：每个 run 依 start 找互不重叠的合格链，再按 run 分层轮转取 n 条。"""
    runs: dict = {}
    for r in rows_all:
        runs.setdefault(_run_key(r), {})[int(r["start"])] = r
    by_run = {}
    for k, by_start in runs.items():
        cands, nxt = [], -1
        for s in sorted(by_start):
            if s < nxt:
                continue
            ch = find_chain(by_start, s, n_cells)
            if ch is not None:
                cands.append(ch)
                nxt = s + CLIP_STRIDE * len(ch)
        by_run[k] = cands
    return stratified(by_run, n)


def load_chain(root: str, chain: list[dict], n_cells: int, table: dict, kid: dict, vocab: str) -> dict:
    """把 find_chain 的链读成与 GcRolloutDataset.load 同形的样本（z0 = 链首 clip 的 cell 0）。另带 'clips'（逐 clip 的
    clip_id / 相对首帧 / look）供 GT 拼接。cond_ext 与下一条 clip 的重叠帧 / 动作逐位核对（数据不连续直接报错）。"""
    last_f = DG.CELL_FRAMES * (n_cells - 1)
    sem, prompts, clips, prev_ext = [], [], [], None
    for i, r in enumerate(chain):
        cid = r["clip_id"]
        lat = torch.load(os.path.join(root, "latents", f"{cid}.pt"), map_location="cpu", weights_only=False)
        ids = np.load(os.path.join(root, "cond", f"{cid}.npz"))["ids"]
        p = [str(x) for x in lat["prompts"]]
        if i == 0:
            z0 = lat["latent"][:, :1].to(torch.bfloat16).contiguous()
            prompts.append(p[0])
        if prev_ext is not None:                                  # 上一条的 cond_ext 帧 96..108 / 块 24..26 == 这一条的开头
            e_ids, e_p = prev_ext
            if not (np.array_equal(e_ids[15:28], ids[:13]) and e_p[4:7] == p[1:4]):
                raise RuntimeError(f"{chain[i - 1]['clip_id']} 的 cond_ext 与 {cid} 的开头对不上：链不连续")
        sem.append(ids)
        prompts += p[1:DG.CLIP_CELLS]
        clips.append({"clip_id": cid, "frame0": CLIP_STRIDE * i, "look": r.get("look")})
        prev_ext = None
        if r.get("has_ext"):
            z = np.load(os.path.join(root, "cond_ext", f"{cid}.npz"))
            e_p = [str(x) for x in z["prompts"]] if "prompts" in z.files else \
                [_block_prompt(b) for b in z["blocks"]]
            prev_ext = (z["ids"], e_p)
            sem.append(z["ids"][:CLIP_STRIDE - DG.CLIP_FRAMES])
            prompts += e_p[:STRIDE_CELLS - (DG.CLIP_CELLS - 1)]
    sem = np.concatenate(sem, 0)[:last_f + 1]
    prompts = prompts[:n_cells]
    assert sem.shape[0] == last_f + 1 and len(prompts) == n_cells, (sem.shape, len(prompts), n_cells)
    ctx, cids = DG.cell_context(prompts, table, kid)
    head = chain[0]["clip_id"]
    return {"z0": z0, "sem_ids": torch.from_numpy(np.ascontiguousarray(DG.to_vocab(sem, vocab))), "context": ctx,
            "context_ids": cids, "clip_id": f"{head}+chain{len(chain)}", "prompts": prompts, "clips": clips,
            "subset": chain[0]["subset"], "run": chain[0]["run"]}


def _block_prompt(block) -> str:
    import text as TXT
    return TXT.block_to_prompt(*map(int, block))


# ----------------------------------------------------------------------------
# 解码：Wan VAE（全量）/ TAEHV（live.py 的 TinyStreamVAE 同一份权重）
# ----------------------------------------------------------------------------

class Decoder:
    def __init__(self, kind: str, device, *, model_dir: str = MODEL_DIR, taehv_dir: str = TAEHV_DIR,
                 taehv_weights: str = TAEHV_W):
        self.kind, self.device = kind, device
        if kind == "wan":
            import infer as INF
            self.INF = INF
            self.pipe = INF.load_vae(str(device), model_dir)
        elif kind == "tae":
            if taehv_dir not in sys.path:
                sys.path.insert(0, taehv_dir)
            from taehv import TAEHV
            self.tae = TAEHV(taehv_weights).to(device, torch.float16).eval()
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def __call__(self, lat: torch.Tensor) -> np.ndarray:
        """latent [1,48,f,H/16,W/16] → uint8 [4f−3,H,W,3]。"""
        if self.kind == "wan":
            return self.INF.decode_video(self.pipe, lat)
        x = lat.to(self.device, torch.float16).permute(0, 2, 1, 3, 4)          # NCTHW → NTCHW（TAEHV 不做 mean/std 反缩放）
        v = self.tae.decode_video(x, parallel=x.shape[1] <= 32, show_progress_bar=False)[0]   # [T,3,H,W] ∈ [0,1]
        #   ↑ 长链（上百个 cell）走串行解码：parallel 会把整条时间轴的全分辨率特征一次摊开
        frames = (v.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        want = DG.frames_of(lat.shape[2])
        assert frames.shape[0] >= want, (frames.shape, want)
        return frames[-want:] if frames.shape[0] != want else frames


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------

def luma(frames: np.ndarray, chunk: int = 64) -> np.ndarray:
    """uint8 [F,H,W,3] → float32 [F,H,W]（BT.601）。分块算：长链上千帧时不一次性摊开 fp32 的 [F,H,W,3]。"""
    out = np.empty(frames.shape[:-1], np.float32)
    for i in range(0, len(frames), chunk):
        out[i:i + chunk] = (frames[i:i + chunk].astype(np.float32) * LUMA).sum(-1)
    return out


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).mean() * (b * b).mean()))
    return float((a * b).mean() / den) if den > 0 else float("nan")


def cell_means(x: np.ndarray) -> list:
    """逐帧标量 [T=4n−3] → 逐 cell 均值 [n]（cell 0 = 帧 0）；NaN 帧（没有 GT）不计，非有限值（帧 0 的 PSNR = inf、
    整个 cell 都没有 GT）记 None，json 才合法。"""
    n = (len(x) + 3) // 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # 全 NaN 的 cell
        vals = [float(x[0])] + [float(np.nanmean(x[4 * k - 3: 4 * k + 1])) for k in range(1, n)]
    return [v if np.isfinite(v) else None for v in vals]


def rollout_metrics(gen: np.ndarray, gt: np.ndarray | None, gt_valid: np.ndarray | None = None) -> dict:
    """gt：与 gen 帧对齐的前缀（单 clip），或与 gen 等长、gt_valid（bool [T]）标出哪些帧有可比的 GT（长链）。"""
    import metrics as MET
    y_gen = luma(gen).mean(axis=(1, 2))
    out = {"frames": int(len(gen)), "luma_gen_cell": cell_means(y_gen)}
    out["drift"] = float(out["luma_gen_cell"][-1] - out["luma_gen_cell"][0])
    if gt is not None:
        n = min(len(gen), len(gt))
        valid = np.ones(n, bool) if gt_valid is None else np.asarray(gt_valid[:n], bool).copy()
        valid[0] = False                                          # 帧 0 = 同一个 z0 的解码（gt_frames 仍把它算上）
        ps = np.full(n, np.nan)
        zn = np.full(n, np.nan)
        yt = np.full(n, np.nan)
        idx = np.nonzero(valid)[0]
        ps[idx] = MET.psnr_per_frame(gen[idx], gt[idx])
        yg_f, yt_f = luma(gen[idx]), luma(gt[idx])
        zn[idx] = [zncc(yg_f[i], yt_f[i]) for i in range(len(idx))]
        yt[idx] = yt_f.mean(axis=(1, 2))
        yt[0] = float(luma(gt[:1]).mean())
        ps[0] = np.inf
        out.update(gt_frames=int(len(idx)) + 1, psnr=float(np.mean(ps[idx])) if len(idx) else float("nan"),
                   zncc=float(np.nanmean(zn[idx])) if len(idx) else float("nan"),
                   psnr_cell=cell_means(ps), zncc_cell=cell_means(zn), luma_gt_cell=cell_means(yt),
                   gt_luma_gap=float(np.mean(y_gen[:n][idx] - yt[idx])) if len(idx) else float("nan"))
    return out


# ----------------------------------------------------------------------------
# 视频：GT | generated | cond（GT 只有 81 帧，之后画黑）
# ----------------------------------------------------------------------------

def side_by_side(gen: np.ndarray, gt: np.ndarray | None, sem: np.ndarray, prompts: list[str], title: str,
                 gt_valid: np.ndarray | None = None):
    import viz as VIZ
    H, W, HDR = VIZ.PANEL_H, VIZ.PANEL_W, VIZ.HEADER_H
    names = ["GT", "generated (deploy rollout)", "cond (sem ids)"]
    for i in range(len(gen)):
        k = 0 if i == 0 else (i + 3) // 4
        has = gt is not None and i < len(gt) and (gt_valid is None or bool(gt_valid[i]))
        g_panel = gt[i] if has else np.zeros((H, W, 3), np.uint8)
        out = np.empty((H + HDR, 3 * W, 3), np.uint8)
        out[:HDR] = VIZ.BG
        out[HDR:] = np.concatenate([g_panel, gen[i], VIZ.colorize(sem[i])], axis=1)
        for j, nm in enumerate(names):
            VIZ._put(out, nm if (j or has) else "GT (none here: clip gap / other look)", j * W + 8, 18,
                     VIZ.ACCENT if j == 1 else VIZ.FG)
            if j:
                out[:, j * W:j * W + 1] = VIZ.BG
        VIZ._put(out, f"frame {i}/{len(gen) - 1} | cell {k}{' (past clip 0)' if k >= DG.CLIP_CELLS else ''} | "
                      f"{VIZ.motion_text(prompts[k])}   {title}", 8, 40, VIZ.FG)
        yield out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="部署孪生长 rollout 评测（gc25d 蒸馏学生）")
    ap.add_argument("--ckpt", required=True, help="stage2 / stage3 的 ema/ema_step{N}.safetensors（拒收 /dits/）")
    ap.add_argument("--data_root", default=DG.DATA_ROOT)
    ap.add_argument("--text_table", default=None, help="None ⇒ <data_root>/text_table_gc25d.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clip_ids", default=None,
                    help="逗号分隔（长链模式下是链首 clip）；不给 ⇒ 按 run 分层取 --n_clips 条")
    ap.add_argument("--n_clips", type=int, default=1, help="clip / 链的条数；≤ val run 数时每条来自不同 run")
    ap.add_argument("--cells", type=int, default=25, help="单 clip 模式 rollout 的 cell 数（含 cell 0）；>21 接 cond_ext，≤ 28")
    ap.add_argument("--chain_cells", type=int, default=0,
                    help=">0 ⇒ 长链模式：相邻 clip 首尾相接成这么多 cell（241 = 60 s）；忽略 --cells")
    ap.add_argument("--denoise_list", default="1000,750,500,250")
    ap.add_argument("--kv_window", type=int, default=17)
    ap.add_argument("--sink_size", type=int, default=3)
    ap.add_argument("--rope_cap", type=int, default=20)
    ap.add_argument("--vae", choices=("wan", "tae"), default="wan")
    ap.add_argument("--taehv_dir", default=TAEHV_DIR)
    ap.add_argument("--taehv_weights", default=TAEHV_W)
    ap.add_argument("--base_init", choices=("skeleton", "wan"), default="skeleton")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--out", required=True, help="输出目录（绝对路径，local disk）")
    a = ap.parse_args(argv)
    refuse_dits(a.ckpt)
    if a.rope_cap != a.sink_size + a.kv_window:
        raise SystemExit(f"rope_cap({a.rope_cap}) 必须 == sink({a.sink_size}) + window({a.kv_window})")
    if a.chain_cells:
        if a.chain_cells < 2:
            raise SystemExit(f"--chain_cells 应 ≥ 2，得到 {a.chain_cells}")
    elif not 2 <= a.cells <= DG.ROLLOUT_CELLS:
        raise SystemExit(f"--cells 应在 [2, {DG.ROLLOUT_CELLS}]（更长用 --chain_cells），得到 {a.cells}")
    if not os.path.isabs(a.out):
        raise SystemExit(f"--out must be an absolute path, got {a.out}")
    if a.text_table is None:
        a.text_table = os.path.join(a.data_root, "text_table_gc25d.pt")
    return a


def assemble_gt(dec, root: str, clips: list[dict], n_frames: int, device) -> tuple[np.ndarray | None, np.ndarray]:
    """逐 clip 解码 GT latent，放到它在 rollout 里的相对帧上。只取与第一条 clip **同一个 look** 的（v10_real 每条
    clip 换 look，换了之后 GT 与 z0 的外观不可比）。返回 (gt uint8 [n_frames,H,W,3] 或 None，valid bool [n_frames])。"""
    gt, valid = None, np.zeros(n_frames, bool)
    for c in clips:
        f0 = int(c["frame0"])
        if f0 >= n_frames or c.get("look") != clips[0].get("look"):
            continue
        lat = torch.load(os.path.join(root, "latents", f"{c['clip_id']}.pt"), map_location="cpu",
                         weights_only=False)["latent"][None].to(device)
        fr = dec(lat)
        if gt is None:
            gt = np.zeros((n_frames,) + fr.shape[1:], np.uint8)
        m = min(len(fr), n_frames - f0)
        gt[f0:f0 + m], valid[f0:f0 + m] = fr[:m], True
    return gt, valid


def _group_means(rows: list[dict], key) -> dict:
    keys = ("psnr", "zncc", "drift", "abs_drift", "gt_luma_gap")
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {g: {"n": len(v), **{k: float(np.nanmean([m[k] for m in v])) for k in keys}} for g, v in out.items()}


def main(argv=None):
    a = parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DG.check_res(ckpts=[("--ckpt", a.ckpt)], root=a.data_root)    # 加载 5B 之前先查分辨率
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model = load_student(a.ckpt, dev, base_init=a.base_init, model_dir=a.model_dir)
    vocab = GD.GM.sem_vocab(model.dit)
    print(f"[model] {a.ckpt}（vocab={vocab}）{time.time() - t0:.0f}s", flush=True)
    rows_all = DG.read_index(a.data_root, a.split, keep_drop=True)
    row_of = {r["clip_id"]: r for r in rows_all}
    if a.chain_cells:
        n_cells = a.chain_cells
        table = DG.load_text_table(a.text_table)
        kid = DG.key_to_id(table)
        if a.clip_ids:
            chains = []
            for cid in a.clip_ids.split(","):
                r = row_of[cid]
                by_start = {int(x["start"]): x for x in rows_all if _run_key(x) == _run_key(r)}
                ch = find_chain(by_start, int(r["start"]), n_cells)
                if ch is None:
                    raise SystemExit(f"{cid} 起凑不出 {n_cells} cell 的连续链（同 episode / camera、has_ext、未 drop）")
                chains.append(ch)
        else:
            chains = pick_chains(rows_all, a.n_clips, n_cells)
        if not chains:
            raise SystemExit(f"{a.data_root} split={a.split} 里没有 {n_cells} cell 的连续链")
        items = [(ch[0]["clip_id"], lambda ch=ch: load_chain(a.data_root, ch, n_cells, table, kid, vocab)) for ch in chains]
        print(f"[data] 长链 {len(chains)} 条 × {n_cells} cell（{DG.frames_of(n_cells)} 帧），run："
              f"{[ch[0]['run'] for ch in chains]}", flush=True)
    else:
        n_cells = a.cells
        ds = DG.GcRolloutDataset(a.data_root, a.split, a.text_table, vocab=vocab, n_cells=n_cells, verbose=True)
        clips = a.clip_ids.split(",") if a.clip_ids else pick_clips(ds.rows, a.n_clips)

        def _one(cid):
            s = ds.load(cid)
            r = row_of[cid]
            s.update(clips=[{"clip_id": cid, "frame0": 0, "look": r.get("look")}], subset=r["subset"], run=r["run"])
            return s
        items = [(cid, lambda cid=cid: _one(cid)) for cid in clips]
        print(f"[data] 单 clip {len(clips)} 条 × {n_cells} cell，run：{[row_of[c]['run'] for c in clips]}", flush=True)
    dec = Decoder(a.vae, dev, model_dir=a.model_dir, taehv_dir=a.taehv_dir, taehv_weights=a.taehv_weights)
    n_gen = n_cells - 1
    summary = []
    for _, load in items:
        s = load()
        cid = s["clip_id"]
        b = DG.rollout_collate([{k: s[k] for k in ("z0", "sem_ids", "context", "context_ids")}])
        timing: list = []
        lat = deploy_rollout(model, b["z0"].to(dev), b["context"].to(dev), b["context_ids"].to(dev),
                             b["sem_ids"].to(dev), n_gen, denoise_list=[float(x) for x in a.denoise_list.split(",")],
                             kv_window=a.kv_window, sink_size=a.sink_size, rope_cap=a.rope_cap, seed=a.seed,
                             timing=timing)
        assert torch.isfinite(lat.float()).all(), f"{cid}: rollout 出现 NaN/Inf"
        gen = dec(lat)
        gt, gt_valid = assemble_gt(dec, a.data_root, s["clips"], len(gen), dev)
        met = rollout_metrics(gen, gt, gt_valid)
        met["abs_drift"] = abs(met["drift"])
        met.update(clip_id=cid, subset=s["subset"], run=s["run"], clips=[c["clip_id"] for c in s["clips"]],
                   ckpt=a.ckpt, cells=n_cells, vae=a.vae, seed=a.seed,
                   geometry=dict(sink=a.sink_size, window=a.kv_window, cap=a.rope_cap, denoise_list=a.denoise_list),
                   dit_ms_per_cell=float(np.mean(timing[2:] or timing)), latent_abs_mean_cell=[
                       float(lat[0, :, k].float().abs().mean()) for k in range(lat.shape[2])])
        od = os.path.join(a.out, cid)
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "metrics.json"), "w") as fh:
            json.dump(met, fh, indent=1)
        if not a.no_video:
            import viz as VIZ
            sem_np = b["sem_ids"][0].numpy()
            VIZ.write_mp4(side_by_side(gen, gt, sem_np, s["prompts"], os.path.basename(os.path.dirname(os.path.dirname(a.ckpt)))
                                       + "/" + os.path.basename(a.ckpt), gt_valid), os.path.join(od, "rollout.mp4"))
            VIZ.write_video(os.path.join(od, "gen.mp4"), gen)
        print(f"[eval] {cid} ({s['run']}): {n_cells} cell / {len(gen)} 帧（GT {met.get('gt_frames', 0)} 帧）  "
              f"psnr {met['psnr']:.2f}  zncc {met['zncc']:.3f}  drift {met['drift']:+.2f}  "
              f"gt_luma_gap {met['gt_luma_gap']:+.2f}  DiT {met['dit_ms_per_cell']:.0f} ms/cell", flush=True)
        summary.append(met)
        del lat, gen, gt, s, b
    keys = ("psnr", "zncc", "drift", "abs_drift", "gt_luma_gap", "dit_ms_per_cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agg = {k: float(np.nanmean([m[k] for m in summary])) for k in keys}
    per_clip = [{k: m[k] for k in ("clip_id", "subset", "run", "gt_frames", "psnr", "zncc", "drift", "abs_drift",
                                    "gt_luma_gap")} for m in summary]
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"ckpt": a.ckpt, "res": DG.GEO.RES_TAG, "mode": "chain" if a.chain_cells else "clip", "cells": n_cells, "vae": a.vae,
                   "clips": [m["clip_id"] for m in summary], "mean": agg,
                   "per_subset": _group_means(per_clip, lambda r: r["subset"]),
                   "per_run": _group_means(per_clip, lambda r: r["run"]), "per_clip": per_clip}, fh, indent=1)
    print(json.dumps({"mean": agg}), flush=True)
    return agg


if __name__ == "__main__":
    main()
