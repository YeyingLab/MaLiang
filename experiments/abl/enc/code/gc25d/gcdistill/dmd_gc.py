#!/usr/bin/env python3
"""stage3：self-forcing DMD —— 用分布匹配把 stage2 的少步因果学生再抛光一遍（gc25d 版）。
（ARoPE distill/stage3_dmd/dmd_arope.py 的 fork，结构逐段对应；改动点见 distill_port.md §8 表。）

    # 正式训练（配方常量钉在 run_stage3.sh 里，argparse 默认值与它、config_stage3.json 逐项相同）
    STUDENT=<stage2 的 ema/ema_step5000.safetensors> SFT=<双向 SFT> bash code/gc25d/gcdistill/run_stage3.sh

    # 30 秒 dry run（tiny 四模型 + 合成数据；验 rollout、两个 loss、导出、续训）
    CUDA_VISIBLE_DEVICES="" $PY code/gc25d/gcdistill/dmd_gc.py --dry_run --output <abs tmp>

这一阶段是什么（同上游）
---------------------
学生真的自回归 rollout（self-forcing：滑窗 KV、逐 cell 提交、部署那张 4 步 renoise 表），再用 DMD 把 rollout
出来的分布推向双向教师的分布：

    student (因果, FSDP 可训)   ← 生成器，从 stage2 的 ema/ema_step{N} 起步（不是 dits/）
    critic  (双向, FSDP 可训)   ← fake-score，初值 = 双向 SFT
    teacher (双向, 冻结 bf16)   ← real-score，= 双向 SFT（outputs/gc25d_v10/step-N）
    ema     (因果, FSDP 冻结)   ← **部署权重**

DMD 梯度 = (pred_fake − pred_real) / normalizer，全在 x0 空间、全程 no_grad 算出来，再套 0.5·MSE 的 surrogate。
teacher 与 critic **只**在它们原生的双向整片模式下被驱动（没有 clean_x、没有 KV 缓存）。

相对 dmd_arope.py 的改动（distill_port §8）
-----------------------------------------
* **条件**：没有世界坐标 / known 掩码 / World-Canvas。每个 cell 的条件 = 它自己的 4 帧语义图（cell 0 = 帧 0）+ 它自己
  的动作 prompt，全部来自**真实**数据（GcRolloutDataset：cond + cond_ext 拼成 109 帧 / 28 个 cell）。上游的
  extend_offsets / extend_context("hold") 删掉 —— gc 的语义图与动作每个 cell 都在变，"保持"等于对着移动的语义图训
  站着不动（§6.6）。
* **KV 策略**：只有 recency（GcKVDriver）：sink 3 个最早 cell + 最近 17 个，rope_cap 20（gc25d_v8 部署几何）。
  流式条件 = `mask_in_k = cell_frames(sem_ext, k)`，offset_win = None（普通查表 RoPE）。
* **打分尾巴默认 frankenstein**：cat([真 cell 0, 最后 K 个生成 cell])（gc25d_v8 ckpt 的 tail_mode）。真实 cell 0
  是流形上的锚点，整窗的全局亮度偏移对教师可见（DMD-1；连续尾巴的锚点是生成的 cell，看不见这种偏移）。
  gc 没有世界坐标，缝合片在几何上没有 ARoPE 那个不连续问题。`--tail_mode contiguous` 留作对照。
* **模型 / 导出**：四个 GcDiT（词表按 ckpt 自动判断）+ gc_dit.save_gc（831 键），metadata 带 tail_mode / res。
  分辨率闸：student / teacher（及续训的 dits/critic/ema）ckpt 与数据根的 res 必须 = GC_RES（data_gc.check_res）。
* **数值与部署对齐**：学生输出转回 latent dtype（_as_latent）⇒ rollout 全 bf16、t = 1000/752/500/252（= live.py /
  SFT / stage1/2）；FSDP 的 cast_root_forward_inputs 关掉 ⇒ teacher 与 critic 实参相同（step 0 的 DMD 恰好 0）。
  streamed 的 record 趟里 grad 帧的退出前向也开 grad（随即 detach）⇒ 重放逐位等于 rollout（GPU 上 SDPA 分 grad /
  no_grad 两个 kernel）。opt_g.step 之前释放 FSDP 根单元的 unsharded 参数（_free_fsdp_root；否则 rollout 用上一步的
  根参数）。续训时两个 lr 调度快进到 resume_step（上游没快进）。
* 原封不动：grad_start_range / draw_grad_start、dmd_targets / critic_loss 的数学、sync-RNG 抽签、FSDP 闸、_clip、
  probe、S3-GUARD-3。_budget_gate 的静态显存按卡数缩放，活化 / KV 项按每 cell token 数缩放（GC_RES），
  dry run（tiny 模型）不过闸。

四个必须守住的坑（同上游，每一个都不会报错，只会静默训歪）
--------------------------------------------------------
1. **提交趟恒定 t=0**，作用在 detach 过的 x0 上，每个 frame_index 恰好一次（tests/test_stage3.py::test_1）。
2. **流式路径 first_frame_cond=False**（单 cell 前向里"第 0 帧"就是全部）。
3. **绝不对 rollout 做梯度检查点**（滑窗 commit 是破坏性的）。
4. **rollout 区的 autocast 必须 cache_enabled=False**（no_grad 前向缓存的 bf16 权重副本没有 grad_fn）。
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import socket
import sys
import time
import warnings

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
# 分配器要在 import torch 之前设：launch 时 export 的环境变量到不了 accelerate 拉起的子进程
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GC_ROOT not in sys.path:
    sys.path.insert(0, GC_ROOT)

from gcdistill import data_gc as DG  # noqa: E402
from gcdistill import gc_dit as GD  # noqa: E402
from gcdistill.gc_dit import cell_frames  # noqa: E402
import geometry as GEO  # noqa: E402
from distill import runlib  # noqa: E402
from distill.arope_dit import MODEL_DIR, unwrap  # noqa: E402
from distill.data import to_device, window_context  # noqa: E402
from distill.ema import ema_update  # noqa: E402
from distill.flow_pf import DMD_TIMESTEP_CLAMP, assert_ti2v, deploy_schedule, ti2v_noise  # noqa: E402
# 滑窗算术的启动自检（tiny CPU 模型，纯 KV 算术，与条件无关 —— ARoPE 的 sink 警告对 gc 不适用）
from distill.tests.preflight import preflight_sliding_kv  # noqa: E402
from text import NULL_PROMPT  # noqa: E402

CELLS = 21          # 一条 clip 的 latent cell 数（= teacher / critic 的训练片长）
STAGE = "stage3_dmd"
TAIL_MODES = ("frankenstein", "contiguous")


# ----------------------------------------------------------------------------
# 对齐闸：teacher / critic 每次前向之前跑一遍
# ----------------------------------------------------------------------------

def _assert_native_scoring(latent: torch.Tensor, ts_pf: torch.Tensor, cond: dict) -> None:
    """teacher/critic 只能在它们**原生**的双向整片模式下被驱动（gc：语义帧 uint8 [B,4f−3,H,W]）。"""
    b, c, f = latent.shape[:3]
    assert c == 48, f"[gate] 需要 channels-first [B,48,F,H,W]，得到 {tuple(latent.shape)}"
    ctx = cond["context"]
    assert ctx.dim() == 4 and tuple(ctx.shape[:2]) == (b, f), \
        f"[gate] 需要逐 cell 文本 [B,{f},L,4096]，得到 {tuple(ctx.shape)}"
    ids = cond["context_ids"]
    assert ids is not None and tuple(ids.shape) == (b, f), \
        f"[gate] context_ids 需要 [B,{f}]，得到 {None if ids is None else tuple(ids.shape)}"
    mi = cond["mask_input"]
    assert mi is not None and mi.dtype == torch.uint8 and tuple(mi.shape[:2]) == (b, DG.frames_of(f)), \
        f"[gate] mask_input 需要语义帧 uint8 [B,{DG.frames_of(f)},H,W]，得到 " \
        f"{None if mi is None else (mi.dtype, tuple(mi.shape))}"
    assert cond.get("offset_px") is None, "[gate] gc25d 没有世界坐标 RoPE"
    assert cond.get("first_frame_cond") is True, \
        "[gate] 整片打分路径必须 first_frame_cond=True（cell 0 是干净锚点）"
    assert cond.get("clean_x") is None and cond.get("kv_cache") is None, \
        "[gate] teacher/critic 绝不走 teacher forcing / KV 缓存 —— 那是学生的路径"
    assert ts_pf.shape == (b, f), f"[gate] ts_pf 需要 [B,{f}]，得到 {tuple(ts_pf.shape)}"


# ----------------------------------------------------------------------------
# rollout 的逐 cell 条件
# ----------------------------------------------------------------------------

class GcKVDriver:
    """一条 rollout 的 KV 缓存（recency：sink 个最早 cell + 最近 kv_window 个）—— rollout / 流式重放 / 梯度探针三处共用，
    保证它们看到的历史逐位相同。cell k 的条件 = 它自己的语义帧 cell_frames(sem_ext, k)（k=0 ⇒ [b,1,H,W]，否则 [b,4,H,W]）。"""

    def __init__(self, bare, sem_ext: torch.Tensor, *, kv_window: int, rope_cap: int, sink_size: int):
        self.bare, self.sem_ext = bare, sem_ext
        self.kv_window, self.rope_cap, self.sink_size = kv_window, rope_cap, sink_size
        self.cache = None

    def cond(self, k: int):
        """cell k 前向前：返回 (kv_cache, mask_in_k)。"""
        if self.cache is None:
            self.cache = self.bare.init_kv_cache(kv_window=self.kv_window, sink_size=self.sink_size,
                                                 rope_cap=self.rope_cap)
        return self.cache, cell_frames(self.sem_ext, k)

    def close(self) -> None:
        if self.cache is not None:
            self.bare.release_kv_cache(self.cache)
            self.cache = None


# ----------------------------------------------------------------------------
# self-forcing rollout
# ----------------------------------------------------------------------------

def _as_latent(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """学生输出转回 latent 的 dtype（gc 相对上游的改动）。accelerate.prepare 给模型 forward 套了 convert_outputs_to_fp32
    ⇒ 训练里 v 是 fp32（值是 bf16 的，转回去无损）。不转的话 den / 重新加噪的 noisy 变 fp32，而 arope_dit 的
    _per_token_timestep_pf 把 t 转成 **latent 的 dtype** ⇒ 学生在 750.0 / 500.6 / 251.6 上去噪；部署（live.py 不包装，
    v / den / noisy 全 bf16）、SFT、stage1/2 看到的都是 bf16 的 752 / 500 / 252。转回之后 x0 估计与重新加噪就是
    live.py::step_ids 的同一串 bf16 算术，打分尾巴也是 bf16 ⇒ teacher 与 critic 看到同一个（bf16）t。"""
    return v.to(x.dtype)


def self_forcing_rollout(student, initial_latent: torch.Tensor, ctx_ext: torch.Tensor,
                         ids_ext: torch.Tensor, sem_ext: torch.Tensor, num_gen_cells: int,
                         deploy: tuple, grad_horizon: int, *,
                         kv_window: int = 17, rope_cap: int = 20, sink_size: int = 3,
                         score_last_k: int | None = None, gen: torch.Generator | None = None,
                         record: dict | None = None, grad_start: int | None = None,
                         exit_step: int | None = None):
    """学生用自己提交的历史自回归生成 num_gen_cells 个 cell（self-forcing）。

    initial_latent [b,48,1,h,w]  真实首帧 latent（clip 的 cell 0）
    ctx_ext/ids_ext              [b,1+N,L,4096] / [b,1+N]，逐 cell 文本（cell k 用 [:,k]）
    sem_ext                      uint8 [b,4N+1,H,W] 真实语义帧（cell k 用 cell_frames(sem_ext, k)）
    deploy                       (idx, sigma, t_grid) —— flow_pf.deploy_schedule 的返回值
    grad_start / exit_step       调用方给的**同步过**的抽签结果（None ⇒ 本地抽，仅 smoke 用）
    record                       非 None ⇒ 不留任何图，只把 stream_grad_replay 要重放的东西记下来。grad 窗口里的退出前向
                                 照样在 grad 模式下跑、随即 detach：cuDNN SDPA 在 grad 模式下换 kernel（差 ~1 bf16 ulp），
                                 不这样做重放的 den 与这里的 den 不逐位相同 ⇒ surrogate 带一项伪梯度（step 0 也 ≠ 0）

    v 一律转回 latent 的 dtype 再算 x0 估计（_as_latent）。返回 (generated [b,48,1+N,h,w]（dtype = initial_latent），
    grad_mask [1+N] bool)。
    """
    b, c, _, lat_h, lat_w = initial_latent.shape
    dev = initial_latent.device
    total = 1 + num_gen_cells
    _, sig_dep, t_dep = deploy
    nsteps = len(t_dep)
    DG.assert_rollout_aligned_gc(sem_ext, ctx_ext, ids_ext, num_gen_cells)

    # init_kv_cache 在**裸模块**上调（FSDP 只包 forward，不包自定义方法）
    bare = unwrap(student)
    drv = GcKVDriver(bare, sem_ext, kv_window=kv_window, rope_cap=rope_cap, sink_size=sink_size)

    def _call(x, t, k, commit, mk, cache):
        # 必须走 forward(..., kv_cache=)：FSDP 的参数 all-gather 挂在包装模块的 forward 上。
        # 与 codeworld/live.py::LiveGame._call 同一个调用（逐 cell 文本 + 本 cell 语义帧，first_frame_cond=False）
        return student(x, t, ctx_ext[:, k:k + 1], context_ids=ids_ext[:, k:k + 1],
                       first_frame_cond=False, frame_index=k, kv_cache=cache, commit=commit, mask_in_k=mk)

    # ---- cell 0：真实干净首帧，t=0 提交进缓存 ----
    cache, mk0 = drv.cond(0)
    with torch.no_grad():
        _call(initial_latent, torch.zeros(b, device=dev), 0, True, mk0, cache)
    frame_list = [initial_latent[:, :, 0:1].detach()]

    # ---- grad 窗口：钉进 scored tail（否则 eval_idx ∩ grad_mask 为空 ⇒ loss 恒 0 且不报错）----
    if grad_start is None:
        grad_start = draw_grad_start(total, grad_horizon, score_last_k, gen=gen)
    grad_end = min(grad_start + grad_horizon, total)
    # 每条 rollout 抽**一个**退出档：去噪循环走到它就 break，梯度只穿过这一次退出前向
    if exit_step is None:
        exit_step = int(torch.randint(0, nsteps, (1,), generator=gen).item())
    exit_step = int(exit_step) % nsteps

    for k in range(1, total):
        cache, mk = drv.cond(k)
        noisy = torch.randn(b, c, 1, lat_h, lat_w, device=dev,
                            dtype=initial_latent.dtype, generator=gen)
        in_window = grad_start <= k < grad_end
        den = None
        for si in range(nsteps):
            is_exit = (si == exit_step)
            grad_fwd = in_window and is_exit                    # 这一次前向在 grad 模式下跑
            grad_on = grad_fwd and record is None               # …… 且把图留给 fused 的 backward
            if grad_fwd and record is not None:
                record["exit"][k] = (noisy.detach().clone(), si)
            ts_k = torch.full((b,), float(t_dep[si]), device=dev)
            with (torch.enable_grad() if grad_fwd else torch.no_grad()):
                v = _call(noisy, ts_k, k, False, mk, cache)
                if not grad_on:
                    v = v.detach()
                den = noisy - float(sig_dep[si]) * _as_latent(v, noisy)    # x0 估计
            if grad_fwd and record is not None:
                record.setdefault("den", {})[k] = den               # 重放时核对逐位相同（replay_dev）
            if is_exit:
                break
            ns = float(sig_dep[si + 1])                        # x0 → 下一档，重新加噪（detach）
            noisy = (1.0 - ns) * den.detach() + ns * torch.randn_like(den)
        frame_list.append(den)                                  # in_window 且 record is None 时带梯度
        # ---- 提交趟：**detach 过的 x0** + **t=0**，每个 frame_index 恰好一次（文件头坑 1）----
        with torch.no_grad():
            commit_lat = den.detach()
            if record is not None:
                record["commit"][k] = commit_lat
            _call(commit_lat, torch.zeros(b, device=dev), k, True, mk, cache)

    generated = torch.cat(frame_list, dim=2)                    # [b,48,1+N,h,w]
    assert generated.dtype == initial_latent.dtype, \
        f"rollout 的 latent dtype 漂了：{generated.dtype} != {initial_latent.dtype}（部署全程 bf16，t 跟着 latent dtype）"
    grad_mask = torch.tensor([grad_start <= k < grad_end for k in range(total)], device=dev)
    drv.close()   # cache 与 _ref 成环，不显式清会拖着十几 GB 不放
    if record is not None:
        record.update(total=total, grad_start=grad_start, grad_end=grad_end, exit_step=exit_step)
    return generated, grad_mask


def grad_start_range(total: int, grad_horizon: int, score_last_k: int | None) -> tuple[int, int]:
    """grad 窗口起点的合法闭区间 [lo, hi]（上游逐字）。

    lo = total − K：连续尾巴的锚点是 total−1−K（不打分），frankenstein 尾巴的第一个生成 cell 也是 total−K，
    两种尾巴的 grad 窗口完全相同。grad_horizon == K 时 lo == hi，窗口就正好是整条 scored tail。"""
    if score_last_k is None:
        return 1, max(1, total - grad_horizon)
    tail_start = max(0, total - 1 - score_last_k)
    lo = tail_start + 1
    return lo, max(lo, total - grad_horizon)


def draw_grad_start(total: int, grad_horizon: int, score_last_k: int | None,
                    gen: torch.Generator | None = None) -> int:
    """在 grad_start_range 里抽一个（smoke / 单测的本地兜底；训练主循环走同步抽签）。"""
    lo, hi = grad_start_range(total, grad_horizon, score_last_k)
    return int(torch.randint(lo, hi + 1, (1,), generator=gen).item())


def stream_grad_replay(student, initial_latent: torch.Tensor, ctx_ext: torch.Tensor,
                       ids_ext: torch.Tensor, sem_ext: torch.Tensor, record: dict,
                       deploy: tuple, frame_target, backward, *,
                       kv_window: int = 17, rope_cap: int = 20, sink_size: int = 3) -> float:
    """把带梯度的那几个 cell **一帧一帧**重做，每帧算完立刻 backward 再丢掉图（上游 streamed 路径，去掉 revisit）。

    精确（不是近似）：rollout 里每一次缓存提交都在 no_grad 下 ⇒ 各 grad 帧是独立的单帧图；surrogate 是
    reduction=mean 的 mse ⇒ 第 j 帧的份额正好是 mse_j / n_eval。重放按原顺序重新提交 record["commit"] 里**同一个**
    张量重建缓存，所以到第 k 帧时缓存状态与第一趟逐位相同；退出前向两趟都在 grad 模式下（同一个 SDPA kernel），
    所以重做的 den 与第一趟逐位相同 —— record["replay_dev"] = max|den_重放 − den_rollout|，应当恰好是 0。
    不能用 torch.utils.checkpoint（滑窗 KV 是破坏性的）。

    frame_target(k) -> 该 cell 的 surrogate 目标 [b,48,1,h,w]，None 表示这帧不打分。
    backward(loss)  -> 每个 grad 帧调一次（accel.backward）。返回累加的 loss 值（只为打日志）。
    """
    b = initial_latent.shape[0]
    dev = initial_latent.device
    _, sig_dep, t_dep = deploy
    bare = unwrap(student)
    drv = GcKVDriver(bare, sem_ext, kv_window=kv_window, rope_cap=rope_cap, sink_size=sink_size)

    def _call(x, t, k, commit, mk, cache):
        return student(x, t, ctx_ext[:, k:k + 1], context_ids=ids_ext[:, k:k + 1],
                       first_frame_cond=False, frame_index=k, kv_cache=cache, commit=commit, mask_in_k=mk)

    cache, mk0 = drv.cond(0)
    with torch.no_grad():
        _call(initial_latent, torch.zeros(b, device=dev), 0, True, mk0, cache)

    n_eval = max(1, int(record["n_eval"]))
    total_loss, dev_max, dev_cells = 0.0, 0.0, {}
    den_roll = record.get("den", {})
    for k in range(1, record["total"]):
        cache, mk = drv.cond(k)
        if k in record["exit"]:
            tgt = frame_target(k)
            if tgt is not None:
                noisy, si = record["exit"][k]
                ts_k = torch.full((b,), float(t_dep[si]), device=dev)
                with torch.enable_grad():
                    v = _call(noisy, ts_k, k, False, mk, cache)
                    den = noisy - float(sig_dep[si]) * _as_latent(v, noisy)
                    loss_k = 0.5 * F.mse_loss(den.float(), tgt.float()) / n_eval
                if k in den_roll:
                    d = (den.detach().float() - den_roll[k].float()).abs()
                    dk = float(d.max())
                    if dk > 0:
                        dev_cells[k] = (dk, float((d > 0).float().mean()))
                    dev_max = max(dev_max, dk)
                backward(loss_k)
                total_loss += float(loss_k.detach())
                del v, den, loss_k                              # 这一帧的图立刻还给分配器
        with torch.no_grad():                                   # 严格按第一趟的顺序重建缓存
            _call(record["commit"][k], torch.zeros(b, device=dev), k, True, mk, cache)
    drv.close()        # 每个训练步建一份，不显式清就是每步泄漏一份整条 rollout 的 K/V
    record["replay_dev"], record["replay_dev_cells"] = dev_max, dev_cells     # {cell: (max|Δ|, 不等元素占比)}
    return total_loss


# ----------------------------------------------------------------------------
# DMD 生成器 loss（分布匹配）—— 数学与上游逐字相同，只换了打分闸
# ----------------------------------------------------------------------------

def dmd_targets(teacher, critic, generated: torch.Tensor, cond_t: dict, grad_mask: torch.Tensor,
                *, real_guidance_scale: float = 1.0, uncond_context=None, uncond_ids=None,
                clamp=DMD_TIMESTEP_CLAMP, gen: torch.Generator | None = None):
    """dmd_generator_loss 里**全部 no_grad** 的那一半：给（detach 过的）片子打分、造 surrogate 目标。
    返回 (target, eval_idx, log)。"""
    b, _, f = generated.shape[:3]
    eval_idx = torch.nonzero(grad_mask, as_tuple=False).flatten()
    eval_idx = eval_idx[eval_idx >= 1]                          # cell 0 是干净锚点，永不打分
    assert eval_idx.numel() > 0, \
        "eval_idx 为空：grad 窗口落在 scored tail 之外了，loss 会恒等于 0 且不报错"
    with torch.no_grad():
        clip = generated.detach()
        noisy, ts_pf, sigma, _ = ti2v_noise(clip, clamp=clamp, gen=gen)
        assert_ti2v(noisy, clip, ts_pf)
        _assert_native_scoring(clip, ts_pf, cond_t)
        sig = sigma.view(b, 1, 1, 1, 1)
        # real-score。cfg==1.0 时 u + 1.0*(c−u) ≡ c，直接走 cond-only
        x0_real_c = noisy - sig * teacher(noisy, ts_pf, **cond_t)
        if abs(real_guidance_scale - 1.0) < 1e-6:
            pred_real = x0_real_c
        else:
            cond_u = dict(cond_t, context=uncond_context, context_ids=uncond_ids)
            x0_real_u = noisy - sig * teacher(noisy, ts_pf, **cond_u)
            pred_real = x0_real_u + real_guidance_scale * (x0_real_c - x0_real_u)
        pred_fake = noisy - sig * critic(noisy, ts_pf, **cond_t)          # fake-score
        raw_grad = pred_fake - pred_real                                   # x0 空间的 DMD 梯度
        p_real = clip - pred_real
        # normalizer 只在 **eval_idx** 上归约（不是硬写的 [:, :, 1:]）
        normalizer = p_real[:, :, eval_idx].abs().mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-2)
        grad = torch.nan_to_num(raw_grad / normalizer)
    target = (generated - grad).detach()
    log = {"dmd_gnorm": float(grad[:, :, eval_idx].norm()),
           "dmd_t": float(ts_pf[:, 1].mean()),
           "dmd_norm": float(normalizer.mean())}
    return target, eval_idx, log


def dmd_generator_loss(teacher, critic, generated: torch.Tensor, cond_t: dict,
                       grad_mask: torch.Tensor, **kw):
    """fused 路径：目标算完当场形成 surrogate。`d loss / d generated == grad`（只在 grad 窗口上）。"""
    target, eval_idx, log = dmd_targets(teacher, critic, generated, cond_t, grad_mask, **kw)
    g = generated[:, :, eval_idx].float()
    t = target[:, :, eval_idx].float()
    return 0.5 * F.mse_loss(g, t), log


def critic_loss(critic, generated: torch.Tensor, cond_t: dict, eval_idx: torch.Tensor, *,
                clamp=DMD_TIMESTEP_CLAMP, gen: torch.Generator | None = None,
                use_gradient_checkpointing: bool = True):
    """critic 在**学生自己生成的（detach 过的）**样本上做普通流匹配；打分下标与 DMD 同一套 eval_idx。"""
    clip = generated.detach()
    noisy, ts_pf, _, noise = ti2v_noise(clip, clamp=clamp, gen=gen)
    assert_ti2v(noisy, clip, ts_pf)
    _assert_native_scoring(clip, ts_pf, cond_t)
    v_fake = critic(noisy, ts_pf, use_gradient_checkpointing=use_gradient_checkpointing, **cond_t)
    target = noise - clip
    loss = F.mse_loss(v_fake[:, :, eval_idx].float(), target[:, :, eval_idx].float())
    return loss, {"critic": float(loss.detach())}


# ----------------------------------------------------------------------------
# scored tail
# ----------------------------------------------------------------------------

def score_tail(generated: torch.Tensor, grad_mask: torch.Tensor, sem_ext: torch.Tensor,
               ctx_ext: torch.Tensor, ids_ext: torch.Tensor, N: int, Keff: int | None, *,
               tail_mode: str = "frankenstein"):
    """只给 teacher/critic 看 1+Keff 个 cell 的片子（整条 rollout 照样端到端跑完，驱逐真的被练到了）。

    frankenstein（默认，gc25d_v8）：cat([真 cell 0, cells a..N])，a = N+1−Keff。语义帧 = cat(帧 0, 帧 4a−3 … 4N)，
        文本 = cat(cell 0, cells a..N)。片子的 cell 0 是真实 z0 —— 流形上的锚点。
    contiguous：generated[:, :, a:]，a = N−Keff；锚点是**生成**的 cell a（用它的最后一帧 4a 当首帧语义，window_frames）。
    返回 (tail_sl, cond_t, grad_mask_t, tail_to_frame)。
    """
    if tail_mode not in TAIL_MODES:
        raise ValueError(f"tail_mode 只能是 {TAIL_MODES}，得到 {tail_mode!r}")
    total = 1 + N
    a, tail_to_frame = DG.tail_slices(total, Keff, tail_mode == "frankenstein")
    if a == 0:
        tail_sl, grad_mask_t = generated, grad_mask
        sem_t = sem_ext[:, :DG.frames_of(total)]
        ctx_t, ids_t = window_context(ctx_ext, ids_ext, 0, N)
    elif tail_mode == "frankenstein":
        tail_sl = DG.frankenstein_cells(generated, a, dim=2)
        sem_t = DG.frankenstein_frames(sem_ext, a, total)
        ctx_t, ids_t = DG.frankenstein_cells(ctx_ext, a), DG.frankenstein_cells(ids_ext, a)
        grad_mask_t = torch.cat([grad_mask[:1], grad_mask[a:]])
    else:
        tail_sl = generated[:, :, a:]
        sem_t = GD.window_frames(sem_ext, a, total - a)
        ctx_t, ids_t = window_context(ctx_ext, ids_ext, a, N)
        grad_mask_t = grad_mask[a:]
    assert not bool(grad_mask_t[0]), "tail 的 cell 0 是干净锚点，不能落在梯度窗口里"
    cond_t = dict(context=ctx_t, context_ids=ids_t, mask_input=sem_t, first_frame_cond=True)
    assert tail_sl.shape[2] == len(tail_to_frame) == ctx_t.shape[1] == ids_t.shape[1], \
        "scored tail 的 latent / 文本 cell 数对不齐"
    assert sem_t.shape[1] == DG.frames_of(tail_sl.shape[2]), "scored tail 的语义帧数与 cell 数对不齐"
    return tail_sl, cond_t, grad_mask_t, tail_to_frame


# ----------------------------------------------------------------------------
# 一次性梯度可达性自检
# ----------------------------------------------------------------------------

def probe_student_grad(student, initial_latent: torch.Tensor, ctx_ext: torch.Tensor,
                       ids_ext: torch.Tensor, sem_ext: torch.Tensor, deploy: tuple,
                       backward, clip_fn, *, kv_window: int, rope_cap: int, sink_size: int) -> float:
    """走一遍**真正的** rollout 前向（滑窗 KV、单 cell），对一个人为目标 backward，确认梯度真的落到学生参数上。
    （critic 与 teacher 同一份双向 SFT 初值 ⇒ step 0 的 DMD 梯度**必然恰好是 0**，拿第 1 步的 gnorm_g 判断不了接线。）"""
    bare = unwrap(student)
    b = initial_latent.shape[0]
    dev = initial_latent.device
    _, sig_dep, t_dep = deploy
    drv = GcKVDriver(bare, sem_ext, kv_window=kv_window, rope_cap=rope_cap, sink_size=sink_size)
    cache, mk0 = drv.cond(0)
    with torch.no_grad():
        student(initial_latent, torch.zeros(b, device=dev), ctx_ext[:, 0:1],
                context_ids=ids_ext[:, 0:1], first_frame_cond=False, frame_index=0,
                kv_cache=cache, commit=True, mask_in_k=mk0)
    cache, mk = drv.cond(1)
    noisy = torch.randn_like(initial_latent)
    with torch.enable_grad():
        v = student(noisy, torch.full((b,), float(t_dep[0]), device=dev), ctx_ext[:, 1:2],
                    context_ids=ids_ext[:, 1:2], first_frame_cond=False, frame_index=1,
                    kv_cache=cache, commit=False, mask_in_k=mk)
        den = noisy - float(sig_dep[0]) * _as_latent(v, noisy)
        probe = den.float().square().mean()      # 人为目标：只要不是常数就行
    backward(probe)
    gn = float(clip_fn(student, float("inf")))   # max_norm=inf ⇒ 只取范数，不缩放
    del v, den, probe
    drv.close()
    return gn


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="stage3：self-forcing DMD（gc25d）")
    # 默认值 = run_stage3.sh 的实参 = config_stage3.json（只允许存在一份配方）
    ap.add_argument("--student_ckpt", default=None,
                    help="生成器初始化 = stage2 的 **ema/ema_step{N}.safetensors**（不是 dits/）")
    ap.add_argument("--teacher_ckpt", default=None,
                    help="real-score 教师 + critic 初始化 = **双向** SFT ckpt（不是 stage1 的因果 ckpt）")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--base_init", choices=("skeleton", "wan"), default="skeleton",
                    help="skeleton：不读 Wan 基础权重、从 831 键 ckpt 严格装满（四个模型各省一次 19 GB 读盘）")
    ap.add_argument("--output", required=True, help="run 目录（写绝对路径）")

    ap.add_argument("--data_root", default=DG.DATA_ROOT)
    ap.add_argument("--text_table", default=None, help="None ⇒ <data_root>/text_table_gc25d.pt")
    ap.add_argument("--vocab", default="auto", choices=("auto", "parts", "parts_fx", "ids"))
    ap.add_argument("--limit", type=int, default=None, help="smoke 用：只取这么多训练 clip")
    ap.add_argument("--synthetic", type=int, default=0, help=">0 ⇒ 全尺寸合成样本（prep 写完之前的真权重冒烟）")

    # ---- 部署采样表 ----
    ap.add_argument("--denoise_list", default="1000,750,500,250",
                    help="少步采样表（原始 t 值，从噪到净严格降序）。stage2 的 CD 轴、stage3、部署推理是同一张表")
    ap.add_argument("--real_guidance_scale", type=float, default=1.0,
                    help="教师 real-score 的 CFG。1.0 ⇒ 完全跳过 uncond 前向（CF++ 铁律）")

    # ---- 滑窗 KV（部署几何，gc25d_v8：sink 3 / window 17 / cap 20）----
    ap.add_argument("--kv_window", type=int, default=17, help="滑动窗口里保留的最近 cell 数")
    ap.add_argument("--sink_size", type=int, default=3,
                    help="恒可见的开头 cell 数（外观锚点）。gc 没有世界坐标 RoPE，ARoPE 的 sink 漂移问题不存在")
    ap.add_argument("--rope_cap", type=int, default=20, help="时间 RoPE 位上限 = F−1。硬不变量 rope_cap == sink + window")
    ap.add_argument("--rollout_min", type=int, default=21,
                    help="每步 rollout 生成的 cell 数下界（rank 同步抽）。必须 ≥ sink+window+1 = 21 才练得到驱逐后的读取")
    ap.add_argument("--rollout_max", type=int, default=27, help="≤ 27：cond + cond_ext 一共 28 个 cell 的真实条件")
    ap.add_argument("--num_gen_cells", type=int, default=None, help="给了就固定 N（调试用），不再抽")

    # ---- DMD ----
    ap.add_argument("--dmd_score_frames", type=int, default=20,
                    help="K：teacher/critic 看 1+K 个 cell 的片子（20 ⇒ 21 = 它们的训练片长）")
    ap.add_argument("--grad_horizon", type=int, default=20, help="rollout 里保留 autograd 的 cell 数，必须 ≤ K")
    ap.add_argument("--grad_pass", choices=["streamed", "fused"], default="streamed",
                    help="streamed：rollout 不建图，之后每个 grad 帧单独重做 + backward（同一个梯度，峰值不随 horizon 涨）")
    ap.add_argument("--exit_step", type=int, default=None, help="固定退出档（只给排查工具用）；训练留 None")
    ap.add_argument("--tail_mode", choices=TAIL_MODES, default="frankenstein",
                    help="frankenstein（gc25d_v8）= cat([真 cell 0, 最后 K 个生成 cell])；contiguous = 上游 ARoPE 的连续尾巴")
    ap.add_argument("--dmd_timestep_clamp", default="20,980", help="DMD 打分抽的**下标**区间 [lo, hi)")
    ap.add_argument("--critic_steps", type=int, default=5, help="每个生成器步跑几次 critic 更新（CF++ 的 5）")

    # ---- 优化器 ----
    ap.add_argument("--lr", type=float, default=2e-6, help="生成器 lr（5e-6 是已知的高 lr min-max 崩溃）")
    ap.add_argument("--lr_critic", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--adam_betas", default="0.0,0.999", help="beta1=0 是 CF 对非平稳 fake-score 目标的约定")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=10.0, help="生成器与 critic 都是 10.0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="只作用在 critic 的整片前向上（必须开）；**绝不会**用到 rollout 上（arope_dit 硬断言）")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # ---- EMA / 存档 ----
    ap.add_argument("--use_ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.99)
    ap.add_argument("--ema_start", type=int, default=100, help="这之前每步硬拷贝 ema ← student")
    ap.add_argument("--save_every", type=int, default=50, help="部署权重要逐个里程碑挑。0 ⇒ 只在结束时导出")
    ap.add_argument("--keep_last_dits", type=int, default=3)
    ap.add_argument("--keep_last_emas", type=int, default=25, help="全部里程碑都要留着挑")
    ap.add_argument("--keep_last_critics", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--profile_mem", type=int, default=0, help="1 ⇒ 前几步逐阶段打峰值显存")
    ap.add_argument("--resume", default=None,
                    help="auto ⇒ 有 <run>/progress.json 就从 dits/ + critic/ + ema/ 接着跑（只恢复权重）")
    ap.add_argument("--force_run_dir", action="store_true",
                    help="--output 目录里 train.log 刚被写过时仍然继续（默认拒）")
    ap.add_argument("--dry_run", action="store_true", help="tiny 四模型 + 合成数据，不碰真权重 / 真数据")
    ap.add_argument("--dry_run_full_hw", action="store_true",
                    help="[dry run] 合成数据用当前 GC_RES 的全尺寸 latent / 语义图网格（tiny 模型照旧），验分辨率链路")
    args = ap.parse_args(argv)

    if args.rope_cap != args.sink_size + args.kv_window:
        raise SystemExit(f"--rope_cap({args.rope_cap}) 必须 == --sink_size({args.sink_size}) + "
                         f"--kv_window({args.kv_window})")
    if args.grad_horizon > args.dmd_score_frames > 0:
        raise SystemExit(f"--grad_horizon({args.grad_horizon}) 必须 ≤ --dmd_score_frames"
                         f"({args.dmd_score_frames})：否则梯度窗口跑出打分区，loss 恒 0 且不报错")
    if 0 < args.grad_horizon < args.dmd_score_frames:
        print(f"[warn] grad_horizon({args.grad_horizon}) < dmd_score_frames({args.dmd_score_frames})："
              f"critic 与 normalizer 只覆盖梯度窗口的 {args.grad_horizon} 帧。默认 20/20 无此问题。", flush=True)
    if args.critic_steps < 1:
        raise SystemExit("--critic_steps 必须 ≥ 1")
    if args.rollout_min > args.rollout_max:
        raise SystemExit(f"--rollout_min({args.rollout_min}) > --rollout_max({args.rollout_max})")
    if args.rollout_min < args.sink_size + args.kv_window + 1 and not args.dry_run:
        raise SystemExit(f"--rollout_min({args.rollout_min}) 必须 ≥ sink+window+1 = "
                         f"{args.sink_size + args.kv_window + 1}，否则驱逐要么不发生、要么发生了也没被读过")
    if args.rollout_max + 1 > DG.ROLLOUT_CELLS and not args.dry_run:
        raise SystemExit(f"--rollout_max({args.rollout_max}) 要 {args.rollout_max + 1} 个 cell 的真实条件，"
                         f"数据只有 {DG.ROLLOUT_CELLS} 个（cond 21 + cond_ext 7）")
    try:
        b1, b2 = (float(x) for x in args.adam_betas.split(","))
    except ValueError:
        raise SystemExit(f"--adam_betas 应形如 '0.0,0.999'，得到 {args.adam_betas!r}") from None
    args.betas = (b1, b2)
    try:
        lo, hi = (int(x) for x in args.dmd_timestep_clamp.split(","))
    except ValueError:
        raise SystemExit(f"--dmd_timestep_clamp 应形如 '20,980'，得到 {args.dmd_timestep_clamp!r}") from None
    args.clamp = (lo, hi)
    args.denoise = [float(t) for t in args.denoise_list.split(",") if t.strip()]
    if args.text_table is None:
        args.text_table = os.path.join(args.data_root, "text_table_gc25d.pt")
        if (args.synthetic or args.dry_run) and not os.path.isfile(args.text_table):
            args.text_table = DG.REF_TEXT_TABLE
    if args.dry_run:
        args.student_ckpt = args.teacher_ckpt = None
        args.vocab = "parts" if args.vocab == "auto" else args.vocab
    else:
        if not args.student_ckpt:
            raise SystemExit("--student_ckpt 必填：stage2 的 ema/ema_step{N}.safetensors")
        # 先查这一条再查存在性：拿错文件的人给的路径通常是存在的
        if "/dits/" in args.student_ckpt.replace(os.sep, "/"):
            raise SystemExit(f"--student_ckpt 指向 dits/：{args.student_ckpt}\n"
                             "  stage2 的部署权重是 ema/ema_step{N}.safetensors，dits/ 只为续训存在。")
        if not args.teacher_ckpt:
            raise SystemExit("--teacher_ckpt 必填：双向 SFT（outputs/gc25d_v10/step-N.safetensors）")
        for p, name in ((args.student_ckpt, "--student_ckpt"), (args.teacher_ckpt, "--teacher_ckpt")):
            if not os.path.isfile(p):
                raise SystemExit(f"{name} 不存在：{p}")
        vs, vt = GD.ckpt_vocab(args.student_ckpt), GD.ckpt_vocab(args.teacher_ckpt)
        if vs is None or vs != vt:
            raise SystemExit(f"student / teacher 的 gc_sem_cond 词表不一致或缺失：{vs} / {vt}")
        if args.vocab == "auto":
            args.vocab = vs
        elif args.vocab != vs:
            raise SystemExit(f"--vocab {args.vocab} 与 ckpt 的词表 {vs} 不符")
        # 分辨率闸（加载 4×5B 之前）：两份 ckpt 的 metadata res、数据根的 res 都必须 = GC_RES
        DG.check_res(ckpts=[("--student_ckpt", args.student_ckpt), ("--teacher_ckpt", args.teacher_ckpt)],
                     root=None if args.synthetic else args.data_root)
    return args


# ----------------------------------------------------------------------------
# dry run：tiny 四模型 + 合成数据
# ----------------------------------------------------------------------------

def _tiny_model(causal: bool, seed: int = 0, vocab: str = "parts") -> GD.GcDiT:
    return DG.tiny_gcdit(causal=causal, seed=seed, vocab=vocab)


def _shrink_for_dry_run(args) -> None:
    """把配方缩到 tiny 几何上还成立的规模：sink 1 / 窗口 3 / cap 4、rollout 6–8、K=H=4。"""
    args.kv_window, args.sink_size, args.rope_cap = 3, 1, 4
    args.rollout_min, args.rollout_max = 6, 8
    args.dmd_score_frames, args.grad_horizon = 4, 4
    if args.max_steps > 30:
        args.max_steps, args.save_every, args.warmup, args.log_every = 6, 3, 2, 1
        args.critic_steps, args.ema_start = 2, 2


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

# 上游实测（GPU 0、真实 5B、bf16、latent 30×52 ⇒ 390 token/cell、30 层、窗口 20 个 cell —— sink3/w17 同样是 20 个）。
# 活化 / KV 三项与每 cell 的 token 数成正比（SDPA 的显存对序列长度线性）⇒ 乘 TOK_SCALE = GEO.TOK_PER_CELL / 390
# （832×480 ⇒ 1.0 原值；1280×704 ⇒ 880/390 = 2.26）；参数 / 优化器状态与分辨率无关。
TOK_SCALE = GEO.TOK_PER_CELL / 390
GIB_PER_GRAD_FRAME = 6.46 * TOK_SCALE    # fused 每多保留一帧的图
GIB_KV_CACHE = 2.72 * TOK_SCALE          # n_cached=20 的 K/V 缓存（30 层）
GIB_SHARDED_TOTAL = 167.7    # student 74.5 + critic 74.5（fp32 参数 + 梯度 + AdamW m,v）+ ema 18.6，FULL_SHARD 按卡数均分
GIB_TEACHER = 9.31           # teacher bf16 不分片、整份复制
GIB_SCORING_MAX = 5.4 * TOK_SCALE        # teacher no_grad 1.69 + critic 带梯度检查点 3.69（21 cell 上界）


def _budget_gate(args, accel, device, log) -> None:
    """显存预算闸：在加载 4×5B 之前拒绝跑不下的配置（fused 按 horizon 线性吃活化值；streamed 恒定）。
    静态项按 FULL_SHARD 的卡数缩放，活化项按 TOK_SCALE。
    dry run（tiny 模型，常数是 5B 的）不过闸。"""
    if device.type != "cuda" or args.dry_run:
        return
    tot = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    static = GIB_SHARDED_TOTAL / max(1, accel.num_processes) + GIB_TEACHER
    frames = args.grad_horizon if args.grad_pass == "fused" else 1
    need = static + GIB_KV_CACHE + GIB_SCORING_MAX + frames * GIB_PER_GRAD_FRAME
    budget = 0.90 * tot
    log(f"[budget] {GEO.RES_TAG}（{GEO.TOK_PER_CELL} token/cell，×{TOK_SCALE:.2f}）每卡预估峰值 {need:.1f} GiB / {tot:.0f} GiB"
        f"（静态 {static:.1f} @ {accel.num_processes} 卡 + KV "
        f"{GIB_KV_CACHE:.1f} + 打分 {GIB_SCORING_MAX:.1f} + {frames} 帧图 × {GIB_PER_GRAD_FRAME:.2f}）"
        f"[grad_pass={args.grad_pass} horizon={args.grad_horizon}]")
    if need > budget:
        raise SystemExit(f"ABORT: --grad_pass {args.grad_pass} --grad_horizon {args.grad_horizon} 每卡要 "
                         f"{need:.0f} GiB，超过 {budget:.0f} GiB 的预算（卡 {tot:.0f} GiB）。"
                         f"用 --grad_pass streamed（与 fused 逐位等价，峰值不随 horizon 涨）或更多卡。")


def _assert_run_dir_safe(run_dir: str, stage: str, force: bool, log=print, dry_run: bool = False) -> None:
    """开跑之前确认 `--output` 指的不是**别人的** run 目录（同 cd_gc._assert_run_dir_safe，上游也是两份逐字副本）。"""
    prog = runlib.read_progress(run_dir)
    if prog and prog.get("stage") and str(prog["stage"]) != stage:
        raise SystemExit(
            f"ABORT: {run_dir} 是 stage={prog['stage']!r} 的 run 目录（progress.json 记在 step "
            f"{prog.get('step')}），本次是 {stage}。保存时会 rotate 掉那边的 dits/ema —— 换个 --output。")
    log_path = os.path.join(run_dir, "train.log")
    if os.path.isfile(log_path):
        age = time.time() - os.path.getmtime(log_path)
        if age < 600 and not (force or dry_run):
            raise SystemExit(
                f"ABORT: {log_path} 在 {age / 60:.1f} 分钟前还被写过 —— 多半有另一个 run 正活在这个目录里。"
                "换个 --output；如果确认上一次已经死了，加 --force_run_dir（launcher 里是 FORCE_RUN_DIR=1）。")
        if age < 600:
            log(f"[warn] {log_path} {age / 60:.1f} 分钟前刚被写过，--force_run_dir 强行继续。")


def _fsdp_gate(accel, log) -> None:
    """启动闸 2：FSDP 的致命开关（main 与 tests/fsdp_t_parity_worker.py 共用同一份）。"""
    if accel.distributed_type == DistributedType.FSDP:
        plugin = accel.state.fsdp_plugin
        if not getattr(plugin, "use_orig_params", False):
            raise SystemExit("fsdp_use_orig_params 必须为 true：false 时优化器 step 的是过期张量、永远收不到梯度，"
                             "zip-EMA 也要求 Parameter 的身份/顺序/分片完全一致。")
        # ★ gc 相对上游的改动：关掉 FSDP1 MixedPrecision 默认的 cast_root_forward_inputs，让包装过的 critic 与不包装的
        #   teacher 拿到**同样的**实参（模型内部的 t 跟着 latent 的 dtype 走，见 _as_latent）。真正决定 t 的是 latent
        #   dtype：rollout 把 v 转回 bf16 ⇒ 学生 rollout / 重放 / 探针的 t = 1000 / 752 / 500 / 252（= 部署 live.py、
        #   SFT、stage1/2），打分尾巴是 bf16 ⇒ teacher 与 critic 的 DMD t 一样按 bf16 取整（如 974.9 → 976）。
        #   曾经的 bug：tail 是 fp32 时 teacher 看 fp32 t、critic 被 FSDP 转成 bf16 t ⇒ step 0 的 DMD = +0.009 ≠ 0。
        mpp = getattr(plugin, "mixed_precision_policy", None)
        if mpp is not None and getattr(mpp, "cast_root_forward_inputs", False):
            mpp.cast_root_forward_inputs = False
        log(f"[fsdp] use_orig_params={plugin.use_orig_params} "
            f"reshard={getattr(plugin, 'reshard_after_forward', None)} "
            f"version={getattr(plugin, 'fsdp_version', 1)} "
            f"cast_root_forward_inputs={getattr(mpp, 'cast_root_forward_inputs', None)}"
            f"（t 跟 latent dtype：rollout 与打分全 bf16）")
    elif accel.num_processes > 1:
        log(f"[warn] 期望 FSDP（两个可训模型），实际是 {accel.distributed_type}")


def _free_fsdp_root(model) -> bool:
    """FSDP1 的根单元在 forward 之后**不**释放 unsharded 参数（_runtime_utils._post_forward_reshard：留给"紧接着的"
    backward）。学生最后一次前向之后若没有 backward（重放 / rollout 末尾的 no_grad 提交趟），这份 bf16 副本就一直留着；
    opt_g.step 之后下一次前向 needs_unshard() 为假 ⇒ **不重新 all-gather**，根单元的参数（patch / text / time embedding、
    head、gc_sem_cond；DiTBlock 各自是子单元，不受影响）用的是**上一步**的值：rollout 整条、重放的第一个 grad 帧都落后一步
    （上游 dmd_arope 同样如此）。每次 opt_g.step 之前显式释放一次（同 FSDP 自己 post-backward 的 _reshard(free=True)）。
    返回是否真的释放了（非 FSDP / 已释放 ⇒ False）。"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if not isinstance(model, FSDP):
        return False
    h = getattr(model, "_handle", None)
    if h is None or not h.uses_sharded_strategy or h.needs_unshard():
        return False
    from torch.distributed.fsdp._runtime_utils import _reshard
    _reshard(model, h, True)
    return True


def _gc_model(ckpt: str, causal: bool, args) -> GD.GcDiT:
    DG.check_res(ckpts=[("ckpt", ckpt)])              # 续训的 dits/critic/ema 也过一遍（metadata res = GC_RES）
    dit = GD.empty_wan_dit() if args.base_init == "skeleton" else None
    m = GD.GcDiT(dit=dit, causal=causal, model_dir=args.model_dir, vocab=args.vocab)
    GD.load_gc_ckpt(m, ckpt)
    return m


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        _shrink_for_dry_run(args)
    accel = Accelerator(step_scheduler_with_optimizer=False)
    is_main = accel.is_main_process
    device = accel.device
    set_seed(args.seed, device_specific=True)   # 每个 rank 的噪声 / 抽签不同；同步项另有 sync_rng

    run_dir = runlib.resolve_run_dir(args.output)
    _assert_run_dir_safe(run_dir, STAGE, bool(args.force_run_dir), dry_run=bool(args.dry_run))
    DG.check_res(run_dir=run_dir)
    accel.wait_for_everyone()
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    log = runlib.TextTee(os.path.join(run_dir, "train.log"), enabled=is_main)
    jsonl = runlib.JsonlLogger(os.path.join(run_dir, "train_log.jsonl"), enabled=is_main)
    log(f"=== {STAGE} (gc25d) | run={run_dir} world={accel.num_processes} "
        f"dist={accel.distributed_type} mp={accel.mixed_precision} ===")

    # ---- 启动闸 1：滑窗 KV 的算术（tiny CPU 模型；在烧掉 4×5B 之前跑）----
    pf = preflight_sliding_kv(args.sink_size, args.kv_window, args.rope_cap,
                              log=(log if is_main else (lambda *a: None)))
    if args.sink_size > 0:
        log("[preflight] 上面那条 sink 警告说的是 ARoPE 的世界坐标 RoPE；gc25d 没有世界坐标（offset_win=None），"
            "sink 3 只是外观锚点，不适用。")
    # ---- 启动闸 2：FSDP 的致命开关 ----
    _fsdp_gate(accel, log)

    idx_dep, sig_dep, t_dep = deploy_schedule(args.denoise)
    deploy = (idx_dep, sig_dep, t_dep)
    log(f"[recipe] denoise={args.denoise} ⇒ 网格 t={[round(float(x), 1) for x in t_dep]} "
        f"sigma={[round(float(x), 4) for x in sig_dep]}")
    log(f"[recipe] res={GEO.RES_TAG}（{GEO.RES_ENV}）kv_window={args.kv_window} sink={args.sink_size} cap={args.rope_cap} "
        f"rollout={args.rollout_min}..{args.rollout_max} K={args.dmd_score_frames} tail={args.tail_mode} "
        f"horizon={args.grad_horizon} grad_pass={args.grad_pass} critic_steps={args.critic_steps} "
        f"lr_g={args.lr} lr_c={args.lr_critic} betas={args.betas} clip={args.grad_clip} "
        f"cfg={args.real_guidance_scale} ema={args.use_ema}/{args.ema_decay}@{args.ema_start} "
        f"vocab={args.vocab} max_steps={args.max_steps}")
    _budget_gate(args, accel, device, log)
    low = runlib.warn_if_low_disk(run_dir)
    if low:
        log(f"[warn] {low}")

    # ---- 数据：真实 z0 + 1+rollout_max 个 cell 的真实条件 ----
    n_cells = args.rollout_max + 1
    if args.dry_run:
        c, _, h, w = DG.TINY_SHAPE
        h, w = DG.LAT_HW if args.dry_run_full_hw else (h, w)
        train_ds = DG.SynGcRolloutDataset(64, n_cells, (h, w), vocab=args.vocab)
        log(f"[dry_run] tiny 四模型 + 合成数据（{n_cells} cell）")
    elif args.synthetic:
        train_ds = DG.SynGcRolloutDataset(args.synthetic, n_cells, DG.LAT_HW, vocab=args.vocab,
                                          text_table=args.text_table, seed=args.seed)
        log(f"[data] 合成 ×{args.synthetic}（{n_cells} cell，文本取自 {args.text_table}）")
    else:
        train_ds = DG.GcRolloutDataset(args.data_root, "train", args.text_table, vocab=args.vocab,
                                       n_cells=n_cells, limit=args.limit, seed=args.seed, verbose=is_main)
        log(f"[data] {args.data_root} train {len(train_ds)}/{train_ds.n_total} clip（has_ext 且未 drop），"
            f"{n_cells} cell / {DG.frames_of(n_cells)} 帧")
    dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=DG.rollout_collate,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    # ---- 续训（只恢复权重，必须在 prepare 之前装进裸模块）----
    resume_step, resume_paths = 0, {}
    if args.resume:
        if args.resume == "auto":
            src = run_dir
        elif os.path.isdir(args.resume):
            src = args.resume
        else:
            raise SystemExit(f"--resume 只接受 'auto' 或一个 run 目录，得到 {args.resume!r}。")
        rp = runlib.read_progress(src)
        if rp and int(rp.get("step", 0)) > 0:
            resume_step = int(rp["step"])
            resume_paths = {"student": runlib.dit_path(src, resume_step),
                            "critic": runlib.critic_path(src, resume_step),
                            "ema": runlib.ema_path(src, resume_step)}
            miss = [k for k, p in resume_paths.items() if not os.path.isfile(p)]
            if miss:
                raise SystemExit(f"--resume 想从 step {resume_step} 接着跑，但缺 {miss}")
            log(f"[resume] 从 {src} 的 step {resume_step} 续：student←dits/ critic←critic/ ema←ema/"
                f"（优化器状态不存，续训 ≠ 不间断的一次训练）")
        else:
            log("[resume] 没有 progress.json，从头开始")

    # ---- 四个模型 ----
    t0 = time.time()
    if args.dry_run:
        student = _tiny_model(causal=True, seed=0, vocab=args.vocab)
        # ★ dry run 里 critic 故意与 teacher **不同**种子（真实配方两者同一份 ⇒ step 0 的 DMD 恰好是 0）
        critic = _tiny_model(causal=False, seed=2, vocab=args.vocab)
        teacher = _tiny_model(causal=False, seed=1, vocab=args.vocab)
        ema = _tiny_model(causal=True, seed=0, vocab=args.vocab) if args.use_ema else None
        if resume_step:
            runlib.load_dit_ckpt(student, resume_paths["student"], allow_missing_mask=False)
            runlib.load_dit_ckpt(critic, resume_paths["critic"], allow_missing_mask=False)
            if ema is not None:
                runlib.load_dit_ckpt(ema, resume_paths["ema"], allow_missing_mask=False)
    else:
        s_src = resume_paths.get("student", args.student_ckpt)
        c_src = resume_paths.get("critic", args.teacher_ckpt)
        e_src = resume_paths.get("ema", args.student_ckpt)
        student = _gc_model(s_src, True, args)            # 生成器 ← stage2 的 EMA
        critic = _gc_model(c_src, False, args)            # critic ← 双向 SFT
        ema = _gc_model(e_src, True, args) if args.use_ema else None     # EMA 从学生初值起步
        teacher = _gc_model(args.teacher_ckpt, False, args)
        log(f"[model] student←{os.path.basename(s_src)} critic←{os.path.basename(c_src)} "
            f"teacher←{os.path.basename(args.teacher_ckpt)}（vocab={args.vocab}，base_init={args.base_init}）")

    # ★ prepare **之前** 全部 .float()：FSDP MixedPrecision(bf16) 下 bf16 参数没有 fp32 主权重
    trainable = [("student", student), ("critic", critic)]
    for name, m in trainable + ([("ema", ema)] if ema is not None else []):
        m.float()
        bad = next((n for n, p in m.named_parameters() if p.dtype is not torch.float32), None)
        if bad is not None:
            raise SystemExit(f"{name} 的 {bad} 不是 fp32 —— .float() 没生效，FSDP 下会没有 fp32 主权重")
    student.train().requires_grad_(True)
    critic.train().requires_grad_(True)
    if ema is not None:
        ema.requires_grad_(False).eval()
    # CPU dry run 没有 autocast，fp32 输入撞 bf16 权重会直接报错
    teacher.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).requires_grad_(False).eval()
    n_params = sum(p.numel() for p in student.parameters())
    log(f"[model] 4 × {n_params:,} 参数（student fp32 可训 / critic fp32 可训 / "
        f"teacher bf16 冻结不分片 / ema fp32 冻结），加载 {time.time() - t0:.0f}s")

    def _adamw(params, lr):
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay, betas=args.betas)

    opt_g = _adamw(student.parameters(), args.lr)
    opt_c = _adamw(critic.parameters(), args.lr_critic)
    warm = max(int(args.warmup), 0)

    def lr_lambda(s):
        return min(1.0, (s + 1) / warm) if warm > 0 else 1.0

    slr_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    slr_c = torch.optim.lr_scheduler.LambdaLR(opt_c, lr_lambda)
    if resume_step:
        # 续训：两个 lr 调度快进到 resume_step（同 cd_gc；上游 dmd_arope 没快进 ⇒ 每次续训重走 100 步 warmup）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # "lr_scheduler.step() before optimizer.step()"
            for _ in range(resume_step):
                slr_g.step()
                slr_c.step()
        log(f"[resume] lr 调度快进到 step {resume_step}：lr_g={slr_g.get_last_lr()[0]:.3g} "
            f"lr_c={slr_c.get_last_lr()[0]:.3g}（Adam 状态不存，从零开始）")

    # FSDP 分支可以同时包多个 nn.Module；teacher 不进 prepare（冻结 bf16 整份复制）
    if ema is not None:
        student, critic, ema, opt_g, opt_c, dataloader = accel.prepare(
            student, critic, ema, opt_g, opt_c, dataloader)
    else:
        student, critic, opt_g, opt_c, dataloader = accel.prepare(
            student, critic, opt_g, opt_c, dataloader)
    teacher.to(device)
    for m in (student, critic, teacher) + ((ema,) if ema is not None else ()):
        unwrap(m).move_rope_table(device)
    assert unwrap(teacher).dit.freqs[0].device.type == device.type

    # ---- CFG 的 uncond 文本（cfg==1 时完全不需要）----
    uncond_row = None
    if abs(args.real_guidance_scale - 1.0) >= 1e-6:
        if args.dry_run:
            uncond_row = torch.randn(8, 4096, dtype=torch.bfloat16)
        else:
            table = DG.load_text_table(args.text_table)
            if NULL_PROMPT not in table:
                raise SystemExit(f"文本表里没有 NULL_PROMPT {NULL_PROMPT!r}，cfg>1 无法取 uncond")
            uncond_row = table[NULL_PROMPT]
        uncond_row = uncond_row.to(device=device, dtype=torch.bfloat16)
        log(f"[cfg] real_guidance_scale={args.real_guidance_scale}，uncond = NULL_PROMPT {tuple(uncond_row.shape)}")

    def uncond_cond(b: int, f: int):
        if uncond_row is None:
            return None, None
        return (uncond_row.view(1, 1, *uncond_row.shape).expand(b, f, *uncond_row.shape),
                torch.zeros(b, f, dtype=torch.int64, device=device))

    # ---- config.json ----
    if is_main:
        cfg = {
            "stage": STAGE, "res": GEO.RES_TAG, "args": {k: v for k, v in vars(args).items() if k != "betas"},
            "betas": list(args.betas), "preflight": pf,
            "deploy_schedule": {"t": [float(x) for x in t_dep], "sigma": [float(x) for x in sig_dep],
                                "idx": [int(x) for x in idx_dep]},
            "dataset": ({"synthetic": True} if args.dry_run else {
                "data_root": args.data_root, "synthetic": args.synthetic, "train_used": len(train_ds),
                "n_cells": n_cells, "text_table": args.text_table, "vocab": args.vocab}),
            "model": {"n_params": n_params, "student_ckpt": args.student_ckpt, "cond": "gc_sem_cond",
                      "teacher_ckpt": args.teacher_ckpt, "start_step": resume_step, "base_init": args.base_init},
            "env": {"world_size": accel.num_processes, "distributed_type": str(accel.distributed_type),
                    "mixed_precision": accel.mixed_precision, "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "python": sys.executable, "platform": platform.platform(), "hostname": socket.gethostname(),
                    "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        }
        try:
            import accelerate  # noqa: F401
            cfg["env"]["accelerate"] = accelerate.__version__
        except Exception:
            pass
        runlib.save_config(run_dir, cfg)

    csv_fields = ["step", "dmd", "critic", "dmd_gnorm", "gnorm_g", "gnorm_c", "N", "grad_start",
                  "exit_step", "n_eval", "lr_g", "lr_c", "replay_dev", "step_time", "mem_gb", "epoch"]
    train_log = runlib.CsvLog(os.path.join(run_dir, "train_log.csv"), csv_fields, enabled=is_main)

    def _clip(model, max_norm):
        # accel.clip_grad_norm_ 在 >1 个 FSDP 模型时会崩；FSDP 自带 clip_grad_norm_（处理分片梯度）
        if hasattr(model, "clip_grad_norm_"):
            return model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    ac = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16,      # noqa: E731
                                enabled=(device.type == "cuda"))
    # ★ rollout 区必须 cache_enabled=False（文件头坑 4）
    ac_roll = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16,  # noqa: E731
                                     enabled=(device.type == "cuda"), cache_enabled=False)

    # rank 同步的抽签流：N / grad_start / exit_step 三者必须**全 rank 一致**；每步恒定一次 randint(…, (3,))
    sync_rng = torch.Generator().manual_seed(4321 + args.seed)
    _SYNC_HI = 2 ** 31 - 1

    def _draw_sync():
        return torch.randint(0, _SYNC_HI, (3,), generator=sync_rng).tolist()

    for _ in range(resume_step):
        _draw_sync()

    def _sync3(n: int, gs: int, es: int) -> tuple[int, int, int]:
        if accel.num_processes <= 1:
            return n, gs, es
        t = torch.tensor([n, gs, es], device=device, dtype=torch.long)
        torch.distributed.broadcast(t, src=0)
        return int(t[0]), int(t[1]), int(t[2])

    def do_save(step: int):
        # save_gc 是集合操作（accelerator.get_state_dict），每个 rank 都必须进来
        meta = {"stage": STAGE, "step": step, "denoise_list": args.denoise_list,
                "kv_window": args.kv_window, "sink_size": args.sink_size, "rope_cap": args.rope_cap,
                "dmd_score_frames": args.dmd_score_frames, "grad_horizon": args.grad_horizon,
                "tail_mode": args.tail_mode, "rollout_min": args.rollout_min, "rollout_max": args.rollout_max,
                "student_ckpt": args.student_ckpt or "", "teacher_ckpt": args.teacher_ckpt or "",
                "vocab": args.vocab, "data": "v10" if not (args.dry_run or args.synthetic) else "synthetic"}
        GD.save_gc(accel, student, runlib.dit_path(run_dir, step), meta, dry_run=args.dry_run, log=log)
        GD.save_gc(accel, critic, runlib.critic_path(run_dir, step), {**meta, "role": "critic"},
                   dry_run=args.dry_run, log=log)
        if ema is not None:
            GD.save_gc(accel, ema, runlib.ema_path(run_dir, step),
                       {**meta, "role": "ema(deploy)", "ema_decay": args.ema_decay}, dry_run=args.dry_run, log=log)
        if is_main:
            gone = runlib.rotate_run(run_dir, keep_dits=args.keep_last_dits,
                                     keep_emas=args.keep_last_emas, keep_critics=args.keep_last_critics)
            n_gone = sum(len(v) for v in gone.values())
            if n_gone:
                log(f"[rotate] 删掉 {n_gone} 个旧 ckpt（dits {args.keep_last_dits} / "
                    f"ema {args.keep_last_emas} / critic {args.keep_last_critics}）")
            runlib.write_progress(run_dir, step, stage=STAGE)

    # ---- 训练循环 ----
    step, epoch, last_saved = resume_step, 0, resume_step
    _ls = resume_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"[train] 从 step {step} 到 {args.max_steps}")
    data_iter = iter(dataloader)
    t_step = time.time()
    last = {}
    kv = dict(kv_window=args.kv_window, rope_cap=args.rope_cap, sink_size=args.sink_size)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = to_device(batch, device)
        if device.type != "cuda":                      # CPU 路径把输入跟着提到学生的 dtype
            pdt = next(unwrap(student).parameters()).dtype
            for _k in ("z0", "context"):
                batch[_k] = batch[_k].to(pdt)
        initial = batch["z0"]                                    # [b,48,1,h,w] 真实干净首帧
        b, _, _, lat_h, lat_w = initial.shape

        # ---- rollout 长度 N / grad 窗口 / 退出档（三者都必须 rank 同步）----
        r0, r1, r2 = _draw_sync()
        N = (int(args.num_gen_cells) if args.num_gen_cells is not None
             else args.rollout_min + r0 % (args.rollout_max - args.rollout_min + 1))
        total = 1 + N
        Keff = min(args.dmd_score_frames, N) if args.dmd_score_frames > 0 else None
        lo_gs, hi_gs = grad_start_range(total, args.grad_horizon, Keff)
        grad_start = lo_gs + r1 % (hi_gs - lo_gs + 1)
        exit_step = (r2 if args.exit_step is None else int(args.exit_step)) % len(t_dep)
        N, grad_start, exit_step = _sync3(N, grad_start, exit_step)
        total = 1 + N

        # ---- 1+N 个 cell 的真实条件，逐下标锁死 ----
        sem_ext, ctx_ext, ids_ext = DG.slice_rollout(batch, N)

        if step == resume_step:
            # 开跑前问一次"梯度到底能不能到学生参数"（step 0 的 dmd 梯度必然是 0，问不出来）
            with ac_roll():
                probe_gn = probe_student_grad(student, initial, ctx_ext, ids_ext, sem_ext, deploy,
                                              accel.backward, _clip, **kv)
            opt_g.zero_grad(set_to_none=True)
            log(f"[probe] rollout 梯度可达性自检 gnorm={probe_gn:.4e}")
            if not (probe_gn > 0 and math.isfinite(probe_gn)):
                raise SystemExit(f"[probe] 学生的 rollout 梯度范数 = {probe_gn}：梯度根本没到学生参数。"
                                 "依次查 fsdp_use_orig_params、rollout 区 autocast 的 cache_enabled、多余的 detach。")
            if resume_step == 0 and not args.dry_run:          # 续训 / dry run 的 critic ≠ teacher
                log("[note] step 0 的 dmd / dmd_gnorm / gnorm_g 都会恰好是 0：critic 与 teacher 是同一份双向 SFT 初值，"
                    "pred_fake ≡ pred_real，重放逐位等于 rollout。dmd 应当随 critic 被拉开而升到 +0.15…+0.68，不是单调下降。")

        _prof = bool(args.profile_mem) and step < resume_step + 3 and is_main and device.type == "cuda"

        def _mem(tag):
            if _prof:
                torch.cuda.synchronize(device)
                log(f"  [mem step{step}] {tag}: peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f}GB "
                    f"cur {torch.cuda.memory_allocated(device) / 2 ** 30:.1f}GB")

        if args.profile_mem and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # streamed：rollout 完全不建图，之后每个 grad 帧单独重做 + backward（同一个梯度）
        record = {"exit": {}, "commit": {}} if args.grad_pass == "streamed" else None
        t_roll = time.time()
        with ac_roll():
            gen_clip, grad_mask = self_forcing_rollout(
                student, initial, ctx_ext, ids_ext, sem_ext, N, deploy, args.grad_horizon,
                score_last_k=Keff, record=record, grad_start=grad_start, exit_step=exit_step, **kv)
        t_roll = time.time() - t_roll
        if step == resume_step and gen_clip.dtype != initial.dtype:
            raise SystemExit(f"[dtype] rollout 出来的 latent 是 {gen_clip.dtype}，z0 是 {initial.dtype}：t 会离开部署的 bf16 网格")
        _mem("after rollout")

        tail_sl, cond_t, grad_mask_t, tail_to_frame = score_tail(
            gen_clip, grad_mask, sem_ext, ctx_ext, ids_ext, N, Keff, tail_mode=args.tail_mode)

        # ---- 生成器先更新（学生 backward 必须在 critic backward 之前），且**每步**都更新（FSDP backward setup）----
        u_ctx, u_ids = uncond_cond(b, tail_sl.shape[2])
        if record is None:
            with ac():
                gl, glog = dmd_generator_loss(
                    teacher, critic, tail_sl, cond_t, grad_mask_t,
                    real_guidance_scale=args.real_guidance_scale,
                    uncond_context=u_ctx, uncond_ids=u_ids, clamp=args.clamp)
            _mem("after dmd_loss")
            accel.backward(gl)
            eval_idx = torch.nonzero(grad_mask_t, as_tuple=False).flatten()
            eval_idx = eval_idx[eval_idx >= 1]
            gl_val = float(gl.detach())
        else:
            with ac():
                target_t, eval_idx, glog = dmd_targets(
                    teacher, critic, tail_sl, cond_t, grad_mask_t,
                    real_guidance_scale=args.real_guidance_scale,
                    uncond_context=u_ctx, uncond_ids=u_ids, clamp=args.clamp)
            _mem("after dmd_loss")
            record["n_eval"] = int(eval_idx.numel())
            frame_to_tail = {tail_to_frame[int(j)]: int(j) for j in eval_idx}

            def _tgt(k, _t=target_t, _m=frame_to_tail):
                j = _m.get(k)
                return None if j is None else _t[:, :, j:j + 1]

            with ac_roll():
                gl_val = stream_grad_replay(student, initial, ctx_ext, ids_ext, sem_ext, record,
                                            deploy, _tgt, accel.backward, **kv)
            del target_t
        _mem("after gen backward")
        # ★ opt_g.step 之前释放 FSDP 根单元留着的 unsharded 参数，否则下一步的 rollout 用的是这一步更新前的根参数
        freed = _free_fsdp_root(student)
        if step == resume_step:
            log(f"[fsdp] 生成器 step 前释放根单元的 unsharded 参数：{'是' if freed else '否（非 FSDP 或已释放）'}")
        gng = float(_clip(student, args.grad_clip))
        replay_dev = float(record.get("replay_dev", 0.0)) if record is not None else 0.0
        # ★ S3-GUARD-3：step > resume_step 时 critic 已走过 critic_steps 步 ⇒ gnorm_g > 0 是硬不变量。
        #   重放逐位等于 rollout（replay_dev == 0）时它没有噪声底：critic ≡ teacher 的 step 0 恰好是 0。
        if step > resume_step and (gng == 0.0 or math.isnan(gng)):
            raise SystemExit(f"[gnorm] generator grad norm {gng} at step {step}: student receives no gradient "
                             "(check FSDP backward setup / fsdp_use_orig_params / autocast cache / stray detach)")
        if replay_dev > 0.0 and step < resume_step + 5:
            cells = {c: (f"{mx:.3g}", f"{fr:.1e}") for c, (mx, fr) in sorted(record["replay_dev_cells"].items())}
            log(f"[warn] streamed 重放的 den 与 rollout 差 {replay_dev:.3g}（应当恰好是 0）：surrogate 带一项伪梯度。"
                f"逐 cell (max|Δ|, 不等元素占比)：{cells}")
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        slr_g.step()
        if ema is not None:                                     # 每个生成器步更新一次
            ema_update(ema, student, decay=(0.0 if step < args.ema_start else args.ema_decay))
        _mem("after opt_g.step")

        # ---- critic 更新：在**同一份 detach 样本**上跑 critic_steps 次 ----
        gen_det = tail_sl.detach()
        gnc, cl_val = float("nan"), float("nan")
        for _ci in range(args.critic_steps):
            with ac():
                cl, _ = critic_loss(critic, gen_det, cond_t, eval_idx, clamp=args.clamp,
                                    use_gradient_checkpointing=args.gradient_checkpointing)
            accel.backward(cl)
            _free_fsdp_root(critic)                             # backward 已释放时是空操作；防将来在这之间插前向
            gnc = float(_clip(critic, args.grad_clip))
            opt_c.step()
            opt_c.zero_grad(set_to_none=True)
            cl_val = float(cl.detach())
        slr_c.step()
        _mem("after critic")

        step += 1
        last = {"dmd": gl_val, "critic": cl_val, "gnorm_g": gng, "gnorm_c": gnc,
                "N": N, "grad_start": grad_start, "exit_step": exit_step, "replay_dev": replay_dev,
                "n_eval": int(eval_idx.numel()), "rollout_s": t_roll, **glog}
        if step % args.log_every == 0 or step == args.max_steps:
            now = time.time()
            sec = (now - t_step) / max(step - _ls, 1)
            t_step, _ls = now, step
            avg_g = accel.reduce(torch.tensor(gl_val, device=device), reduction="mean").item()
            avg_c = accel.reduce(torch.tensor(cl_val, device=device), reduction="mean").item()
            row = {"step": step, "dmd": avg_g, "critic": avg_c, "dmd_gnorm": last["dmd_gnorm"],
                   "gnorm_g": gng, "gnorm_c": gnc, "N": N, "grad_start": grad_start,
                   "exit_step": exit_step, "n_eval": last["n_eval"],
                   "lr_g": slr_g.get_last_lr()[0], "lr_c": slr_c.get_last_lr()[0], "replay_dev": replay_dev,
                   "step_time": sec,
                   "mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                              if device.type == "cuda" else 0.0), "epoch": epoch}
            if is_main:
                train_log.write(row)
                jsonl.log({**row, "rollout_s": t_roll})
            log(f"step {step}/{args.max_steps}  dmd {avg_g:+.4g}  critic {avg_c:.4f}  "
                f"dmd_gnorm {last['dmd_gnorm']:.3g}  gnorm_g {gng:.3g} gnorm_c {gnc:.3g}  replay_dev {replay_dev:.3g}  "
                f"N {N} grad {grad_start}..{grad_start + args.grad_horizon} exit {exit_step} "
                f"n_eval {last['n_eval']}  lr_g {row['lr_g']:.3g}  rollout {t_roll:.1f}s  {sec:.1f}s  "
                f"mem {row['mem_gb']:.1f}GB")
        if (args.save_every > 0 and step % args.save_every == 0) or step == args.max_steps:
            do_save(step)
            last_saved = step

    if last_saved != step and step > resume_step:
        do_save(step)
    if device.type == "cuda":
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2 ** 30, device=device)
        peak_max = accel.reduce(peak, reduction="max").item()
    else:
        peak_max = 0.0
    log(f"[done] step {step}，各进程峰值显存 {peak_max:.1f} GB。"
        f"部署权重按 ema/ema_step*.safetensors **逐个里程碑挑**（不是拿最后一步），用 gcdistill/eval_rollout.py 量长链。")
    if is_main:
        runlib.write_progress(run_dir, step, stage=STAGE, finished=True, peak_mem_gb=peak_max, last=last)
    train_log.close()
    jsonl.close()
    log.close()
    accel.wait_for_everyone()
    accel.end_training()
    return step


if __name__ == "__main__":
    main()
