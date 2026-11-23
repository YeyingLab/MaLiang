#!/usr/bin/env python3
"""逐 cell（per-frame）的 flow-matching 工具，全部建在 DiffSynth 的 FlowMatchScheduler("Wan") 上。

ARoPE 的 SFT（actionrope/train.py:218-219）用的就是 `FlowMatchScheduler("Wan")` +
`set_timesteps(1000, training=True)`，网格是：

    s      = linspace(1, 0, 1001)[:-1]                    # 1000 个点
    sigma  = 5·s / (1 + 4·s)                              # Wan shift=5
    t      = sigma · 1000                                 # 与 sigma 一一对应
    x_t    = (1 − sigma)·x0 + sigma·eps
    target = eps − x0
    weight = bsmntw 钟形（set_training_weight）

蒸馏要的东西 DiffSynth 没有：它的 `add_noise` / `training_weight` 只吃**标量** timestep（内部
argmin 反查下标），没有 `[B,F]` 的逐 cell 版本。这里补上，并且一律**按下标**索引，不再 argmin
反查（浮点反查在 1000 点网格上是无谓的风险）。

方向陷阱（写反了不会报错，只会悄悄训错噪声档）
------------------------------------------------
    sigmas[0]   = 1.0      纯噪声   t = 1000
    sigmas[999] ≈ 0.00498  近干净   t ≈ 4.98
即 **越噪 == sigma 越大 == 下标越小**。an earlier internal project 的 `--tf_noise_aug_max_idx` 就是把这个搞反了
（近干净档应该是 `randint(N-k, N)`，不是 `randint(0, k)`）。

模块级 scheduler 的不变量（对抗评审 [MINOR]，必须遵守）
--------------------------------------------------------
DiffSynth 的 `set_timesteps` 是**破坏性**的：它就地覆写 `self.sigmas` / `self.timesteps`，而且
只在 `training=True` 时重建 `linear_timesteps_weights`（flow_match.py:365-375）。an earlier internal project 自己
那份写进独立的 `infer_sigmas` / `infer_timesteps`，不碰训练网格 —— 移植时**不能**照抄那个用法。
所以这里：

  * 模块级 `_SCHED` 只在 import 时 `set_timesteps(1000, training=True)` 一次；
  * 之后 `_SCHED.set_timesteps` 被实例级替换成一个直接抛异常的桩，物理上堵死第二次调用；
  * `cd_axis()` / `deploy_schedule()` 这类需要别的点数的，一律**新建一个临时 scheduler**；
  * 每个 public 函数开头都跑 `_assert_sched()`：断言 `len(timesteps)==1000`、
    `linear_timesteps_weights is not None`，并且三个张量与 import 时的快照逐位相等。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from diffsynth.diffusion.flow_match import FlowMatchScheduler  # noqa: E402

__all__ = [
    "NUM_TRAIN_TIMESTEPS", "WAN_SHIFT", "DMD_TIMESTEP_CLAMP", "DEPLOY_DENOISE_LIST",
    "sched", "sigmas", "timesteps", "weights",
    "sigma_of_idx", "t_of_idx", "idx_of_t", "sigma_of_t",
    "sample_pf_timestep_ids", "add_noise_pf", "training_target", "training_weight_pf",
    "cd_axis", "deploy_schedule", "assert_deploy_covered",
    "ti2v_noise", "assert_ti2v",
]

NUM_TRAIN_TIMESTEPS = 1000
WAN_SHIFT = 5.0
# stage3 DMD 抽 t 的下标区间（an earlier internal project dmd_gm.py:47-50 的 int(0.02·n)/int(0.98·n)，n=1000）。
# 是**下标**区间不是 t 值区间：idx∈[20,980) ⇒ sigma∈(0.0969, 0.9959] ⇒ t∈(96.9, 995.9]。
DMD_TIMESTEP_CLAMP = (20, 980)
# 用户拍板的部署采样表：4 步，原始 t 值（不是下标）。stage2 的 CD 轴和 stage3 都按这个训。
# 用显式 t 值而不是 set_timesteps(4)：后者在 shift=5 下给出 [1000, 833, 625, 357]，最后一步在
# sigma=0.357 处读 x0，读出来是条件均值（软/糊）。见 an earlier internal project dmd_gm.py:461-467 的同一坑。
DEPLOY_DENOISE_LIST = (1000.0, 750.0, 500.0, 250.0)


# ----------------------------------------------------------------------------
# 模块级 scheduler —— 建一次，之后只读
# ----------------------------------------------------------------------------

def _forbidden_set_timesteps(*a, **kw):
    raise RuntimeError(
        "distill.flow_pf: 模块级 scheduler 是 1000 点训练网格，set_timesteps 会就地覆写它。"
        "需要别的点数请用 cd_axis(N) / deploy_schedule()（它们新建临时实例）。"
    )


_SCHED = FlowMatchScheduler("Wan")
_SCHED.set_timesteps(NUM_TRAIN_TIMESTEPS, training=True)
_SCHED.set_timesteps = _forbidden_set_timesteps  # 实例级遮蔽类方法，堵死第二次调用

# import 时的快照：_assert_sched 拿它逐位比对，任何外部改写都当场炸而不是悄悄算错
_SNAP_SIGMAS = _SCHED.sigmas.detach().clone()
_SNAP_TIMESTEPS = _SCHED.timesteps.detach().clone()
_SNAP_WEIGHTS = _SCHED.linear_timesteps_weights.detach().clone()


def _assert_sched() -> FlowMatchScheduler:
    """每个 public 函数开头都调：确认 1000 点训练网格和钟形权重都还在、且没被人改过。"""
    assert getattr(_SCHED, "timesteps", None) is not None, "flow_pf: scheduler 未初始化"
    assert len(_SCHED.timesteps) == NUM_TRAIN_TIMESTEPS, \
        f"flow_pf: 训练网格被覆写了，len(timesteps)={len(_SCHED.timesteps)} != {NUM_TRAIN_TIMESTEPS}"
    assert len(_SCHED.sigmas) == NUM_TRAIN_TIMESTEPS, \
        f"flow_pf: 训练网格被覆写了，len(sigmas)={len(_SCHED.sigmas)} != {NUM_TRAIN_TIMESTEPS}"
    assert getattr(_SCHED, "linear_timesteps_weights", None) is not None, \
        "flow_pf: linear_timesteps_weights 丢了（set_timesteps(training=False) 不重建它）"
    assert torch.equal(_SCHED.sigmas, _SNAP_SIGMAS) and torch.equal(_SCHED.timesteps, _SNAP_TIMESTEPS) \
        and torch.equal(_SCHED.linear_timesteps_weights, _SNAP_WEIGHTS), \
        "flow_pf: 模块级 scheduler 的张量被就地改写过"
    return _SCHED


def sched() -> FlowMatchScheduler:
    """取模块级（唯一的）训练 scheduler，取之前先验不变量。"""
    return _assert_sched()


def sigmas(device=None, dtype=torch.float32) -> torch.Tensor:
    """1000 点 sigma 网格 [1000]，降序：sigmas[0]=1.0（纯噪声）→ sigmas[999]≈0.00498（近干净）。"""
    return _assert_sched().sigmas.to(device=device, dtype=dtype)


def timesteps(device=None, dtype=torch.float32) -> torch.Tensor:
    """1000 点 t 网格 [1000] = sigma·1000，降序：t[0]=1000 → t[999]≈4.98。"""
    return _assert_sched().timesteps.to(device=device, dtype=dtype)


def weights(device=None, dtype=torch.float32) -> torch.Tensor:
    """bsmntw 钟形训练权重 [1000]（DiffSynth set_training_weight，len==1000 时不走经验缩放分支）。"""
    return _assert_sched().linear_timesteps_weights.to(device=device, dtype=dtype)


# ----------------------------------------------------------------------------
# 下标 ↔ sigma ↔ t
# ----------------------------------------------------------------------------

def _as_long(idx, device=None) -> torch.Tensor:
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx)
    if idx.dtype not in (torch.int32, torch.int64):
        assert not idx.is_floating_point(), "flow_pf: 下标必须是整型（浮点 t 值请走 idx_of_t）"
        idx = idx.long()
    idx = idx.long()
    return idx if device is None else idx.to(device)


def _check_idx(idx: torch.Tensor) -> torch.Tensor:
    lo, hi = int(idx.min()), int(idx.max())
    assert 0 <= lo and hi < NUM_TRAIN_TIMESTEPS, \
        f"flow_pf: 下标越界 [{lo},{hi}]，合法区间 [0,{NUM_TRAIN_TIMESTEPS - 1}]"
    return idx


def _gather(grid: torch.Tensor, idx, device, dtype) -> torch.Tensor:
    idx = _check_idx(_as_long(idx))
    out = grid.to(device=idx.device, dtype=torch.float32)[idx]
    if device is not None:
        out = out.to(device)
    return out.to(dtype)


def sigma_of_idx(idx, device=None, dtype=torch.float32) -> torch.Tensor:
    """下标 → sigma，形状原样保留（标量 int 给回 0 维张量）；device=None 表示跟 idx 走。"""
    return _gather(_assert_sched().sigmas, idx, device, dtype)


def t_of_idx(idx, device=None, dtype=torch.float32) -> torch.Tensor:
    """下标 → t 值（= sigma·1000），形状原样保留。stage1 的 ts_pf 就是 t_of_idx(idx_pf)。"""
    return _gather(_assert_sched().timesteps, idx, device, dtype)


def idx_of_t(t) -> torch.Tensor:
    """原始 t 值 → 1000 点网格上最近的下标（an earlier internal project dmd_gm.py:104 的 `_sigma_of` 反查）。

    形状原样保留；标量 float 给回 0 维 long 张量。部署表 [1000,750,500,250] 就是靠这个落到
    sigma [1.0, 0.75, 0.5, 0.25]（750 在网格上是精确点，500/250 差 <5e-4）。"""
    s = _assert_sched()
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.to(torch.float32)
    grid = s.timesteps.to(device=t.device, dtype=torch.float32)
    return (grid - t.unsqueeze(-1)).abs().argmin(dim=-1)


def sigma_of_t(t, device=None, dtype=torch.float32) -> torch.Tensor:
    """原始 t 值 → sigma（先 idx_of_t 再查表）。stage3 rollout 的 `_sigma_of`。"""
    return sigma_of_idx(idx_of_t(t), device=device, dtype=dtype)


# ----------------------------------------------------------------------------
# 逐 cell 加噪（Diffusion Forcing / stage1 teacher forcing）
# ----------------------------------------------------------------------------

def sample_pf_timestep_ids(B: int, F: int, dev, gen: torch.Generator | None = None) -> torch.Tensor:
    """[B,F] 独立同分布的逐 cell 噪声档下标，**严格均匀** randint(0,1000)。

    不要换成 `round((1-u)·999)` 那种写法：它把下标 0 和 999 的概率各砍一半。"""
    _assert_sched()
    assert B > 0 and F > 0
    g_dev = dev if gen is None else gen.device
    idx = torch.randint(0, NUM_TRAIN_TIMESTEPS, (B, F), device=g_dev, generator=gen)
    return idx.to(dev)


def add_noise_pf(x0: torch.Tensor, noise: torch.Tensor, idx_pf: torch.Tensor) -> torch.Tensor:
    """逐 cell 加噪：x_t = (1−sigma)·x0 + sigma·eps，sigma 按 [B,1,F,1,1] 广播。

    x0 / noise: [B,C,F,H,W]；idx_pf: [B,F] long。返回 x0.dtype。

    为什么系数在 fp32 里算完再 cast：sigma 接近 1 时 `1 − bf16(0.99593) = 0.00391`，而真值是
    0.00407 —— bf16 直接相减是灾难性抵消，4% 的相对误差直接打在 x0 的系数上。fp32 里先算
    `1−sigma` 再 cast 成 bf16 是精确的（bf16 表示小数本身没问题），这也正是 DiffSynth 标量
    `add_noise` 的实际行为（0 维 fp32 标量参与类型提升，先算差再降位）。"""
    _assert_sched()
    assert x0.dim() == 5, f"flow_pf: x0 需要 [B,C,F,H,W]，得到 {tuple(x0.shape)}"
    assert noise.shape == x0.shape, f"flow_pf: noise {tuple(noise.shape)} != x0 {tuple(x0.shape)}"
    assert noise.dtype == x0.dtype, f"flow_pf: noise dtype {noise.dtype} != x0 {x0.dtype}"
    idx_pf = _check_idx(_as_long(idx_pf, x0.device))
    assert idx_pf.dim() == 2 and idx_pf.shape[0] == x0.shape[0] and idx_pf.shape[1] == x0.shape[2], \
        f"flow_pf: idx_pf 需要 [B,F]=[{x0.shape[0]},{x0.shape[2]}]，得到 {tuple(idx_pf.shape)}"
    b, f = idx_pf.shape
    sig = _SCHED.sigmas.to(device=x0.device, dtype=torch.float32)[idx_pf].view(b, 1, f, 1, 1)
    return (1.0 - sig).to(x0.dtype) * x0 + sig.to(x0.dtype) * noise


def training_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """flow-matching 目标：eps − x0（与噪声档无关，DiffSynth training_target 同）。"""
    assert noise.shape == x0.shape
    return noise - x0


def training_weight_pf(idx_pf: torch.Tensor) -> torch.Tensor:
    """逐 cell 的 bsmntw 钟形权重，形状与 idx_pf 相同，fp32。中段 t≈500 最重、两端最轻。"""
    _assert_sched()
    idx = _check_idx(_as_long(idx_pf))
    w = _SCHED.linear_timesteps_weights.to(device=idx.device, dtype=torch.float32)
    return w[idx]


# ----------------------------------------------------------------------------
# stage2 的一致性轴 / 部署采样表（都用**临时** scheduler，绝不碰模块级那个）
# ----------------------------------------------------------------------------

def cd_axis(N: int = 48, device=None, check_deploy: bool = True):
    """stage2 CD 的离散一致性轴：`FlowMatchScheduler("Wan").set_timesteps(N)` 的 (sigmas, timesteps)。

    N=48 时 sigmas = [1.0, 0.99576, …, 0.17857, 0.09615]，t = sigma·1000，都是**降序**（下标越大越干净）。
    轴永远到不了 sigma=0（最低一对是 0.17857 → 0.09615），所以部署采样表必须结束在低噪端。

    临时实例：DiffSynth 的 set_timesteps 就地覆写 sigmas/timesteps，在模块级 scheduler 上调一次
    就把 1000 点训练网格换成 N 点的，之后每次 idx 抽样要么读错 sigma 要么直接越界（对抗评审 [MINOR]）。

    check_deploy：顺带断言这条轴的 sigma 区间覆盖 DEPLOY_DENOISE_LIST 的所有读点。"""
    _assert_sched()
    assert N >= 2, f"flow_pf: cd_axis 需要 N>=2，得到 {N}"
    tmp = FlowMatchScheduler("Wan")           # 一次性实例，出了这个函数就被回收
    tmp.set_timesteps(N)                      # training=False：轴上不需要钟形权重
    sig = tmp.sigmas.detach().clone().to(device=device, dtype=torch.float32)
    tst = tmp.timesteps.detach().clone().to(device=device, dtype=torch.float32)
    assert sig.shape == (N,) and tst.shape == (N,), f"flow_pf: cd_axis 点数不对 {tuple(sig.shape)}"
    assert bool((sig[:-1] > sig[1:]).all()), "flow_pf: cd_axis 的 sigma 必须严格降序"
    if check_deploy:
        assert_deploy_covered(sig)
    _assert_sched()                           # 再验一次：临时实例没污染模块级的
    return sig, tst


def deploy_schedule(denoise_list=None, device=None):
    """部署采样表（原始 t 值）→ 落到 1000 点训练网格上的 (idx[K], sigma[K], t_grid[K])。

    默认就是用户拍板的 4 步 `DEPLOY_DENOISE_LIST = (1000, 750, 500, 250)`。
    返回的 t_grid 是网格上的实际值（1000 / 750 / 500.0? / 250.0? 见下），采样时应当用它而不是
    原始输入值，这样 sigma 与 t 严格一一对应。"""
    _assert_sched()
    ds = tuple(DEPLOY_DENOISE_LIST if denoise_list is None else denoise_list)
    assert len(ds) >= 1, "flow_pf: 部署表不能为空"
    t_req = torch.as_tensor([float(x) for x in ds], dtype=torch.float32)
    if len(ds) > 1:
        assert bool((t_req[:-1] > t_req[1:]).all()), \
            f"flow_pf: 部署表必须严格降序（从噪到净），得到 {ds}"
    idx = idx_of_t(t_req)
    sig = sigma_of_idx(idx)
    t_grid = t_of_idx(idx)
    # 落点误差不该超过半个格。注意网格在 t 轴上**不等距**（sigma 等距、t=sigma·1000 也等距在
    # sigma 上，但相邻 sigma 之间的间隔是 5/(1+4s)²·0.001，从噪端 0.2 涨到净端 ~4.98），
    # 所以阈值取全局最大间隔的一半：请求 t=250 落到 251.6 是正常的，请求 t=1.0（低于网格下界
    # 4.98）才会被拦下。
    gap = float((_SCHED.timesteps[:-1] - _SCHED.timesteps[1:]).max())
    assert bool((t_grid - t_req).abs().max() <= 0.5 * gap + 1e-3), \
        f"flow_pf: 部署 t 值 {ds} 落不到网格上（最近点 {t_grid.tolist()}，半格 {0.5 * gap:.3f}）"
    if device is not None:
        idx, sig, t_grid = idx.to(device), sig.to(device), t_grid.to(device)
    return idx, sig, t_grid


def assert_deploy_covered(sig_axis: torch.Tensor, denoise_list=None) -> None:
    """断言 CD 一致性轴的 sigma 区间覆盖部署表的每个读点。

    覆盖不到意味着：学生在部署要读 x0 的那个噪声档上，从来没被一致性目标约束过。
    N=48 轴是 [0.09615, 1.0]，部署表最低读点 sigma=0.25 —— 覆盖住。"""
    _, sig_dep, _ = deploy_schedule(denoise_list)
    lo, hi = float(sig_axis.min()), float(sig_axis.max())
    d_lo, d_hi = float(sig_dep.min()), float(sig_dep.max())
    assert d_lo >= lo - 1e-6, \
        f"flow_pf: 部署表最低 sigma {d_lo:.5f} 低于 CD 轴下界 {lo:.5f}，该档没被一致性约束过"
    assert d_hi <= hi + 1e-6, \
        f"flow_pf: 部署表最高 sigma {d_hi:.5f} 高于 CD 轴上界 {hi:.5f}"


# ----------------------------------------------------------------------------
# stage3 DMD 的原生 TI2V 加噪
# ----------------------------------------------------------------------------

def ti2v_noise(clip: torch.Tensor, clamp=DMD_TIMESTEP_CLAMP, gen: torch.Generator | None = None):
    """原生 TI2V 加噪（= ARoPE SFT / infer 的条件形式）：cell 0 干净且 t=0，cell 1..F−1
    共享**一个**抽出来的 t。an earlier internal project dmd_gm.py:53-66 的移植。

    clip: [B,C,F,H,W]（可以带梯度：cell 0 用 torch.where 写回，不做原地赋值）
    clamp: **下标**区间 [lo, hi)，默认 (20, 980) —— 即 an earlier internal project 的 [0.02, 0.98]·1000。
    返回 (noisy, ts_pf[B,F] fp32, sigma[B] fp32, noise)；noise 与 clip 同 dtype/shape。"""
    _assert_sched()
    assert clip.dim() == 5, f"flow_pf: clip 需要 [B,C,F,H,W]，得到 {tuple(clip.shape)}"
    b, _, f = clip.shape[:3]
    lo, hi = int(clamp[0]), int(clamp[1])
    assert 0 <= lo < hi <= NUM_TRAIN_TIMESTEPS, f"flow_pf: clamp 下标区间不合法 {clamp}"
    dev = clip.device
    g_dev = dev if gen is None else gen.device
    idx = torch.randint(lo, hi, (b,), device=g_dev, generator=gen).to(dev)
    noise = torch.randn(clip.shape, device=g_dev, dtype=clip.dtype, generator=gen).to(dev)
    sig = _SCHED.sigmas.to(device=dev, dtype=torch.float32)[idx]                 # [B] fp32
    s5 = sig.view(b, 1, 1, 1, 1)
    noisy = (1.0 - s5).to(clip.dtype) * clip + s5.to(clip.dtype) * noise
    ts_pf = _SCHED.timesteps.to(device=dev, dtype=torch.float32)[idx][:, None].expand(b, f).clone()
    ts_pf[:, 0] = 0.0
    f0 = (torch.arange(f, device=dev)[None, :] == 0).view(1, 1, f, 1, 1)
    noisy = torch.where(f0, clip, noisy)                                        # cell 0 = 干净 GT
    return noisy, ts_pf, sig, noise


def assert_ti2v(noisy: torch.Tensor, clip: torch.Tensor, ts_pf: torch.Tensor) -> None:
    """teacher / critic 每次前向前的 fail-fast 门（an earlier internal project dmd_gm.py:31-42 的输入部分）：
    cell 0 必须干净且 t=0，cell 1.. 必须共享同一个 t。"""
    assert noisy.shape == clip.shape and noisy.dim() == 5
    assert ts_pf.shape == (clip.shape[0], clip.shape[2]), \
        f"[gate] ts_pf 需要 [B,F]={(clip.shape[0], clip.shape[2])}，得到 {tuple(ts_pf.shape)}"
    assert bool((ts_pf[:, 0] == 0).all()), "[gate] cell 0 必须是干净条件帧（t=0）"
    assert torch.equal(noisy[:, :, 0], clip[:, :, 0]), "[gate] cell 0 的 latent 必须原样是干净 GT"
    if ts_pf.shape[1] > 1:
        assert torch.allclose(ts_pf[:, 1:], ts_pf[:, 1:2].expand_as(ts_pf[:, 1:])), \
            "[gate] cell 1..F−1 必须共享同一个 t（原生 TI2V）"
