#!/usr/bin/env python3
"""滑窗 KV 的**启动前自检**：亚秒级、纯 CPU、tiny 模型，在烧掉任何一次大分配之前把
sink / 驱逐 / 窗口锚定的算术跑一遍。stage3 启动时第一件事就调它。
（部署配置 sink=0/window=20/cap=20 实测 ~0.4 s 含建模型、~0.2 s 复用模型；
对照的是 4 个 5B 模型 ~40 GB 的加载与 FSDP 包装。）

    from distill.tests.preflight import preflight_sliding_kv
    preflight_sliding_kv(sink_size=0, kv_window=20, rope_cap=20)      # stage3 的部署配置

移植自 `_wm/GraphGWM/train/code_distill/stage3_dmd/dmd_gm.py:446`（在 `:861` 被调用）。
上游那一版存在的理由是一个具体的 bug：滑窗读缓存时算
`R = k_hist.shape[1] // hw - sink`，sink>1 时 rollout 早期缓存里根本不够 sink 帧，
切片给出 n_cached*hw 个 token 而 RoPE 网格按 sink*hw 建 —— frame_index=1 上直接形状崩。
an earlier internal project 原版有 `n_sink = min(sink, n_cached)` 这一行，GraphGWM 抄漏了，而 SFT 从不用 KV 缓存，
所以这个 bug 在整条 SFT 链路上是隐形的。`distill/arope_dit.py:638` 保留了那一行守卫，
本文件就是**证明它还在**的那个毫秒级证据。

ARoPE 比上游多两类必须自检的算术（world RoPE 的重新设计，PORT_设计 risk 3）：

* **窗口锚定坐标**。缓存里留哪几个绝对 cell 完全由 sink/驱逐决定（`retained_cells`），
  `offset_win` 必须逐格对上这个列表并且减掉最老那一格（`offset_win[:,0] == 0`，
  `arope_dit.py:628` 硬断言）。差一格不会报错——它会让世界坐标整体错位一个 cell，
  rollout 照跑、loss 照降，训的是错的几何。
* **sink 与漂移坐标不相容**。sink_size>0 时 cell 0 被永久留在缓存里，于是窗口跨度
  = 整条 rollout 的位移，30 个 cell 之后是训练分布外的几十个 token
  （`tests/test_world_rope_window.py` 量了这条）。stage3 argparse 默认 sink_size=0，正式训练/部署是 sink 3 / window 17，
  这里对 sink_size>0 只做算术自检、并在日志里点名。

为什么是 tiny CPU 模型而不是真权重
--------------------------------
自检要在 `WanVideoPipeline.from_pretrained`（多个大模型）之前跑完，否则"先分配再发现
配置是错的"就毫无意义。dim 24 / 2 层 / latent 4×4 ⇒ 一次 rollout 22 帧仍是毫秒级，
而 **算术**（n_cached、驱逐、tpos、锚定 offset 的形状与值）与 5B 上完全一样。
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from actionrope.arope import PX_PER_LATENT, install_arope, known_mask  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from distill.arope_dit import AropeDiT, unwrap  # noqa: E402
from distill.data import WALK_SPEED_PX_S, window_offsets  # noqa: E402

__all__ = [
    "retained_cells", "window_offsets_for_frame", "cache_len",
    "tiny_dit", "tiny_arope_dit", "preflight_sliding_kv",
]

# stage3 的部署配置（PORT_设计 stage3）。rope_cap == sink + kv_window == F-1 == 20 是硬不变量。
DEFAULT_SINK, DEFAULT_KV_WINDOW, DEFAULT_ROPE_CAP = 0, 20, 20

# tiny 模型几何：latent 4×4 → patch (1,2,2) → 2×2 = 4 token/cell。
# head_dim 必须让 precompute_freqs_cis_3d 的三段列数加起来正好是 head_dim/2：
# head_dim=12 ⇒ f 段 12-2*(12//3)=4 → 2 列，h/w 段各 4 → 2 列，合计 6 = 12/2 ✓。
# （head_dim=16 就不行：6/2+5/2*2 = 3+2+2 = 7 ≠ 8，rope_apply 会形状崩。）
TINY = dict(dim=24, in_dim=48, ffn_dim=48, out_dim=48, text_dim=4096, freq_dim=256,
            eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=False)
TINY_LAT_H, TINY_LAT_W, TINY_L_TXT = 4, 4, 4

_TINY_CACHE: dict = {}


# --------------------------------------------------------------------------
# 缓存内容的绝对 cell 下标（stage3 每帧都要用它来切 offset / 文本）
# --------------------------------------------------------------------------

def retained_cells(frame_index: int, sink_size: int = DEFAULT_SINK,
                   kv_window: int | None = DEFAULT_KV_WINDOW) -> list[int]:
    """**要前向 cell k 的那一刻**，缓存里还留着哪几个绝对 cell（升序，长度 = n_cached）。

    提交顺序是 0,1,2,...；`arope_dit._evict` 保留开头 sink 个、丢掉第 sink 个，直到长度回到
    sink + kv_window。所以留下来的恒等于 **开头 sink 个 + 最近 kv_window 个**：

        k <= sink + kv_window   ⇒ [0, k)                         （还没满，没驱逐）
        k >  sink + kv_window   ⇒ [0, sink) ∪ [k - kv_window, k)  （sink 段 + 最近窗口）

    kv_window=None（绝对模式，只用于等价性测试）⇒ [0, k) 永不驱逐。
    注意 sink_size>0 时这个列表 **不连续**，`data.window_offsets` 的连续切片就不够用了，
    必须走下面的 `window_offsets_for_frame`。
    """
    if frame_index < 0:
        raise ValueError(f"frame_index 应 ≥ 0，得到 {frame_index}")
    if kv_window is None:
        return list(range(frame_index))
    if kv_window < 1 or sink_size < 0:
        raise ValueError(f"kv_window 应 ≥ 1、sink_size 应 ≥ 0，得到 {kv_window} / {sink_size}")
    cap = sink_size + kv_window
    if frame_index <= cap:
        return list(range(frame_index))
    return list(range(sink_size)) + list(range(frame_index - kv_window, frame_index))


def cache_len(frame_index: int, sink_size: int = DEFAULT_SINK,
              kv_window: int | None = DEFAULT_KV_WINDOW) -> int:
    """前向 cell k 之前缓存里的 cell 数 = min(k, sink + kv_window)。`forward_frame` 断言的就是它。"""
    return len(retained_cells(frame_index, sink_size, kv_window))


SINK_ANCHORS = ("absolute", "window")


def window_offsets_for_frame(offset_px: torch.Tensor, frame_index: int,
                             sink_size: int = DEFAULT_SINK,
                             kv_window: int | None = DEFAULT_KV_WINDOW,
                             sink_anchor: str = "absolute") -> torch.Tensor:
    """`forward_frame(offset_win=...)` 要的 [b, n_cached+1, 2]：缓存里那几个 cell + 当前 cell，
    整体减去锚点的世界坐标（于是 `[:, 0] == 0`，满足 arope_dit 的锚定断言）。

    sink_anchor（只在 sink_size>0 且缓存已饱和时有区别；sink_size=0 两者逐位相同）：
      absolute —— 锚在**列表里最老那一格** = cell 0。sink 段留在真实世界坐标，代价是当前 cell 的 h/w RoPE
                  位随行走无界增长（k=40 时约 1.8× 训练包络），审计 GEO-1 量过。
      window   —— 锚在**滑动窗最老的 cell**（k − kv_window），与 sink_size=0 的口径相同，RoPE 不出包络；
                  sink 段的偏移钉成 0（= 锚点位置）：它们的内容照样可见（外观锚点），只是空间对齐信息
                  被放弃 —— 对"全局参照"来说本来也不需要。
    sink_size=0 时窗口连续，退化成 `data.window_offsets(off, a, k)`（本函数会自己核对这一点）。
    """
    if sink_anchor not in SINK_ANCHORS:
        raise ValueError(f"sink_anchor 只能是 {SINK_ANCHORS}，得到 {sink_anchor!r}")
    if offset_px.dim() != 3 or offset_px.shape[-1] != 2:
        raise ValueError(f"offset_px 形状应为 [b, F, 2]，得到 {tuple(offset_px.shape)}")
    idx = retained_cells(frame_index, sink_size, kv_window) + [frame_index]
    f = offset_px.shape[1]
    if idx[-1] >= f:
        raise ValueError(f"offset 只有 {f} 个 cell，要不到 cell {idx[-1]}（rollout 前先 extend_offsets）")
    sel = torch.as_tensor(idx, device=offset_px.device, dtype=torch.long)
    out = offset_px.index_select(1, sel)
    saturated = kv_window is not None and frame_index > sink_size + kv_window
    if sink_anchor == "window" and sink_size > 0 and saturated:
        # 保留集 = [0, sink) ∪ [k−kv_window, k)：锚在 idx[sink_size] = k − kv_window，sink 段钉 0
        out = out - out[:, sink_size:sink_size + 1]
        out = torch.cat([torch.zeros_like(out[:, :sink_size]), out[:, sink_size:]], dim=1)
    else:
        out = out - out[:, :1]
    if sink_size == 0:
        # 连续窗口：与 data.window_offsets 必须逐位相同 —— 两处口径钉死在一起
        ref = window_offsets(offset_px, idx[0], frame_index)
        assert torch.equal(out, ref), (
            f"锚定窗口与 data.window_offsets(off, {idx[0]}, {frame_index}) 不一致 —— 下标口径分叉了")
    return out


# --------------------------------------------------------------------------
# tiny 模型
# --------------------------------------------------------------------------

def tiny_dit(seed: int = 1234, device: str = "cpu", dtype=torch.float32, **over) -> WanModel:
    """tiny WanModel + ARoPE。mask 通道随机化（零初始化 = 没接上，那就测不到它）。"""
    cfg = dict(TINY)
    cfg.update(over)
    torch.manual_seed(seed)
    dit = WanModel(**cfg).to(device=device, dtype=dtype).eval().requires_grad_(False)
    install_arope(dit, mask_channel=True)
    with torch.no_grad():
        dit.arope_mask_embedding.weight.normal_(0.0, 0.02)
        dit.arope_mask_embedding.bias.normal_(0.0, 0.02)
    dit.freqs = tuple(t.to(device) for t in dit.freqs)
    return dit


def tiny_arope_dit(causal: bool = True, seed: int = 1234, device: str = "cpu",
                   dtype=torch.float32, **over) -> AropeDiT:
    return AropeDiT(dit=tiny_dit(seed, device, dtype, **over), mask_channel=True, causal=causal)


def _tiny_cached(device: str, dtype) -> AropeDiT:
    """同一进程里反复调 preflight 时不重复建模型（stage3 只调一次，测试里会调很多次）。"""
    key = (device, str(dtype))
    if key not in _TINY_CACHE:
        _TINY_CACHE[key] = tiny_arope_dit(causal=True, device=device, dtype=dtype)
    return _TINY_CACHE[key]


# --------------------------------------------------------------------------
# 自检本体
# --------------------------------------------------------------------------

def _const_velocity_offsets(n_cells: int, device, dtype=torch.float32) -> torch.Tensor:
    """匀速直行的 [1, n_cells, 2] 世界坐标：实测步速 × 0.25 s/cell（cell 0 起步只有 2.5/16）。

    只用来喂形状与数值范围合理的坐标——自检查的是算术，不是画面。
    """
    vx, vy = WALK_SPEED_PX_S
    per = torch.tensor([vx, vy], dtype=torch.float64) / 4.0  # 4 帧 / 16 fps = 0.25 s
    off = torch.zeros(1, n_cells, 2, dtype=torch.float64)
    if n_cells > 1:
        off[0, 1] = per * (2.5 / 4.0)                        # cell 0 只有帧 0，起步项减半
        for k in range(2, n_cells):
            off[0, k] = off[0, k - 1] + per
    return off.to(device=device, dtype=dtype)


def _rollout(m: AropeDiT, n_frames: int, sink_size: int, kv_window, rope_cap: int,
             off: torch.Tensor, dtype, device) -> tuple[torch.Tensor, dict, list[int]]:
    """提交式 rollout：每个 cell 一次 commit=True 前向，返回 (拼好的输出, 缓存, 每步的 n_cached)。"""
    bare = unwrap(m)
    hw = (TINY_LAT_H // 2) * (TINY_LAT_W // 2)
    cache = bare.init_kv_cache(kv_window=kv_window, sink_size=sink_size, rope_cap=rope_cap)
    g = torch.Generator().manual_seed(0)
    outs, seen = [], []
    for k in range(n_frames):
        k0 = cache["k"][0]
        n_cached = 0 if k0 is None else k0.shape[1] // hw
        want = cache_len(k, sink_size, kv_window)
        assert n_cached == want, f"cell {k}: 缓存里 {n_cached} 个 cell，按 sink/窗口算术应为 {want}"
        seen.append(n_cached)
        ow = window_offsets_for_frame(off, k, sink_size, kv_window)
        assert ow.shape == (1, n_cached + 1, 2), \
            f"cell {k}: offset_win {tuple(ow.shape)} != {(1, n_cached + 1, 2)}"
        mk = known_mask(ow[:, -1:] / PX_PER_LATENT, TINY_LAT_H, TINY_LAT_W).to(dtype).unsqueeze(1)
        x_k = torch.randn(1, TINY["in_dim"], 1, TINY_LAT_H, TINY_LAT_W, generator=g).to(device, dtype)
        ctx = torch.randn(1, 1, TINY_L_TXT, TINY["text_dim"], generator=g).to(device, dtype)
        ids = torch.zeros(1, 1, dtype=torch.int64, device=device)
        with torch.no_grad():
            # 走 forward(kv_cache=...) 这个入口而不是直接调 forward_frame：FSDP 的 all-gather
            # 挂在包装模块的 forward 上，stage3 也必须这么调（arope_dit.py:440）
            o = m(x_k, torch.zeros(1, device=device, dtype=dtype), ctx, context_ids=ids,
                  first_frame_cond=False, frame_index=k, kv_cache=cache, commit=True,
                  offset_win=ow, mask_in_k=mk)
        assert torch.isfinite(o).all(), f"cell {k} 输出出现 NaN/Inf"
        outs.append(o)
    return torch.cat(outs, dim=2), cache, seen


def preflight_sliding_kv(sink_size: int = DEFAULT_SINK, kv_window: int | None = DEFAULT_KV_WINDOW,
                         rope_cap: int = DEFAULT_ROPE_CAP, *, n_extra: int = 2,
                         check_eviction_effect: bool = True, device: str = "cpu",
                         dtype=torch.float32, log=print) -> dict:
    """在 tiny CPU 模型上把 sink / 驱逐 / 锚定坐标的算术跑完整一遍。返回一份测量 dict。

    跑 `sink_size + kv_window + n_extra` 个 cell —— 刚好越过第一次驱逐，于是**同时**覆盖
    上游那个 sink 切片 bug（rollout 早期 n_cached < sink）和驱逐之后的稳态。

    失败一律抬成 RuntimeError 并附上修法：这是启动闸，不是单元测试，调用方不该继续往下走。
    """
    t0 = time.time()
    if kv_window is None or kv_window <= 0:
        log("[preflight] kv_window 未启用（绝对模式），滑窗算术无需自检")
        return {"checked": False}

    # ---- 0. 纯算术闸（不建模型就能判死的）----
    if sink_size < 0:
        raise RuntimeError(f"[preflight] sink_size 必须 ≥ 0，得到 {sink_size}")
    if rope_cap != sink_size + kv_window:
        raise RuntimeError(
            f"[preflight] rope_cap 必须 == sink_size + kv_window（{rope_cap} != {sink_size} + {kv_window}）。"
            "时间位 tpos 恒等于窗口里已缓存的 cell 数，cap 对不上意味着 rollout 会用到训练时"
            "从没出现过的相对时间距离。")
    if sink_size > 0:
        log(f"[preflight] 警告：sink_size={sink_size} > 0。ARoPE 的 h/w RoPE 位是一直在漂的世界坐标，"
            f"被永久钉住的 sink cell 在长 rollout 之后会落在训练分布之外"
            f"（实测 21 cell 内最大位移 373 px = 11.66 token）。stage3 正式训练与部署用的是 sink 3 / window 17 / cap 20。")

    n_frames = sink_size + kv_window + n_extra
    m = _tiny_cached(device, dtype)
    f_rows = m.dit.freqs[0].shape[0]
    if rope_cap + 1 > f_rows:
        raise RuntimeError(f"[preflight] rope_cap={rope_cap} 超出时间轴 RoPE 表的 {f_rows} 行")
    off = _const_velocity_offsets(n_frames, device, dtype)

    try:
        o_slide, cache, seen = _rollout(m, n_frames, sink_size, kv_window, rope_cap, off, dtype, device)
    except (AssertionError, RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"[preflight] 滑窗 KV 路径在 sink_size={sink_size} / kv_window={kv_window} / "
            f"rope_cap={rope_cap} 上失败: {type(e).__name__}: {e}\n"
            f"  首先检查 distill/arope_dit.py::forward_frame 的历史 RoPE 分段：\n"
            f"      n_sink = min(kv_cache['sink_size'], n_cached)   # ← 上游 GraphGWM 抄漏的就是这一行\n"
            f"      R      = n_cached - n_sink\n"
            f"      if n_sink > 0: build_world_freqs(..., n_sink, ..., f_start=0)\n"
            f"      if R > 0:      build_world_freqs(..., R,      ..., f_start=tpos - R)\n"
            f"  以及 offset_win 的切片必须与 preflight.retained_cells({n_frames - 1}, "
            f"{sink_size}, {kv_window}) 逐格对齐、且减掉列表里最老那一格。") from e

    hw = (TINY_LAT_H // 2) * (TINY_LAT_W // 2)
    cap = sink_size + kv_window
    final = cache["k"][0].shape[1] // hw
    if final > cap:
        raise RuntimeError(f"[preflight] 驱逐没生效：rollout 完缓存里有 {final} 个 cell，上限是 {cap}")
    n_evict = max(0, n_frames - cap)
    if n_evict == 0:
        raise RuntimeError(
            f"[preflight] 自检没覆盖到驱逐（{n_frames} 帧 ≤ 窗口 {cap}）——把 n_extra 调大")
    if max(seen) != cap:
        raise RuntimeError(f"[preflight] n_cached 没有饱和到 {cap}（实测最大 {max(seen)}）")

    # ---- 驱逐必须真的改变结果：形状对内容错是这条路径最典型的静默失败 ----
    d_evict = None
    if check_eviction_effect:
        big = n_frames                       # kv_window 大到不会驱逐，作为对照
        o_full, _, _ = _rollout(m, n_frames, sink_size, big, sink_size + big, off, dtype, device)
        d_evict = float((o_full[:, :, -1] - o_slide[:, :, -1]).abs().max())
        if not d_evict > 0:
            raise RuntimeError(
                "[preflight] 驱逐对输出没有任何影响 —— 窗口没在滑，或者最后一帧根本没读到历史。"
                "这正是'形状对、内容错、不报错'的那一类失败。")

    ms = (time.time() - t0) * 1e3
    out = {"checked": True, "sink_size": sink_size, "kv_window": kv_window, "rope_cap": rope_cap,
           "n_frames": n_frames, "n_cached_max": max(seen), "n_evictions": n_evict,
           "cache_cells_final": final, "evict_effect": d_evict, "ms": ms,
           "retained_last": retained_cells(n_frames - 1, sink_size, kv_window)}
    log(f"[preflight] 滑窗 KV OK: sink={sink_size} window={kv_window} cap={rope_cap}；"
        f"{n_frames} cell、n_cached 饱和在 {max(seen)}、驱逐 {n_evict} 次、缓存收在 {final} cell；"
        f"驱逐对末帧的影响 {d_evict:.3e}；{ms:.0f} ms")
    return out


if __name__ == "__main__":
    preflight_sliding_kv(DEFAULT_SINK, DEFAULT_KV_WINDOW, DEFAULT_ROPE_CAP)
    preflight_sliding_kv(1, 4, 5)          # sink>0 的算术分支（非部署配置）
