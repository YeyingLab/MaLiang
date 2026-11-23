#!/usr/bin/env python3
"""蒸馏用的 DiT 包装：`AropeDiT` = DiffSynth `WanModel` + ARoPE（世界坐标 RoPE / known-mask 通道 /
逐 cell 文本 cross-attn），三种前向模式共用一套权重。

为什么要包一层 nn.Module
-----------------------
`actionrope.arope.arope_forward` 是自由函数，直接伸手拿 `dit.patch_embedding` / `dit.time_embedding`
/ `dit.blocks[i].norm1` / `dit.head` 的参数。stage3 用 FSDP1 + TRANSFORMER_BASED_WRAP(DiTBlock)：
block 的参数只在 **该 block 自己的 forward() 里** 才被 all-gather，在外面伸手拿到的是分片 ——
不报错，结果是垃圾（PORT_设计 risk 2，全项目第一号静默腐化风险）。所以：

1. 整个前向搬进 `AropeDiT.forward`（根模块的 forward，根上的 patch/time/head 参数在这里被 gather）；
2. block 一律用 `block(...)` 调用以触发 FSDP 的 pre-forward hook —— 为此在 import 时把
   `diffsynth.models.wan_video_dit.DiTBlock.forward` 打成 `arope_block_forward`
   （= `actionrope/arope.py:277-318` 的 `_block_forward` 逐行照抄，只把 `self_attn(input_x, freqs)`
   换成带 attn_mask / KV 缓存的 `arope_self_attn`；`arm` 钩子去掉，ARoPE 没有臂）。

三种前向模式
-----------
1. **并行双向**（`causal=False, clean_x=None, kv_cache=None`）—— 与 `arope_forward` **逐位相同**，
   这是整个移植的回归锚点（`distill/tests/test_bitparity.py`）。
2. **并行块因果**（`causal=True`）：21 个 cell 之间因果、cell 内空间双向；`clean_x` 非 None 时再加
   teacher-forcing 双流 `[2S,2S]` 掩码（clean 流 + noisy 流；noisy cell f 只看自己 + **严格之前** 的
   clean cell，绝不看 clean cell f —— 那就是答案，泄漏 = loss 掉到 0 而 rollout 是坏的）。
3. **流式单 cell**（`kv_cache` 非 None）：滑窗 KV 缓存。缓存 **RoPE 之前的 K**，读的时候按
   **窗口锚定** 的坐标重新 RoPE（时间位 j-a、世界坐标 offset[j]-offset[a]），驱逐超出 kv_window 的最老 cell。

三条 BLOCKER（PORT_设计 对抗验证，违反任何一条都是"能跑、loss 会降、模型是错的"）
-----------------------------------------------------------------------------
* **B1 first_frame_cond 绝不能进单 cell 路径。** `actionrope/arope.py:227` 的 `t_tok[:, 0] = 0` 抹的是
  "这个张量的第 0 帧"；单 cell 前向里 f==1，于是 rollout 的每一个 cell 都被告知 t=0。
  `forward()` 在 kv_cache 非 None 时硬断言 `first_frame_cond is False`，`forward_frame` 无条件按 `t_k` 建时间步。
* **B2 build_world_freqs 需要 f_start。** 上游签名固定查 `f_table[:f]`，单 cell 永远拿到时间位 0。
  这里给出 distill 本地的 `build_world_freqs(..., f_start=0)` 变体（**不改** `actionrope/arope.py`），
  f_start=0 时与上游逐位相同。
* **B3 逐 cell 时间步。** 上游 `_per_token_timestep` 断言 `timestep.numel() == b`，stage1 要传 `[b, 21]`。
  这里的 `_per_token_timestep_pf` 接受 标量 / `[b]` / `[b, F]`，前两种与上游逐位相同。

flow map（stage2-flowmap）：第二个时间步 r
----------------------------------------
`install_flowmap()` 之后模型学的是 **均速度** `u(x_t, t, r)`，`forward(..., r_timestep=...)` /
`forward_frame(..., r_k=...)` 多收一个 r。默认（`residual` + 读出层零初始化）**对任意 (t, r)**
都与装之前逐位相同，所以它同时是 flow map 的初值和回归锚点。没装的时候 r 这条路径完全不存在，
`r_timestep` 只准是 None 或与 `timestep` 相同 —— 上游那个不带 delta 分支的 embedder 会 **收下 r
然后忽略它**（transformer_far_wan_model.py:241），于是接错线会退化成普通扩散模型且不报错。
新增的 5 个键（`delta_embedding.{0,2}.{weight,bias}` + `delta_gate`）挂在 **dit 上**，827 → 832。

另外两条硬规矩
-------------
* **绝不对 KV rollout 做梯度检查点**（risk 6）：滑窗 commit 是破坏性的（cat + 驱逐），checkpoint 重算时
  读到的是 **驱逐之后** 的缓存 —— 缓存没满时报 `CheckpointError`，满了以后形状对、内容错、**不报错**。
  `forward()` 在 kv_cache 非 None 时断言 `use_gradient_checkpointing` 为假；`forward_frame` 里 block 直接调用。
* **注意力只走 SDPA**（risk 8）：import 时断言 flash-attn / sageattention 都没装 —— 装了的话
  `flash_attention()` 会改走别的 kernel，`arope_forward` 与本文件就悄悄分叉，回归锚点失效。
"""
from __future__ import annotations

import copy
import functools
import os
import sys

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models import wan_video_dit as _wvd
from diffsynth.models.wan_video_dit import WanModel, modulate, rope_apply, sinusoidal_embedding_1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# _axis_freqs / _embed_context 直接复用上游实现：逐位等价是硬要求，不能各写一份
from actionrope.arope import (  # noqa: E402
    MASK_EMBEDDING_NAME,
    PX_PER_TOKEN,
    _axis_freqs,
    _embed_context,
    install_arope,
)

MODEL_DIR = os.environ.get("MALIANG_MODEL_DIR", os.path.join(os.path.dirname(ROOT), "models/Wan2.2-TI2V-5B"))
DIT_FILES = [os.path.join(MODEL_DIR, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]
LAT_H, LAT_W = 30, 52

# ---- flow map（stage2-flowmap）：第二个时间步 r 的分支 ----------------------------------
# 两个名字都挂在 **dit 上**（`dit.add_module` / `dit.register_buffer`），不是挂在 AropeDiT 包装上：
# runlib.strip_dit_prefix 只认 `dit.` / `arm.` 前缀，挂在包装上会被当成"包装层自己的 buffer"静默丢掉。
# 这是 actionrope/arope.py:196-206 给 arope_mask_embedding 用的同一条先例。
DELTA_EMBEDDING_NAME = "delta_embedding"
DELTA_GATE_NAME = "delta_gate"
FLOWMAP_FORMS = ("residual", "convex")

# risk 8：本文件的自注意力恒走 F.scaled_dot_product_attention；上游 flash_attention() 在这三个标志
# 全为 False 时也落到 SDPA，两条路径才逐位相同。哪天有人 pip install flash-attn，这个断言先炸，
# 而不是让 test_bitparity 之外的所有下游悄悄测在另一个模型上。
assert not (_wvd.FLASH_ATTN_3_AVAILABLE or _wvd.FLASH_ATTN_2_AVAILABLE or _wvd.SAGE_ATTN_AVAILABLE), (
    "检测到 flash_attn / sageattention：DiffSynth 的 flash_attention() 会改走它们，"
    "而 distill/arope_dit.py 恒走 SDPA，两条路径不再逐位相同。卸载它们，或先修 arope_self_attn。"
)


# --------------------------------------------------------------------------
# 世界坐标 RoPE（distill 本地变体：多一个 f_start）
# --------------------------------------------------------------------------

def rot2d(vec: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """把 (dx, dy)（屏幕 u 向右、v 向下，单位任意）绕原点转 ang 弧度。vec [..., 2]、ang 可广播到 [...]。

    u' = u·cos − v·sin，v' = u·sin + v·cos。v 轴朝下 ⇒ 正角在屏幕上是顺时针，纯约定。
    """
    c, s_ = torch.cos(ang), torch.sin(ang)
    u, v = vec[..., 0], vec[..., 1]
    return torch.stack([u * c - v * s_, u * s_ + v * c], dim=-1)


def world_pos_maps(offset_tok: torch.Tensor, angle, h: int, w: int, scale=None):
    """逐 token 的世界坐标 (pos_h, pos_w)，各 [b, f, h, w]。

    cell j 的 token (i_h, j_w) 在**当前相机系**里的位置 = 绕画幅中心转 angle[b,j] 之后再加该 cell 的偏移：
        (u, v) = R(angle) · (j_w − ctr_w, i_h − ctr_h) + (dx, dy) + (ctr_w, ctr_h)
    angle 全 0 ⇒ 退化成可分离的 pos_h = i + dy / pos_w = j + dx（与旋转前的实现逐位相同）。
    """
    b, f = offset_tok.shape[:2]
    dev, dt = offset_tok.device, torch.float64
    gi = torch.arange(h, device=dev, dtype=dt).view(1, 1, h, 1).expand(b, f, h, w)
    gj = torch.arange(w, device=dev, dtype=dt).view(1, 1, 1, w).expand(b, f, h, w)
    ctr_h, ctr_w = (h - 1) / 2.0, (w - 1) / 2.0
    local = torch.stack([gj - ctr_w, gi - ctr_h], dim=-1)              # [b, f, h, w, 2] = (u, v)
    ang = angle.to(device=dev, dtype=dt).view(b, f, 1, 1)
    rot = rot2d(local, ang)
    if scale is not None:
        # 位置整体等比缩放（判决实验）：位置铺得越开 ⇒ 这一帧名义上覆盖的世界越大。
        # 若模型把"位置场变大"读成"视野变大"，画面内容就该按 1/scale 缩小。
        rot = rot * torch.as_tensor(scale, device=dev, dtype=dt).view(b, f, 1, 1, 1)
    off = offset_tok.to(dt).view(b, f, 1, 1, 2)
    pos_w = rot[..., 0] + off[..., 0] + ctr_w
    pos_h = rot[..., 1] + off[..., 1] + ctr_h
    return pos_h, pos_w


def build_world_freqs(dit: WanModel, f: int, h: int, w: int, offset_tok=None, device=None,
                      f_start: int = 0, angle=None, scale=None) -> torch.Tensor:
    """`actionrope.arope.build_world_freqs` + `f_start`（BLOCKER B2）。

    f_start：这一段 token 的时间轴 **绝对起点**，查 `f_table[f_start:f_start+f]`。滑窗 KV 路径靠它把
    单个 cell 放到时间位 tpos（否则 f==1 永远拿到位置 0，整条历史与当前帧的相对时间全塌成 0）。
    f_start=0 时与上游函数逐位相同（`f_table[0:f]` 就是 `f_table[:f]`）。

    offset_tok: [b, f, 2] = (dx_tok, dy_tok)，token 单位；None ⇒ 普通查表 RoPE，返回 [f*h*w, 1, 64]，
    否则返回 [b, f*h*w, 1, 64]。dtype 跟随 dit.freqs（complex128）。

    angle: [b, f] 弧度，每个 cell 的相机朝向**相对当前 cell**（当前 cell 恒为 0）。None / 全 0 ⇒ 走可分离
    路径（逐位等于旋转支持加入之前的实现）。非零时改走逐 token 的位置图：h/w 两轴被耦合，位置不再可分离。
    RoPE 本身对小数位置是解析的（`polar(1, pos·inv_freq)`），所以旋转出来的小数坐标天然合法；
    模型没见过的是"位移随 token 位置变化"这种场，而不是某个没见过的数值。
    """
    f_table, h_table, w_table = dit.freqs
    if device is None:
        device = offset_tok.device if offset_tok is not None else f_table.device
    d_h, d_w = h_table.shape[1] * 2, w_table.shape[1] * 2  # 42, 42
    assert f_start >= 0 and f_start + f <= f_table.shape[0], (
        f"时间轴 RoPE 表只有 {f_table.shape[0]} 行，要不到 [{f_start}, {f_start + f})"
    )
    ft = f_table[f_start:f_start + f]

    if offset_tok is None:
        # 走与原函数一样的查表，保证等价性；不重新算 h/w 轴
        return torch.cat([
            ft.view(f, 1, 1, -1).expand(f, h, w, -1),
            h_table[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_table[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)

    offset_tok = offset_tok.to(device=device, dtype=torch.float64)
    b = offset_tok.shape[0]
    assert offset_tok.shape == (b, f, 2), f"offset_tok 形状应为 [b, {f}, 2]，得到 {tuple(offset_tok.shape)}"
    dx = offset_tok[..., 0]  # [b, f]
    dy = offset_tok[..., 1]
    has_rot = angle is not None and bool(torch.any(torch.as_tensor(angle) != 0))
    has_scale = scale is not None and bool(torch.any(torch.as_tensor(scale) != 1))
    if has_rot or has_scale:
        ang = (torch.zeros(b, f, device=device, dtype=torch.float64) if angle is None
               else torch.as_tensor(angle, device=device, dtype=torch.float64))
        assert ang.shape == (b, f), f"angle 形状应为 [b={b}, f={f}]，得到 {tuple(ang.shape)}"
        pos_h, pos_w = world_pos_maps(offset_tok, ang, h, w, scale)      # 各 [b, f, h, w]
        fh = _axis_freqs(pos_h, d_h)                                    # [b, f, h, w, 21]
        fw = _axis_freqs(pos_w, d_w)
        freqs = torch.cat([
            ft.to(device).view(1, f, 1, 1, -1).expand(b, f, h, w, -1), fh, fw,
        ], dim=-1)
        return freqs.reshape(b, f * h * w, 1, -1).to(f_table.dtype)

    pos_h = torch.arange(h, device=device, dtype=torch.float64).view(1, 1, h) + dy.view(b, f, 1)  # [b, f, h]
    pos_w = torch.arange(w, device=device, dtype=torch.float64).view(1, 1, w) + dx.view(b, f, 1)  # [b, f, w]
    fh = _axis_freqs(pos_h, d_h)  # [b, f, h, 21]
    fw = _axis_freqs(pos_w, d_w)  # [b, f, w, 21]
    freqs = torch.cat([
        ft.to(device).view(1, f, 1, 1, -1).expand(b, f, h, w, -1),
        fh.view(b, f, h, 1, -1).expand(b, f, h, w, -1),
        fw.view(b, f, 1, w, -1).expand(b, f, h, w, -1),
    ], dim=-1)
    return freqs.reshape(b, f * h * w, 1, -1).to(f_table.dtype)


def _freqs_cat_dim(freqs: torch.Tensor) -> int:
    """freqs 的 token 轴：无 batch 的 [S,1,64] 是 0，逐样本的 [b,S,1,64] 是 1。"""
    assert freqs.dim() in (3, 4), f"freqs 应为 [S,1,64] 或 [b,S,1,64]，得到 {tuple(freqs.shape)}"
    return 0 if freqs.dim() == 3 else 1


# --------------------------------------------------------------------------
# 时间步：标量 / [b] / [b, F]
# --------------------------------------------------------------------------

def _per_token_timestep_pf(latents: torch.Tensor, timestep, n_tok_per_frame: int,
                           first_frame_cond: bool) -> torch.Tensor:
    """逐 token 时间步。标量 / [b] 与 `actionrope.arope._per_token_timestep` 逐位相同；
    多出的 `[b, F]` 分支是 stage1 的逐 cell 加噪要的（BLOCKER B3）。

    first_frame_cond 在展开之后才作用（把 cell 0 的 t 置 0），所以 `[b, F]` 传进来时
    cell 0 那一列会被覆盖成 0 —— 这正是 ARoPE 的 TI2V 契约（cell 0 是干净锚点）。
    返回 [b, f * n_tok_per_frame]，dtype 跟 latents。
    """
    b, f = latents.shape[0], latents.shape[2]
    timestep = torch.as_tensor(timestep, device=latents.device).to(latents.dtype)
    # 上游 _per_token_timestep 无条件先 reshape(-1)，所以 [b,1] / [1,1] 在它那里是合法的"逐样本"写法。
    # 只在形状 **不可能** 是逐 cell（不是 [b,f]）且元素数为 1 或 b 时压平：接受的输入集合是上游的严格超集，
    # 且 f==1 时两种解释本来就同解，不存在歧义。
    if timestep.dim() > 1 and tuple(timestep.shape) != (b, f) and timestep.numel() in (1, b):
        timestep = timestep.reshape(-1)
    if timestep.dim() <= 1:
        timestep = timestep.reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(b)
        assert timestep.numel() == b, f"timestep 应为标量 / [b={b}] / [b={b}, f={f}]，得到 {timestep.numel()} 个"
        t_tok = timestep.view(b, 1, 1).expand(b, f, n_tok_per_frame).clone()
    else:
        assert timestep.shape == (b, f), f"逐 cell timestep 形状应为 [b={b}, f={f}]，得到 {tuple(timestep.shape)}"
        t_tok = timestep.view(b, f, 1).expand(b, f, n_tok_per_frame).clone()
    if first_frame_cond:
        t_tok[:, 0] = 0
    return t_tok.reshape(b, f * n_tok_per_frame)


def _ts_equal(a, b) -> bool:
    """两个时间步参数（标量 / [b] / [b,F] / python float）在数值上是否完全相同。

    只在"模型没装 flow map 却收到了 r"这条断言路径上用（正常训练/推理都不走），所以允许它做
    一次 device→CPU 的同步。同一个对象直接返回 True，不做任何拷贝。
    """
    if a is b:
        return True
    ta = torch.as_tensor(a).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    tb = torch.as_tensor(b).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if ta.numel() == 1 and tb.numel() > 1:
        ta = ta.expand_as(tb)
    elif tb.numel() == 1 and ta.numel() > 1:
        tb = tb.expand_as(ta)
    return ta.shape == tb.shape and bool(torch.equal(ta, tb))


# --------------------------------------------------------------------------
# 注意力掩码
# --------------------------------------------------------------------------

class _KVRef:
    """指回 KV cache 本体的自引用句柄 —— 用来绕开 FSDP 对 forward 输入容器的重建。

    ★ 这是一个必须记住的 FSDP 行为：混合精度下 FSDP 会 cast forward 的输入，
    `_apply_to_tensors` 会**遍历并重建** dict / list / tuple，于是 `forward_frame` 写进去的
    K/V 落在一份**拷贝**上，调用方的 cache 永远是空的 —— 而且不报错（实测：
    `外层 dict is 内层 dict → False`，commit 之后外层 cell 数恒为 0）。

    但 FSDP 只 cast 张量，**非张量值按引用原样保留**（同一实测：放进去的哨兵对象 `is` 相同）。
    所以在 cache 里塞一个指回自己的普通对象，就能在 forward 里把原始 dict 找回来。

    为什么不改成"把 cache 挂成模块属性"：调用必须走包装模块的 forward（参数 all-gather 挂在
    那里，直接调 forward_frame 拿到的是分片、同样不报错），而 34 处调用点与全部测试都按
    `kv_cache=` 这个 kwarg 写的。用自引用句柄两边都不用动。
    """

    __slots__ = ("cache",)

    def __init__(self, cache: dict):
        self.cache = cache


def build_causal_mask(f: int, hw: int, device, local: int = 0, sink: int = 0,
                      drop_anchor: bool = False) -> torch.Tensor:
    """块因果 [S, S] bool（True=可见）：cell 之间因果，cell 内空间双向。S = f*hw。

    local>0 ⇒ 只看最近 local 个 cell（外加 sink 个恒可见的开头 cell），对角线恒保留。
    stage1/2 都是 local=0 —— F=21 下"窗口 20 + sink 1"与全因果等价，而 sink=0 的 W=20 会把
    cell 0（ARoPE 的干净锚点）从 cell 20 的视野里踢掉，是致命的。
    """
    fidx = torch.arange(f * hw, device=device) // hw  # [S] 每个 token 属于哪个 cell
    causal = fidx.unsqueeze(1) >= fidx.unsqueeze(0)
    if drop_anchor:      # 见 build_tf_mask 的同名参数
        causal = causal & ((fidx.unsqueeze(0) != 0) | (fidx.unsqueeze(1) == 0))
    if local and local > 0:
        in_win = fidx.unsqueeze(0) > (fidx.unsqueeze(1) - local)
        if sink and sink > 0:
            in_win = in_win | (fidx.unsqueeze(0) < sink)
        return (causal & in_win) | (fidx.unsqueeze(1) == fidx.unsqueeze(0))
    return causal


def build_tf_mask(f: int, hw: int, device, local: int = 0, sink: int = 0,
                  drop_anchor: bool = False) -> torch.Tensor:
    """teacher-forcing 双流掩码 [2S, 2S] bool。token [0,S) = clean 流，[S,2S) = noisy 流。

    * clean q：在 clean 半流内块因果（看 clean 0..f）。
    * noisy q（cell f）：看自己那一个 noisy cell + **严格之前** 的 clean cell（fk < fq）。
      绝不看 clean cell f（那就是答案），也绝不看别的 noisy cell。
    对角线（eye）恒保留。

    local>0 时同时约束 **两个半流**——上游 `gamemaster_dit.py:546 vs :549-554` 只约束了 noisy_ctx，
    clean 半流仍是全块因果，于是窗口化训练会从 clean 半流泄漏无界历史。这里顺手修掉（虽然我们不用窗口）。
    """
    S = f * hw
    tok = torch.arange(2 * S, device=device)
    cln = tok < S
    fr = torch.where(cln, tok // hw, (tok - S) // hw)  # 两个半流各自的 cell 序号 0..f-1
    cq, ck = cln[:, None], cln[None, :]
    fq, fk = fr[:, None], fr[None, :]
    clean_mask = cq & ck & (fk <= fq)
    noise_self = (~cq) & (~ck) & (fq == fk)
    noise_ctx = (~cq) & ck & (fk < fq)
    # drop_anchor：把 clean cell 0（TI2V 锚点）从**每一个** noisy cell 的视野里摘掉，其余历史照旧。
    # 这是「只 drop 场景参考、保留历史」那条 null —— 与 local/sink 拼不出来：local=F−1 sink=0 只能
    # 把 cell 0 挤出**最后一格**的窗口，cell 19 的 20 格窗口里它还在。
    # Δv_anchor = v(全) − v(drop_anchor) 才是"锚点相对于历史的话语权"，正是外观漂移要压的那一项。
    if drop_anchor:
        noise_ctx = noise_ctx & (fk != 0)
    if local and local > 0:
        in_win = fk > (fq - local)
        if sink and sink > 0:
            in_win = in_win | (fk < sink)
        noise_ctx = noise_ctx & in_win
        clean_mask = clean_mask & in_win
    eye = tok[:, None] == tok[None, :]
    return eye | clean_mask | noise_self | noise_ctx


# --------------------------------------------------------------------------
# 自注意力：并行（可带 mask） / 流式（滑窗 KV 缓存）
# --------------------------------------------------------------------------

def _heads(x: torch.Tensor, n: int) -> torch.Tensor:
    return rearrange(x, "b s (n d) -> b n s d", n=n)


def _unheads(x: torch.Tensor, n: int) -> torch.Tensor:
    return rearrange(x, "b n s d -> b s (n d)", n=n)


def _evict(t: torch.Tensor, kv: dict) -> torch.Tensor:
    """滑窗驱逐：缓存超过 sink + kv_window 个 cell 就丢掉最老的 **非 sink** cell。t: [b, n*hw, dim]。"""
    kw = kv["kv_window"]
    if kw is None:
        return t
    hw, sink = kv["hw"], kv["sink_size"]
    while t.shape[1] // hw > sink + kw:
        t = torch.cat([t[:, :sink * hw], t[:, (sink + 1) * hw:]], dim=1)
    return t


def arope_self_attn(sa, x: torch.Tensor, freqs: torch.Tensor, attn_mask=None, kv=None) -> torch.Tensor:
    """`SelfAttention.forward` 的替身。attn_mask=None 且 kv=None 时与上游 **逐位相同**
    （上游 `flash_attention()` 在没装 flash/sage 时同样落到 SDPA，见文件头的 import 断言）。

    kv 非 None ⇒ 流式：缓存 **RoPE 之前** 的 K，读的时候按窗口锚定坐标重算
    （`kv["freqs_hist"]` 给历史、`kv["freqs_cur"]` 给当前 cell）。RoPE 是相对编码，所以重锚定对
    注意力分数是**精确**的；但缓存的 K/V 是从带掩码通道的 x 算出来的，掩码按各 cell **提交时**的
    锚点固化，k>20 之后的历史与一段锚在 k−20 的训练片段并不逐位相同（stage2→stage3 输入移位，
    自强迫训练穿过，GEO-2）。
    """
    n = sa.num_heads
    if kv is None:
        q = rope_apply(sa.norm_q(sa.q(x)), freqs, n)
        k = rope_apply(sa.norm_k(sa.k(x)), freqs, n)
        v = sa.v(x)
        o = F.scaled_dot_product_attention(_heads(q, n), _heads(k, n), _heads(v, n), attn_mask=attn_mask)
        return sa.o(_unheads(o, n))

    # ---- 流式单 cell ----
    li = kv["layer"]
    k_raw = sa.norm_k(sa.k(x))          # 进缓存的是这个（RoPE 之前）
    v = sa.v(x)
    q = rope_apply(sa.norm_q(sa.q(x)), kv["freqs_cur"], n)
    k_hist, v_hist = kv["k"][li], kv["v"][li]
    if k_hist is None:
        k_cat, v_cat = rope_apply(k_raw, kv["freqs_cur"], n), v
    else:
        fh = kv["freqs_hist"]
        assert fh is not None and fh.shape[_freqs_cat_dim(fh)] == k_hist.shape[1], (
            f"历史 freqs 与缓存长度不符：freqs {tuple(fh.shape)} vs k_hist {tuple(k_hist.shape)}"
        )
        k_cat = torch.cat([rope_apply(k_hist, fh, n), rope_apply(k_raw, kv["freqs_cur"], n)], dim=1)
        v_cat = torch.cat([v_hist, v], dim=1)
    o = F.scaled_dot_product_attention(_heads(q, n), _heads(k_cat, n), _heads(v_cat, n), attn_mask=None)
    if kv["commit"]:
        nk = k_raw if k_hist is None else torch.cat([k_hist, k_raw], dim=1)
        nv = v if v_hist is None else torch.cat([v_hist, v], dim=1)
        # detach：缓存是跨帧的常量，梯度绝不能穿过它（stage3 的 streamed replay 正是靠这一点才是精确的）
        kv["k"][li] = _evict(nk, kv).detach()
        kv["v"][li] = _evict(nv, kv).detach()
    return sa.o(_unheads(o, n))


# --------------------------------------------------------------------------
# DiTBlock.forward 的猴补丁（FSDP all-gather 必须经过 block(...) 这个入口）
# --------------------------------------------------------------------------

def arope_block_forward(self, x, context, t_mod, freqs, *, f: int | None = None,
                        attn_mask=None, kv=None):
    """= `actionrope/arope.py:277-318` 的 `_block_forward`（arm 去掉），唯一改动：
    `self.self_attn(input_x, freqs)` → `arope_self_attn(self.self_attn, input_x, freqs, attn_mask, kv)`。

    f：逐 cell 文本 cross-attn 的 cell 数（TF 双流时是 2*Fl）。context 是整条 [b,L,dim] 时用不到，
    默认 None —— 这样 DiffSynth 自己的 `model_fn_wan_video`（`block(x, context, t_mod, freqs)`，
    四个位置参数、整条 context）在打完补丁之后照样能跑，且逐位不变。
    """
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
    input_x = modulate(self.norm1(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, arope_self_attn(self.self_attn, input_x, freqs, attn_mask, kv))

    q_in = self.norm3(x)
    if context.shape[0] == x.shape[0]:
        x = x + self.cross_attn(q_in, context)
    else:
        b = x.shape[0]
        assert f is not None, "逐 cell context 需要显式传 f（block 的 cell 数）"
        assert context.shape[0] == b * f, f"逐 cell context 应有 b*f={b * f} 行，得到 {context.shape[0]}"
        q_in = rearrange(q_in, "b (f n) c -> (b f) n c", f=f)
        y = self.cross_attn(q_in, context)
        x = x + rearrange(y, "(b f) n c -> b (f n) c", f=f)

    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x


# 全局、一次性：所有 DiTBlock 实例（含 DiffSynth 自己 pipeline 里的）都换成上面这个 forward。
_wvd.DiTBlock.forward = arope_block_forward


# --------------------------------------------------------------------------
# 模型加载
# --------------------------------------------------------------------------

def load_dit(model_dir: str = MODEL_DIR, device: str = "cpu") -> WanModel:
    """只加载 DiT（不挂 T5 / VAE），bf16，先落 CPU —— 与 `actionrope/train.py:163-174` 一致。"""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    files = [os.path.join(model_dir, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=files)],
        tokenizer_config=None, redirect_common_files=False,
    )
    return pipe.dit


def unwrap(model: nn.Module) -> "AropeDiT":
    """拿到裸的 AropeDiT（DeepSpeed / FSDP / DDP 包装之后 `init_kv_cache`、`dit.freqs` 都要走裸模块）。"""
    while hasattr(model, "module"):
        model = model.module
    return model


# --------------------------------------------------------------------------
# AropeDiT
# --------------------------------------------------------------------------

class AropeDiT(nn.Module):
    """WanModel + ARoPE 的包装。三种前向模式见文件头。

    causal：False = 双向教师 / critic；True = 因果学生（块因果掩码 + 允许 TF 双流 + 允许流式 KV）。
    train_local_attn_size / train_sink_size：窗口化训练（stage1/2 都是 0，即全因果）。
    """

    def __init__(self, model_dir: str = MODEL_DIR, mask_channel: bool = True, causal: bool = False,
                 train_local_attn_size: int = 0, train_sink_size: int = 0, dit: WanModel | None = None,
                 flowmap_form: str | None = None):
        super().__init__()
        if dit is None:
            dit = load_dit(model_dir, device="cpu")
        # mask 通道零初始化：装上的瞬间输出与原模型完全相同
        self.dit = install_arope(dit, mask_channel=mask_channel)
        self.mask_channel = bool(mask_channel)
        self.causal = bool(causal)
        self.train_local_attn_size = int(train_local_attn_size)
        self.train_sink_size = int(train_sink_size)
        # 训练期把 cell 0（TI2V 锚点）从 noisy 的视野里摘掉，其余历史保留 —— Δv_anchor 的 null
        self.train_drop_anchor = False
        # flow map（stage2-flowmap）的第二时间步分支：None = 没装（默认，与装之前逐位相同）。
        # 装了以后是 "residual" / "convex"。__init__ **绝不自己装**，只有 install_flowmap() 会。
        # 传进来的 dit 已经带着 delta 分支（deepcopy / 再包一层 / 从 832 键装载过）时必须显式说明是
        # 哪一种形式：两种形式的融合公式不同，猜错了就是"能跑、数值是另一个模型"。
        assert flowmap_form is None or flowmap_form in FLOWMAP_FORMS, (
            f"flowmap_form 只能是 {FLOWMAP_FORMS} 或 None，得到 {flowmap_form!r}")
        if hasattr(self.dit, DELTA_EMBEDDING_NAME):
            assert flowmap_form is not None, (
                "传进来的 dit 上已经有 delta_embedding 了，请显式给 AropeDiT(..., flowmap_form=...)："
                "residual 与 convex 的融合公式不同，猜错了不会报错，只会安静地跑成另一个模型。")
        self.flowmap_form: str | None = flowmap_form
        # 模块级默认；每次 forward 还可以单独传。KV rollout 上两者都必须是假（risk 6）。
        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        # [16380,16380] 的 bool 掩码 268 MB，按 (f, hw, tf, 窗口, device) 缓存，别每步重建
        self._mask_cache: dict = {}

    # ---- 小工具 ----

    def move_rope_table(self, device) -> "AropeDiT":
        """`dit.freqs` 是普通 tuple 属性（不是 buffer），`.to(device)` / prepare 都搬不动它。
        DeepSpeed / FSDP prepare 之后必须在 **裸模块** 上调一次。"""
        self.dit.freqs = tuple(t.to(device) for t in self.dit.freqs)
        return self

    def _get_mask(self, f: int, hw: int, device, tf: bool):
        if not self.causal:
            return None
        key = (f, hw, bool(tf), self.train_local_attn_size, self.train_sink_size,
               bool(self.train_drop_anchor), str(device))
        m = self._mask_cache.get(key)
        if m is None:
            build = build_tf_mask if tf else build_causal_mask
            m = build(f, hw, device, self.train_local_attn_size, self.train_sink_size,
                      drop_anchor=bool(self.train_drop_anchor))
            self._mask_cache[key] = m
        return m

    def _mask_embed(self, mask_input, ref: torch.Tensor):
        """mask 通道的 embedding。ref 是 patch_embedding 的 **输出**（不是权重）——
        autocast 下输出是 bf16 而权重可能是 fp32，拿错参照物就与 arope.py:372 不再逐位相同。"""
        if mask_input is None:
            return None
        if not hasattr(self.dit, MASK_EMBEDDING_NAME):
            raise ValueError("给了 mask_input 但 dit 没有 arope_mask_embedding，请先 install_arope(dit, mask_channel=True)")
        return getattr(self.dit, MASK_EMBEDDING_NAME)(mask_input.to(device=ref.device, dtype=ref.dtype))

    # ---- flow map：第二个时间步 r ----

    @property
    def has_flowmap(self) -> bool:
        """装了 delta 分支没有。**以 dit 上的模块为准**，不看包装上的 flowmap_form ——
        续训是"先 install 再 strict load"，权重里有没有这五个键才是唯一事实。"""
        return hasattr(self.dit, DELTA_EMBEDDING_NAME)

    def install_flowmap(self, form: str = "residual", gate: float = 1.0) -> "AropeDiT":
        """装上第二时间步 r 的 delta 分支。**绝不从 __init__ 调**（见下面的初始化顺序）。

        form="residual"（默认，本项目用的）
            t = E(t) + g · D(r)，`D = deepcopy(E)` 但把 **第二个 Linear** 的 weight/bias 清零。
            于是 D(·) ≡ 0，对 **任意** (t, r) 都与装之前逐位相同 —— 这是硬要求：我们 50 % 的样本
            按构造就在 r ≠ t 的off-diagonal 上，而初值是 24 GPU-hours 换来的 stage1b 教师。
            只清 **读出层**、保留第一层学好的 256→3072：分支一上来就有 r 的有意义隐表示，
            ∂L/∂delta_embedding[2] 在第 0 步就非零，立刻开始学（arope_mask_embedding 零初始化 Conv3d 的同一套路）。
        form="convex"（上游 AnyFlow，transformer_far_wan_model.py:330）
            t = (1−g)·E(t) + g·D(r)，D 是 E 的原样 deepcopy。**只在 r == t 那条对角线上**与教师相同；
            off-diagonal 是两个不同时间嵌入的 0.75/0.25 混合（CPU 实测 max|Δ| ≈ 1.1e-01）。
            只留作"上游保真度"消融，别指望它是恒等的。
            ⚠ 千万别在 convex 里"顺手零初始化"：那给出 t = 0.75·E(t)，把教师的时间嵌入缩水 25 %，
            比两种做法都糟，却长得像一个安全的零初始化。

        gate：`dit.delta_gate`，**persistent=True 的 buffer**，跟着权重走。上游是 persistent=False
        （从 config 重建），于是训练与推理可以悄悄用不同的门而不报错。residual 形式下门乘的是一个
        本来就是 0 的读出，第 0 步冗余，之后只等价于给 delta_embedding[2] 换一个有效 lr。

        导出格式（runlib）：两个名字都挂在 dit 上 ⇒ state_dict 里是 `dit.delta_embedding.{0,2}.{weight,bias}`
        和 `dit.delta_gate`，strip_dit_prefix 剥掉 `dit.` 之后正好是 5 个新键，827 → 832
        （runlib.AROPE_DIT_KEYS_FLOWMAP / runlib.FLOWMAP_KEYS）。挂在 AropeDiT 包装上会被静默丢掉。

        初始化顺序（写错了不会报错，会静默丢掉教师）：
            新 run   : AropeDiT(...) → load_dit_ckpt(m, stage1b)【strict 827】→ install_flowmap()
            续训     : AropeDiT(...) → install_flowmap() → load_dit_ckpt(m, dit_stepN)【strict 832】
        反过来（先 install 再从 827 的 ckpt 装）会 deepcopy 到 **随机初始化** 的 time_embedding，
        之后 load 又把 time_embedding 覆盖成教师的 —— delta 分支第一层从此与 time_embedding 不一致。
        residual 形式下输出仍然是恒等的（读出是 0），但分支的隐表示是垃圾。所以装完立刻自检
        （见下面的两个 assert），训练脚本再另外跑一次真前向的 max|Δ| == 0.0 闸门。
        """
        assert form in FLOWMAP_FORMS, f"flowmap_form 只能是 {FLOWMAP_FORMS}，得到 {form!r}"
        # ★ convex 形式下门必须落在开区间 (0, 1)。上游 gate_value = 0.25。
        #   g == 1.0（本方法的 residual 默认值）会让 t = 0·E(t) + 1·D(r) —— 时间嵌入被 r 嵌入**整个替换**，
        #   模型对 t 再无任何依赖（实测 |u(t=900,r=100) − u(t=100,r=100)| = 0.000e+00）。
        #   而 r == t 时 D(t) == E(t)（D 是 E 的原样 deepcopy），所以 trainer 那道
        #   _assert_teacher_identity（convex 只查对角线）会**给出绿灯**。静默、灾难性，
        #   只要有人写了 `--flowmap_form convex` 而忘了跟上 `--delta_gate 0.25` 就会中招。
        #   g == 0.0 则是另一个极端：delta 分支恒被乘没，r 完全不起作用，退化成普通扩散模型。
        if form == "convex":
            assert 0.0 < float(gate) < 1.0, (
                f"convex 形式的 delta_gate 必须落在开区间 (0, 1)，得到 {gate}（上游 gate_value = 0.25）。\n"
                f"  t = (1−g)·E(t) + g·D(r)：g=1 把时间嵌入整个换成 r 嵌入（模型对 t 零依赖），"
                f"g=0 把 r 分支整个乘没（退化成普通扩散）。两种都能骗过只查 r==t 对角线的同一性闸。\n"
                f"  想用上游保真度消融请写 --flowmap_form convex --delta_gate 0.25。"
            )
        assert not self.has_flowmap, (
            "已经装过 flow map 了。重复 install 会用当前的 time_embedding 重新 deepcopy 一份，"
            "把已经训练过的 delta 分支整个丢掉（residual 形式下还会把它清零）—— 静默、无报错。"
        )
        dit = self.dit
        emb = dit.time_embedding
        assert isinstance(emb, nn.Sequential) and len(emb) == 3, (
            f"time_embedding 应是 Sequential(Linear, SiLU, Linear)，得到 {emb}"
        )
        delta = copy.deepcopy(emb)
        if form == "residual":
            nn.init.zeros_(delta[2].weight)
            nn.init.zeros_(delta[2].bias)
        dit.add_module(DELTA_EMBEDDING_NAME, delta)

        ref = emb[0].weight
        g = torch.tensor([float(gate)], dtype=ref.dtype, device=ref.device)
        assert float(g.item()) == float(gate), (
            f"gate={gate} 在 {ref.dtype} 下不可精确表示（存回来是 {float(g.item())}）："
            "门要跟着 bf16 权重一起存盘，不精确就意味着训练用的门和装载回来的门不是同一个数"
        )
        dit.register_buffer(DELTA_GATE_NAME, g, persistent=True)
        self.flowmap_form = form

        if form == "residual":
            assert torch.equal(delta[0].weight, emb[0].weight) and torch.equal(delta[0].bias, emb[0].bias), (
                "delta_embedding 第一层应当是 time_embedding 的逐位副本"
            )
            assert float(delta[2].weight.abs().max()) == 0.0 and float(delta[2].bias.abs().max()) == 0.0, (
                "residual 形式的 delta_embedding 读出层必须是全 0"
            )
        return self

    def _embed_time(self, t_tok: torch.Tensor, r_tok: torch.Tensor | None) -> torch.Tensor:
        """[b, S] 的逐 token 时间步 → [b, S, dim] 的时间嵌入。

        没装 flow map ⇒ 与改动之前 **逐字相同** 的一行（回归锚点靠这个）。
        装了 ⇒ 按 flowmap_form 融合。三个要点：
        * 融合发生在 **time_projection 之前**，而 time_projection[0] 就是 SiLU ——
          `dit.time_projection(blended)` 逐字等于上游的 `time_proj(act_fn(rt_emb))`。
        * 两个分支共用 **同一个** 正弦编码和同一个 0..1000 量纲。r 用 [0,1] 的 σ 而 t 用 [0,1000]
          的话，正弦编码会悄悄塌成近似常数。
        * 门与 delta 分支的输出统一 cast 到 **主分支的 dtype**，绝不让 fp32 的门把 bf16 的和提升成
          fp32（上游 :326 把门 cast 成第三种 dtype，混合精度提升进 time_proj，bf16 下 r==t 的恒等性
          在 1e-3 量级就破了）。
        """
        dit = self.dit
        b, S = t_tok.shape
        e_t = sinusoidal_embedding_1d(dit.freq_dim, t_tok.reshape(-1)).view(b, S, -1)
        base = dit.time_embedding(e_t)
        if r_tok is None:
            return base
        assert self.flowmap_form in FLOWMAP_FORMS, (
            f"dit 上有 delta 分支但包装的 flowmap_form 是 {self.flowmap_form!r} —— "
            "谁把两者拆开了（直接 add_module？绕过 install_flowmap？）")
        e_r = sinusoidal_embedding_1d(dit.freq_dim, r_tok.reshape(-1)).view(b, S, -1)
        delta = getattr(dit, DELTA_EMBEDDING_NAME)(e_r).to(base.dtype)
        g = getattr(dit, DELTA_GATE_NAME).to(base.dtype)
        if self.flowmap_form == "convex":
            return (1.0 - g) * base + g * delta
        return base + g * delta

    def _resolve_r(self, timestep, r_timestep):
        """r 的准入闸门，返回真正要用的 r（没装 flow map 时恒为 None）。

        上游那个不带 flow map 的 embedder **收下 r_timestep 然后忽略它**
        （transformer_far_wan_model.py:241），于是"忘了接 delta 分支"会退化成一个普通扩散模型，
        不报错、loss 照降。这里两边都硬断言：装了就必须给 r；没装就只准不给、或者给一个与 t 相同的 r。
        """
        if self.has_flowmap:
            assert r_timestep is not None, (
                "模型装了 flow map（dit.delta_embedding 存在），r_timestep / r_k 是必填的。"
                "扩散槽请显式传 r_timestep=timestep，别省略。"
            )
            return r_timestep
        if r_timestep is None:
            return None
        assert _ts_equal(timestep, r_timestep), (
            "模型没装 flow map（没有 dit.delta_embedding），r_timestep 只能是 None 或与 timestep 完全相同；"
            "给了一个不同的 r 说明调用方以为自己在跑 flow map —— 那会静默地跑成普通扩散。"
            "先 install_flowmap()。"
        )
        return None

    # ---- 前向 ----

    def forward(self, x: torch.Tensor, timestep, context: torch.Tensor, *,
                context_ids=None, offset_px=None, mask_input=None, first_frame_cond: bool = True,
                r_timestep=None, clean_x=None, aug_t=None,
                frame_index=None, kv_cache=None, commit: bool = True,
                offset_win=None, mask_in_k=None, angle_win=None, angle_px=None, scale_px=None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False) -> torch.Tensor:
        """x [b, 48, f, 30, 52] → [b, 48, f, 30, 52]（TF 模式只返回 noisy 半流，形状同样是 f 个 cell）。

        并行路径
            timestep     标量 / [b] / [b, f]（逐 cell）
            context      [b, 512, 4096]（整条）或 [b, f, 512, 4096]（逐 cell）
            context_ids  [b, f] 整数，同 id ⇒ 同串，用于逐 cell 去重
            offset_px    [b, f, 2] px；None ⇒ 普通查表 RoPE
            mask_input   [b, 1, f, 30, 52] float，known=1 / new=0
            first_frame_cond  cell 0 是干净锚点，其 timestep 与 r 都置 0
            r_timestep   flow map 的第二个时间步（0..1000，与 timestep 同量纲、同形状规则）。
                         装了 flow map ⇒ **必填**（扩散槽显式传 r_timestep=timestep）；
                         没装 ⇒ 只能是 None 或与 timestep 完全相同。first_frame_cond 对 r **同样**
                         生效 —— 否则 TI2V 锚点那一格会掉到对角线之外，而那正是最不能变的一格。
            clean_x      [b, 48, f, 30, 52] 干净历史流 ⇒ 打开 teacher forcing 双流（要求 causal=True）
            aug_t        clean 半流的时间步（None ⇒ 0，即精确干净）。clean 半流的 r **硬接** r := t
                         （= 上游把 clean_timestep 同时当两个参数传，trainer:359）：干净流恒在对角线上。

        流式路径（kv_cache 非 None）：转发给 `forward_frame`。这个转发是故意的 ——
        FSDP 的参数 all-gather 挂在 **包装模块的 forward** 上，**绝不要**在被包装的模型上直接调
        `forward_frame`。反过来 `init_kv_cache` 必须在裸模块（`unwrap(m)`）上调。
        """
        gc = bool(use_gradient_checkpointing or self.use_gradient_checkpointing)
        gc_off = bool(use_gradient_checkpointing_offload or self.use_gradient_checkpointing_offload)
        r_eff = self._resolve_r(timestep, r_timestep)   # 没装 flow map ⇒ None，下面整条 r 路径不存在

        if kv_cache is not None:
            # ★ FSDP 会重建 forward 的输入容器（见 _KVRef 的 docstring）：拿 ref 把**调用方的
            #   那一份** dict 找回来，否则 commit 写进拷贝、缓存永远是空的且不报错。
            ref = kv_cache.get("_ref")
            if ref is not None and ref.cache is not kv_cache:
                kv_cache = ref.cache
            # BLOCKER B1：first_frame_cond 抹的是"这个张量的第 0 帧"，单 cell 前向里就是全部
            assert first_frame_cond is False, (
                "流式路径必须 first_frame_cond=False：单 cell 前向 f==1，"
                "arope.py:227 的 t_tok[:,0]=0 会把 rollout 里每一个 cell 的时间步都抹成 0"
            )
            assert clean_x is None, "teacher forcing 是训练路径，不能与 KV 缓存同用"
            assert angle_px is None, "流式路径的相机朝向走 angle_win（[b, n_cached+1]，相对当前 cell）"
            assert offset_px is None, "流式路径的世界坐标走 offset_win（[b, n_cached+1, 2]，窗口锚定）"
            assert mask_input is None, "流式路径的 known 掩码走 mask_in_k（[b, 1, 1, 30, 52]）"
            assert not gc, (
                "绝不能对 KV rollout 做梯度检查点：滑窗 commit 是破坏性的，重算时读到的是驱逐后的缓存 —— "
                "缓存满了以后形状对、内容错、不报错（PORT_设计 risk 6）"
            )
            assert frame_index is not None, "流式路径必须给 frame_index（绝对 cell 序号）"
            return self.forward_frame(x, timestep, context, frame_index, kv_cache, commit,
                                      offset_win=offset_win, mask_in_k=mask_in_k, context_ids=context_ids,
                                      r_k=r_eff, angle_win=angle_win)

        dit = self.dit
        b, _, f_in, lat_h, lat_w = x.shape
        p_f, p_h, p_w = dit.patch_size
        n_tok = (lat_h // p_h) * (lat_w // p_w)
        tf = clean_x is not None
        if tf:
            assert self.causal, "teacher forcing 双流只对因果学生有意义（causal=True）"
            assert clean_x.shape == x.shape, f"clean_x {tuple(clean_x.shape)} 应与 x {tuple(x.shape)} 同形"

        # ---- 时间步：逐 token ----
        ts_noisy_tok = _per_token_timestep_pf(x, timestep, n_tok, first_frame_cond)  # [b, f*n_tok]
        # r 与 t 走 **同一个** 展开函数、同一个 first_frame_cond：cell 0 的 t 和 r 一起被抹成 0
        rs_noisy_tok = (None if r_eff is None
                        else _per_token_timestep_pf(x, r_eff, n_tok, first_frame_cond))
        if tf:
            # clean 半流：aug_t=None ⇒ 全 0（精确干净）。给了 aug_t 就整条按 aug_t，
            # 不再套 first_frame_cond —— clean cell 0 本来就是锚点，与其它 clean cell 同一个噪声级别。
            ts_clean_tok = (torch.zeros_like(ts_noisy_tok) if aug_t is None
                            else _per_token_timestep_pf(clean_x, aug_t, n_tok, False))
            t_tok = torch.cat([ts_clean_tok, ts_noisy_tok], dim=1)  # [b, 2*f*n_tok]
            # clean 半流恒在 r == t 的对角线上（--tf_aug > 0 时就是 r_clean = aug_t，不是 0）
            r_tok = None if rs_noisy_tok is None else torch.cat([ts_clean_tok, rs_noisy_tok], dim=1)
        else:
            t_tok = ts_noisy_tok
            r_tok = rs_noisy_tok
        t = self._embed_time(t_tok, r_tok)
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

        # ---- 文本：TF 时在 4D 层面复制（clean / noisy 两半共用同一条 cell 文本）----
        if tf and context.dim() == 4:
            context = torch.cat([context, context], dim=1)                      # [b, 2f, L, 4096]
            if context_ids is not None:
                context_ids = torch.cat([context_ids, context_ids], dim=1)      # [b, 2f]
        ctx = _embed_context(dit, context, context_ids)

        # ---- patch embedding（+ 零初始化的 mask 通道）----
        xt = dit.patch_embedding(x)
        mask_emb = self._mask_embed(mask_input, xt)
        if mask_emb is not None:
            xt = xt + mask_emb
        f, h, w = xt.shape[2:]
        assert f * h * w == ts_noisy_tok.shape[1], (
            f"patch 之后的 token 数 {f * h * w} 与时间步 {ts_noisy_tok.shape[1]} 不符（temporal patch 必须是 1）"
        )
        tokens = rearrange(xt, "b c f h w -> b (f h w) c").contiguous()
        s_half = f * h * w
        if tf:
            # clean 半流用同一份 mask embedding（数值上与重算一遍完全一样）
            xc = dit.patch_embedding(clean_x)
            if mask_emb is not None:
                xc = xc + mask_emb
            tokens = torch.cat([rearrange(xc, "b c f h w -> b (f h w) c").contiguous(), tokens], dim=1)

        # ---- RoPE：TF 时两半各自从 f_start=0 起算（cat 两份 21 帧的表，而不是造一张 42 帧的表）----
        if offset_px is None:
            assert angle_px is None, "给了 angle_px 就必须给 offset_px（坐标要一起旋）"
            freqs = build_world_freqs(dit, f, h, w, None, device=x.device)
        else:
            # angle_px [b, f] 弧度：**绝对**朝向，锚在 cell 0（并行路径里 cell 0 是干净的 TI2V 锚点，
            # 它的格点必须保持轴对齐）。None ⇒ 逐位走旋转支持加入之前的老路。
            freqs = build_world_freqs(dit, f, h, w, offset_px.to(torch.float64) / PX_PER_TOKEN,
                                      device=x.device, angle=angle_px, scale=scale_px)
        if tf:
            freqs = torch.cat([freqs, freqs], dim=_freqs_cat_dim(freqs))

        # ---- 掩码 + blocks ----
        attn_mask = self._get_mask(f, h * w, tokens.device, tf)
        f_cross = 2 * f if tf else f
        for block in dit.blocks:
            # 用 block(...) 而不是伸手拿 block 的子模块：FSDP 的 all-gather 挂在这个入口上
            tokens = gradient_checkpoint_forward(
                functools.partial(block, f=f_cross, attn_mask=attn_mask, kv=None),
                gc, gc_off,
                tokens, ctx, t_mod, freqs,
            )

        # ---- head：TF 时只取 noisy 半流（clean 前缀连同它的时间调制一起丢掉）----
        if tf:
            tokens = tokens[:, s_half:]
            t = t[:, s_half:]
        tokens = dit.head(tokens, t)
        return dit.unpatchify(tokens, (f, h, w))

    # ---- 滑窗 KV 缓存 ----

    def init_kv_cache(self, kv_window: int | None = 20, sink_size: int = 0, rope_cap: int = 20,
                      num_layers: int | None = None) -> dict:
        """每条 rollout 一份新缓存（换 clip 必须重建）。**在裸模块上调**（`unwrap(m).init_kv_cache(...)`）。

        kv_window=None ⇒ 绝对模式：缓存无限增长，时间位就是 frame_index（等价性测试 / 短 rollout 用）。
        kv_window 给了 ⇒ 滑窗模式：缓存 raw K，读的时候按窗口锚定坐标重算 RoPE，
        超出 sink + kv_window 就驱逐最老的非 sink cell。

        ARoPE 的部署配置是 sink_size=0：an earlier internal project 的 sink 把开头几帧永久钉在 RoPE 位 0..sink-1，
        但这里 h/w 的 RoPE 位是 **一直在漂的世界坐标**（实测 21 个 cell 内最大位移 373 px = 11.66 token），
        钉住的 sink 帧在 rollout 几十个 cell 之后会落在世界坐标几十个 token 之外，模型没见过。
        sink=0 + 窗口锚定 ⇒ 每个注意力窗口的 RoPE 坐标精确等于一段训练片段的坐标（只解决坐标 OOD；
        缓存 cell 的掩码通道按提交时锚点固化，见 arope_self_attn；latent 值离流形见 PORT_设计 R10）。
        """
        n = num_layers if num_layers is not None else len(self.dit.blocks)
        assert self.causal, "KV 缓存是因果学生的流式路径（causal=True）"
        if kv_window is not None:
            assert kv_window >= 1 and sink_size >= 0
            assert rope_cap == sink_size + kv_window, (
                f"rope_cap 必须 == sink_size + kv_window（{rope_cap} != {sink_size} + {kv_window}）："
                "时间位 tpos 恒等于窗口内已缓存的 cell 数，超过 cap 就是训练时没见过的相对距离"
            )
        cache = {
            "k": [None] * n, "v": [None] * n, "frames_cached": 0,
            "kv_window": kv_window, "sink_size": int(sink_size), "rope_cap": int(rope_cap),
            "hw": None, "layer": None, "commit": False, "freqs_cur": None, "freqs_hist": None,
        }
        # 自引用句柄：FSDP 会重建 forward 的输入容器，靠它在 forward 里找回**这一份** dict。
        # 非张量值 FSDP 按引用保留，所以 ref 穿得过去（见 _KVRef 的 docstring 与实测）。
        cache["_ref"] = _KVRef(cache)
        return cache

    @staticmethod
    def release_kv_cache(cache: dict) -> None:
        """用完一条 rollout 必须调 —— `_ref` 与 cache 互相引用，构成**引用环**。

        引用计数放不掉环，只能等循环 GC，而 GC 是惰性的：一条 21 cell 的 rollout 缓存十几 GB
        的 K/V，几步之内显存就爬满（实测 98 → 120 → 126 GB 然后 OOM）。
        `clear()` 把 dict 清空即断环，张量当场释放。
        """
        if cache is not None:
            cache.clear()

    @staticmethod
    def snapshot_kv_cache(cache: dict) -> dict:
        """浅快照：分叉一条历史给别人读写（stage4 OPSD 的教师上下文），**原缓存一个字节不动**。

        为什么浅拷贝就是正确的分支（与上游 GraphGWM `KVCache.clone_shallow` 同一条依据）：
        `arope_self_attn` 的提交趟写的是 `kv["k"][li] = _evict(torch.cat([k_hist, k_raw]), kv).detach()`
        —— `torch.cat` 造**新**张量、`_evict` 要么原样返回它要么再 cat 一个新的，然后**重新绑定**
        列表槽位；缓存里已有的张量从不被就地写。`frames_cached` 是 int、`freqs_*` / `hw` / `layer` /
        `commit` 也都是每次前向重新赋值的瞬态字段。所以：新 dict + 拷贝过的 'k'/'v' 列表 + 指向新 dict
        的 `_KVRef`，就是一条独立分支 —— 往快照里 commit 只改快照的槽位，原缓存的槽位仍指着旧张量。

        与 `forward_frame` 入口断言 `frames_cached == frame_index` 的配合（stage4 的用法）：
        真实历史 0..k−2 的快照 frames_cached == k−1；把学生的 cell k−1 以 frame_index=k−1 commit 进快照
        ⇒ frames_cached == k；随后给 cell k 打分（frame_index=k）正好通过。用完必须
        `release_kv_cache(snapshot)`：快照自己也带 `_ref` 引用环（原缓存不受影响，共享的张量仍被它持有）。
        """
        snap = {k: v for k, v in cache.items() if k != "_ref"}
        snap["k"] = list(cache["k"])
        snap["v"] = list(cache["v"])
        snap["_ref"] = _KVRef(snap)
        return snap

    @staticmethod
    def last_kv_frame(cache: dict) -> tuple[list, list]:
        """取**最近一次 commit** 的那个 cell 的 raw K/V（每层 [b, hw, dim]，RoPE 之前，clone 过、detach 过）。

        提交趟把新 cell cat 在末尾、`_evict` 只从 sink 之后丢最老的，所以最近一格恒是每层最后 hw 个 token。
        clone：调用方通常紧接着 `release_kv_cache` 这份缓存，切片视图会拖着整条 [b, n*hw, dim] 不放。
        stage4 OPSD 用它复现官方的写入顺序（见 `append_kv_frame`）。"""
        hw = cache["hw"]
        assert hw is not None and cache["k"][0] is not None, "缓存还没提交过任何 cell"
        ks = [t[:, -hw:].detach().clone() for t in cache["k"]]
        vs = [t[:, -hw:].detach().clone() for t in cache["v"]]
        return ks, vs

    @staticmethod
    def append_kv_frame(cache: dict, k_slices: list, v_slices: list, frame_index: int) -> None:
        """把一格**已经算好的** raw K/V 按提交趟的规则追加进缓存：逐层 cat + `_evict` + frames_cached 推进。

        与 `arope_self_attn` 的 commit 路径逐字相同（cat 造新张量、重新绑定槽位、原张量不动），所以对
        `snapshot_kv_cache` 的快照调用同样不碰原缓存。用途（stage4 OPSD，官方 gt_kv 的写入顺序）：官方教师把
        自己的 chunk k 写进缓存时缓存里还是 [GT 0..k−2, 教师的 k−1]，**之后**才把 k−1 原位换成 GT；于是打分
        k+1 时最近一格的 K/V（第 1 层起）是在"教师的 k−1"上下文里算出来的，不是在全 GT 历史上重算的。
        我们的缓存 append-only，只能：cell k 打分完把教师的 x0 commit 进它打分用的那条快照（上下文正是官方的），
        `last_kv_frame` 取走那格 raw K/V，cell k+1 时用本函数把它接到真值 0..k−1 的新快照末尾。
        raw K/V 与位置无关（RoPE 读时按 (tpos, offset_win) 重算），搬到另一条同长度的缓存末尾是精确的。"""
        assert cache["frames_cached"] == frame_index, (
            f"append_kv_frame：frame_index={frame_index} 但缓存最后提交的是 cell {cache['frames_cached'] - 1}")
        assert len(k_slices) == len(cache["k"]) == len(v_slices), "层数不符"
        hw = k_slices[0].shape[1]
        assert cache["hw"] in (None, hw), (cache["hw"], hw)
        k0 = cache["k"][0]
        n_cached = 0 if k0 is None else k0.shape[1] // hw
        if cache["kv_window"] is not None:
            cap = cache["sink_size"] + cache["kv_window"]
            assert n_cached == min(frame_index, cap), (n_cached, frame_index, cap)
        else:
            assert n_cached == frame_index, (n_cached, frame_index)
        cache["hw"] = hw
        for li in range(len(cache["k"])):
            kh, vh = cache["k"][li], cache["v"][li]
            kn, vn = k_slices[li], v_slices[li]
            assert kn.shape[1] == hw and vn.shape[1] == hw, (kn.shape, vn.shape)
            nk = kn if kh is None else torch.cat([kh, kn], dim=1)
            nv = vn if vh is None else torch.cat([vh, vn], dim=1)
            cache["k"][li] = _evict(nk, cache).detach()
            cache["v"][li] = _evict(nv, cache).detach()
        cache["frames_cached"] = frame_index + 1

    def forward_frame(self, x_k: torch.Tensor, t_k, ctx_k: torch.Tensor, frame_index: int,
                      kv_cache: dict, commit: bool = True, *,
                      offset_win=None, mask_in_k=None, context_ids=None, r_k=None,
                      angle_win=None) -> torch.Tensor:
        """单 cell 前向，注意力打到缓存里的历史 + 自己。**不要直接调**（走 `forward(..., kv_cache=...)`，
        否则 FSDP 不 all-gather 根模块的参数）。

        x_k:        [b, 48, 1, 30, 52]
        t_k:        标量 / [b] / [b, 1]，**原样使用**，绝不套 first_frame_cond（BLOCKER B1）
        r_k:        flow map 的第二时间步，规则同 t_k：**同样原样使用、同样不套 first_frame_cond**
                    （B1 对 r 一字不差地成立）。装了 flow map ⇒ 必填；没装 ⇒ None 或与 t_k 相同。
        ctx_k:      [b, 1, 512, 4096]（逐 cell）或 [b, 512, 4096]（整条）
        frame_index 绝对 cell 序号 k
        commit      True = 把这一帧的 K/V 写进缓存（干净/最终那一趟）；False = 中间去噪趟，只读不写。
                    滑窗 commit 是 append-only 的：同一个 frame_index commit 两次会把这帧塞进去两遍，
                    n_cached / 驱逐全乱。拦它的是入口的 `frames_cached == frame_index` 断言（每次
                    commit 把 frames_cached 推到 frame_index+1）；下面的 n_cached 断言只在缓存饱和
                    之前有效 —— k ≥ cap 后 `_evict` 让 n_cached 恒等于 cap，重复 / 漏 commit 形状对、
                    内容错、不报错（GEO-3）。
        offset_win  [b, n_cached+1, 2] px，**窗口锚定**：offset_win[:,0] 恒为 0（窗口里最老那个 cell），
                    offset_win[:,-1] 是当前 cell。None ⇒ 普通查表 RoPE。
        mask_in_k   [b, 1, 1, 30, 52]，用 **锚定之后** 的 offset 算的 known 掩码。
        """
        dit = self.dit
        assert self.causal, "forward_frame 是因果学生的流式路径（causal=True）"
        b, _, f1, lat_h, lat_w = x_k.shape
        assert f1 == 1, f"forward_frame 一次只处理一个 cell，得到 f={f1}"
        p_f, p_h, p_w = dit.patch_size
        h, w = lat_h // p_h, lat_w // p_w
        hw = h * w

        # ★ GEO-3：每个 frame_index 恰好 commit 一次。缓存饱和后 n_cached 恒 == cap，下面那条断言
        #   拦不住重复 / 漏 commit（形状对、内容错、不报错），只有这个计数器能。commit=False 的去噪趟
        #   不推进它，所以同一 frame_index 上反复 commit=False 再 commit=True 一次照样通过。
        assert kv_cache["frames_cached"] == frame_index, (
            f"frame_index={frame_index} 但缓存最后提交的是 cell {kv_cache['frames_cached'] - 1} "
            "—— 漏 commit 或重复 commit 了"
        )

        # 缓存长度以第 0 层为准（各层同步推进）
        k0 = kv_cache["k"][0]
        n_cached = 0 if k0 is None else k0.shape[1] // hw
        sliding = kv_cache["kv_window"] is not None
        if sliding:
            cap = kv_cache["sink_size"] + kv_cache["kv_window"]
            assert n_cached == min(frame_index, cap), (
                f"缓存里有 {n_cached} 个 cell，按 frame_index={frame_index} / 窗口 {cap} 应为 "
                f"{min(frame_index, cap)} —— 漏 commit 或重复 commit 了"
            )
        else:
            assert n_cached == frame_index, (
                f"绝对模式下缓存长度必须等于 frame_index（{n_cached} != {frame_index}）"
            )
        tpos = n_cached  # 当前 cell 的时间位；滑窗下恒等于 min(frame_index, rope_cap)

        # ---- 时间步：原样用 t_k / r_k，**没有** first_frame_cond ----
        r_eff = self._resolve_r(t_k, r_k)
        t_tok = _per_token_timestep_pf(x_k, t_k, hw, first_frame_cond=False)  # [b, hw]
        r_tok = (None if r_eff is None
                 else _per_token_timestep_pf(x_k, r_eff, hw, first_frame_cond=False))
        t = self._embed_time(t_tok, r_tok)
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

        ctx = _embed_context(dit, ctx_k, context_ids)

        xt = dit.patch_embedding(x_k)
        mask_emb = self._mask_embed(mask_in_k, xt)
        if mask_emb is not None:
            xt = xt + mask_emb
        assert xt.shape[2] == 1 and xt.shape[3] == h and xt.shape[4] == w
        tokens = rearrange(xt, "b c f h w -> b (f h w) c").contiguous()

        # ---- 窗口锚定的 RoPE ----
        if offset_win is None:
            off_win_tok = None
        else:
            assert offset_win.shape == (b, n_cached + 1, 2), (
                f"offset_win 形状应为 [b={b}, n_cached+1={n_cached + 1}, 2]，得到 {tuple(offset_win.shape)}"
            )
            assert bool(torch.all(offset_win[:, 0] == 0)), (
                "offset_win[:, 0] 必须是 0：世界坐标要锚在窗口里最老那个 cell 上"
                "（off_win = offset_px[a:k+1] - offset_px[a]）"
            )
            off_win_tok = offset_win.to(torch.float64) / PX_PER_TOKEN
        if angle_win is None:
            ang_win = None
        else:
            assert off_win_tok is not None, "给了 angle_win 就必须给 offset_win（坐标要一起进当前相机系）"
            ang_win = angle_win.to(device=tokens.device, dtype=torch.float64)
            assert ang_win.shape == (b, n_cached + 1), (
                f"angle_win 形状应为 [b={b}, n_cached+1={n_cached + 1}]，得到 {tuple(ang_win.shape)}"
            )
            assert bool(torch.all(ang_win[:, -1] == 0)), (
                "angle_win[:, -1] 必须是 0：一切都表达在**当前 cell 的相机系**里，"
                "当前帧自己的格点必须保持轴对齐（模型唯一见过的帧内结构）"
            )
        cur_off = None if off_win_tok is None else off_win_tok[:, n_cached:n_cached + 1]
        cur_ang = None if ang_win is None else ang_win[:, n_cached:n_cached + 1]
        freqs_cur = build_world_freqs(dit, 1, h, w, cur_off, device=tokens.device, f_start=tpos,
                                      angle=cur_ang)

        freqs_hist = None
        if n_cached > 0:
            n_sink = min(kv_cache["sink_size"], n_cached)
            R = n_cached - n_sink
            parts = []
            if n_sink > 0:
                # sink 段永久钉在时间位 0..n_sink-1（sink_size=0 时这一段不存在）
                off_s = None if off_win_tok is None else off_win_tok[:, :n_sink]
                ang_s = None if ang_win is None else ang_win[:, :n_sink]
                parts.append(build_world_freqs(dit, n_sink, h, w, off_s, device=tokens.device, f_start=0,
                                               angle=ang_s))
            if R > 0:
                off_r = None if off_win_tok is None else off_win_tok[:, n_sink:n_cached]
                ang_r = None if ang_win is None else ang_win[:, n_sink:n_cached]
                parts.append(build_world_freqs(dit, R, h, w, off_r, device=tokens.device, f_start=tpos - R,
                                               angle=ang_r))
            freqs_hist = parts[0] if len(parts) == 1 else torch.cat(parts, dim=_freqs_cat_dim(parts[0]))

        kv_cache["hw"] = hw
        kv_cache["commit"] = bool(commit)
        kv_cache["freqs_cur"] = freqs_cur
        kv_cache["freqs_hist"] = freqs_hist

        # 直接调 block，**绝不** gradient_checkpoint_forward（risk 6：滑窗 commit 是破坏性的，
        # 重算时读到的是驱逐之后的缓存，缓存满了以后形状对内容错、不报错）
        for li, block in enumerate(dit.blocks):
            kv_cache["layer"] = li
            tokens = block(tokens, ctx, t_mod, freqs_cur, f=1, attn_mask=None, kv=kv_cache)

        tokens = dit.head(tokens, t)
        if commit:
            kv_cache["frames_cached"] = frame_index + 1
        return dit.unpatchify(tokens, (1, h, w))


# 模块级别名（与 AropeDiT.release_kv_cache 一样是静态的；stage4 OPSD 直接 import 它们）
snapshot_kv_cache = AropeDiT.snapshot_kv_cache
last_kv_frame = AropeDiT.last_kv_frame
append_kv_frame = AropeDiT.append_kv_frame
