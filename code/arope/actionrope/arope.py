"""ARoPE：世界坐标 RoPE + known/new 支持掩码 + 逐 cell 文本 cross-attn。

为什么动作走坐标而不是走文本
-----------------------------
动作（玩家/相机位移）本质上是几何量：玩家向右走 D px ⇒ 背景整体左移 D px ⇒
现在屏幕位置 s 上的内容，在首帧位于 s + D。文本只能给模型一个离散标签
（"moving right"），模型得自己从数据里反推"右移多少像素"；而 RoPE 是相对位置编码，
把每帧 token 的 (h, w) 位置写成 **世界坐标**（窗口内格坐标 + 该帧相机累计偏移，
以 token 为单位），两帧里同一世界位置的 token 就拿到同一编码，注意力天然按世界
位置对齐。这一步不需要任何新参数（`mask_channel=False` 时 ckpt 键集与原模型完全一致）。

符号约定（与 设计.md 一致）
--------------------------
* 屏幕坐标 u 向右、v 向下，单位 px（832×480 画幅）；token = 32 px（latent 16 px × patch 2）。
* `offset_px[b, k] = (dx, dy)`：cell k 相对 cell 0 的玩家屏幕投影位移，cell 0 恒为 0。
* cell k 中屏幕格 (i_h, j_w) 的 token 世界坐标 = `(i_h + dy/32, j_w + dx/32)`（允许小数）。
* known / new：世界坐标落在首帧足迹 `[-0.5, h-0.5] × [-0.5, w-0.5]` 内（含边界）⇒ known，
  否则 new。latent 分辨率同理（30×52，偏移除以 16）。new 区 loss × `new_weight`。

与 `model_fn_wan_video` 的对应关系
--------------------------------
`arope_forward` 是 `diffsynth.pipelines.wan_video.model_fn_wan_video` 在
`seperated_timestep=True and fuse_vae_embedding_in_latents=True` 分支上的复刻，
只改了三处（其它每一步按原函数逐行照抄，`offset_px=None, mask_input=None, context 整条`
时输出与原函数一致）：

1. RoPE 表：原函数把 `dit.freqs` 按 (f, h, w) 查表拼成 `[f*h*w, 1, 64]`；
   这里 f 轴照旧查表，h/w 轴按世界坐标用 `polar(1, pos * inv_freq)` 现算，
   得到逐样本的 `[b, f*h*w, 1, 64]`（`rope_apply` 里 complex 乘法可直接广播）。
2. patch embedding：`x = patch_embedding(latents) + arope_mask_embedding(mask_input)`，
   后者零初始化，所以刚装上时不改变输出。
3. cross-attn：context 为 `[b, f, L, 4096]` 时每帧 token 只 attend 自己 cell 的文本
   （q 重排成 `(b f) (h w) c`，k/v 为 `(b f) L c`）。cell 文本相同 ⇒ 与整条模式等价
   （SDPA 的 batch 切分方式不同，只有 bf16 级别的舍入差）。

block 内部的计算（norm / modulate / self_attn / cross_attn / ffn / gate）照抄
`DiTBlock.forward`，不改 DiffSynth 源码；梯度检查点用 DiffSynth 自己的
`gradient_checkpoint_forward`（`diffsynth/core/gradient/gradient_checkpoint.py`），
所以 DeepSpeed / 非重入 checkpoint / CPU offload 三条路径与原模型一致。
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
from einops import rearrange

from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import (
    WanModel,
    modulate,
    sinusoidal_embedding_1d,
)

# token = 32 px：Wan2.2 VAE 空间压缩 16×，DiT patch (1,2,2) 再 2×。latent 格 = 16 px。
PX_PER_TOKEN = 32
PX_PER_LATENT = 16

# RoPE 的 theta，与 precompute_freqs_cis 的默认值一致
ROPE_THETA = 10000.0


# --------------------------------------------------------------------------
# 世界坐标 RoPE
# --------------------------------------------------------------------------

# 格距混叠频带（，T2）：把 h/w 轴（dim=42）某条休眠低频带的逆频率换成 ω（如 2π）。整数 token 位上
# e^{i·2π·n}=1 ⇒ 普通 RoPE / 同 cell 内相对位置逐位不变（原带 ω≈1.5e-4，26 token 内只转 0.004 rad，本就≈恒等）；
# 只有跨 cell 且偏移带小数时才被调制：相位 = 2π·frac(Δoffset)，即把"坐标的小数部分"显式交给注意力。
# 由 train.py --rope_alias_band "20:6.283185307" 设置并写进 ckpt metadata，infer.load_models 读回。
ALIAS_BANDS: dict = {}
HW_ROPE_DIM = 42


def _inv_freq(dim: int, device=None) -> torch.Tensor:
    """与 precompute_freqs_cis(dim) 完全相同的逆频率：1/theta^(arange(0,dim,2)/dim)，float64。
    dim == HW_ROPE_DIM 且 ALIAS_BANDS 非空时按带覆盖（见上）。"""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].double() / dim))
    if ALIAS_BANDS and dim == HW_ROPE_DIM:
        inv = inv.clone()
        for band, omega in ALIAS_BANDS.items():
            inv[int(band)] = float(omega)
    return inv


def set_alias_bands(spec: str | None) -> dict:
    """"20:6.283185307,19:3.14159" → {20: 6.28…, 19: 3.14…}；空/None ⇒ 关。返回生效的 dict。"""
    ALIAS_BANDS.clear()
    if spec:
        for tok in spec.split(","):
            if tok.strip():
                b, w = tok.split(":")
                ALIAS_BANDS[int(b)] = float(w)
    return dict(ALIAS_BANDS)


def _axis_freqs(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """任意（可小数）位置的 1D RoPE：polar(1, pos * inv_freq)。

    pos 为整数时与 precompute_freqs_cis 查表逐位相同：两边都是 float64 位置乘
    float64 逆频率再取 polar，单次乘法在 IEEE 下与 torch.outer 一致。
    返回 complex128（precompute_freqs_cis 的真实 dtype；源码注释写 complex64 是错的）。
    """
    angle = pos.double().unsqueeze(-1) * _inv_freq(dim, pos.device)
    return torch.polar(torch.ones_like(angle), angle)


def build_world_freqs(dit: WanModel, f: int, h: int, w: int, offset_tok=None, device=None, t_index=None) -> torch.Tensor:
    """把 (frame_idx, i_h + dy_tok[f], j_w + dx_tok[f]) 编成 rope_apply 可用的 freqs。

    offset_tok: [b, f, 2] = (dx_tok, dy_tok)，单位 token；None ⇒ 普通 RoPE，
    此时返回 [f*h*w, 1, 64] 且与 model_fn_wan_video 的查表结果逐位相同。
    否则返回 [b, f*h*w, 1, 64]。dtype 跟随 dit.freqs（complex128）。
    """
    f_table, h_table, w_table = dit.freqs
    # t_index [f]：逐 cell 的时间位（默认 0..f−1）。World Canvas 的"全景锚点"读法用它把画布格放到 t=0：
    # 多张干净 cell 同在 t=0、只在世界坐标上错开 —— 等于一张更大的首帧。
    if t_index is not None:
        # 允许小数 / 负数 / 超表长的时间位：现算（整数时与查表逐位相同，见 _axis_freqs）
        t_pos = torch.as_tensor(t_index, dtype=torch.float64, device=f_table.device).reshape(-1)
        assert t_pos.numel() == f, (t_pos.shape, f)
        f_rows = _axis_freqs(t_pos, f_table.shape[1] * 2).to(f_table.dtype)
    else:
        f_rows = f_table[:f]
    if device is None:
        device = offset_tok.device if offset_tok is not None else f_table.device
    d_h, d_w = h_table.shape[1] * 2, w_table.shape[1] * 2  # 42, 42

    if offset_tok is None:
        # 走与原函数一样的查表，保证等价性；不重新算 h/w 轴
        return torch.cat([
            f_rows.view(f, 1, 1, -1).expand(f, h, w, -1),
            h_table[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_table[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)

    offset_tok = offset_tok.to(device=device, dtype=torch.float64)
    b = offset_tok.shape[0]
    assert offset_tok.shape == (b, f, 2), f"offset_tok 形状应为 [b, {f}, 2]，得到 {tuple(offset_tok.shape)}"
    dx = offset_tok[..., 0]  # [b, f]
    dy = offset_tok[..., 1]
    pos_h = torch.arange(h, device=device, dtype=torch.float64).view(1, 1, h) + dy.view(b, f, 1)  # [b, f, h]
    pos_w = torch.arange(w, device=device, dtype=torch.float64).view(1, 1, w) + dx.view(b, f, 1)  # [b, f, w]
    fh = _axis_freqs(pos_h, d_h)  # [b, f, h, 21]
    fw = _axis_freqs(pos_w, d_w)  # [b, f, w, 21]
    freqs = torch.cat([
        f_rows.to(device).view(1, f, 1, 1, -1).expand(b, f, h, w, -1),
        fh.view(b, f, h, 1, -1).expand(b, f, h, w, -1),
        fw.view(b, f, 1, w, -1).expand(b, f, h, w, -1),
    ], dim=-1)
    return freqs.reshape(b, f * h * w, 1, -1).to(f_table.dtype)


# --------------------------------------------------------------------------
# 支持掩码
# --------------------------------------------------------------------------

def _known_from_offset(offset: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """通用版：offset [b, f, 2] 已换算成当前格子的单位（token 或 latent）。

    格 (i, j) 的世界坐标 (i + dy, j + dx) 落在首帧足迹 [-0.5, h-0.5] × [-0.5, w-0.5]
    内（含边界）⇒ known。边界取闭区间：偏移恰好半格时那一行/列仍算 known。
    """
    offset = offset.to(torch.float64)
    b, f = offset.shape[:2]
    dx = offset[..., 0].view(b, f, 1, 1)
    dy = offset[..., 1].view(b, f, 1, 1)
    wi = torch.arange(h, device=offset.device, dtype=torch.float64).view(1, 1, h, 1) + dy
    wj = torch.arange(w, device=offset.device, dtype=torch.float64).view(1, 1, 1, w) + dx
    in_h = (wi >= -0.5) & (wi <= h - 0.5)
    in_w = (wj >= -0.5) & (wj <= w - 0.5)
    return in_h & in_w  # [b, f, h, w]


def known_mask(offset_tok: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """token 分辨率的 known 掩码。offset_tok [b, f, 2]（token 单位）→ bool [b, f, h, w]。"""
    return _known_from_offset(offset_tok, h, w)


def _dist_outside_footprint(offset: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """每个格子的世界坐标到首帧足迹矩形的欧氏距离（格子单位），足迹内为 0。offset 已换算成格子单位。"""
    offset = offset.to(torch.float64)
    b, f = offset.shape[:2]
    dx = offset[..., 0].view(b, f, 1, 1)
    dy = offset[..., 1].view(b, f, 1, 1)
    wi = torch.arange(h, device=offset.device, dtype=torch.float64).view(1, 1, h, 1) + dy
    wj = torch.arange(w, device=offset.device, dtype=torch.float64).view(1, 1, 1, w) + dx
    dh = torch.clamp(torch.maximum(-0.5 - wi, wi - (h - 0.5)), min=0.0)
    dw = torch.clamp(torch.maximum(-0.5 - wj, wj - (w - 0.5)), min=0.0)
    return torch.sqrt(dh * dh + dw * dw)  # [b, f, h, w]


def cns_sigma(s: float, kap: torch.Tensor, alpha: float, mode: str = "power", shift: float = 5.0,
              shift_known: float = 1.0) -> torch.Tensor:
    """CNS 逐 token 噪声档 σ_tok(σ, κ)（训练与采样共用）。
    power：σ^(1+ακ)。shift：先把 σ 还原成线性时间 u = σ/(S−(S−1)σ)（Wan：σ = S·u/(1+(S−1)u)），
    再按 S_tok = S·(S_known/S)^(ακ) 重新变换 σ_tok = S_tok·u/(1+(S_tok−1)u)；κ=1（α=1）走 shift=S_known 的日程，
    差距开在采样早期而不是晚期。两种模式 σ=1→1、σ=0→0、κ=0 与基线逐步相同。"""
    if mode == "power":
        return s ** (1.0 + alpha * kap)
    if mode != "shift":
        raise ValueError(f"未知 cns_mode {mode!r}")
    S = float(shift); u = s / (S - (S - 1.0) * s)
    S_tok = S * (float(shift_known) / S) ** (alpha * kap)
    return S_tok * u / (1.0 + (S_tok - 1.0) * u)


def knownness_map(offset_px: torch.Tensor, lat_h: int = 30, lat_w: int = 52, tau_latent: float = 0.0) -> torch.Tensor:
    """坐标条件噪声日程（CNS）用的"已知度" κ ∈ [0,1]：float32 [b, 1, f, lat_h, lat_w]。
    世界坐标落在首帧足迹内 ⇒ 1；足迹外按到足迹边界的距离衰减 exp(−d/τ)（tau_latent=0 ⇒ 硬 0/1）。
    每个 token 的噪声档 σ_tok = σ^(1 + α·κ)：已知区（内容由首帧经 RoPE 决定）走更低的噪声、更早收敛，
    新区保持完整噪声并以更干净的已知区为条件 —— "该有多少噪声"由世界坐标决定，而不是所有 token 一个水平。"""
    off = offset_px.to(torch.float64) / PX_PER_LATENT
    d = _dist_outside_footprint(off, lat_h, lat_w)                        # [b, f, h, w]，足迹内为 0
    if tau_latent > 0:
        kappa = torch.exp(-d / float(tau_latent))
    else:
        kappa = (d <= 0).to(torch.float64)
    return kappa.to(torch.float32).unsqueeze(1)


def loss_weight_map(offset_px: torch.Tensor, lat_h: int = 30, lat_w: int = 52, new_weight: float = 2.0,
                    ramp_latent: float = 0.0, sigma=None) -> torch.Tensor:
    """latent 分辨率的 loss 权重。offset_px [b, f, 2] → float32 [b, 1, f, lat_h, lat_w]。

    ramp_latent=0（默认，flat）：known=1，new=new_weight，一刀切。
    ramp_latent>0（dist）：new 区权重按离首帧足迹边界的距离线性爬升，紧贴边界 ≈1（几乎是已知内容的延伸），
      距离 ≥ ramp_latent 格时到 new_weight（纯发明区）。
    sigma（0..1，None 不用）：额外部分 (w−1) 再乘 (0.5+σ)——高噪声步决定布局，new 区多给；低噪声步只剩纹理，少给。
    """
    off = offset_px.to(torch.float64) / PX_PER_LATENT
    if ramp_latent > 0:
        d = _dist_outside_footprint(off, lat_h, lat_w)
        extra = (float(new_weight) - 1.0) * torch.clamp(d / float(ramp_latent), max=1.0)
    else:
        known = _known_from_offset(off, lat_h, lat_w)
        extra = torch.where(known, 0.0, float(new_weight) - 1.0)
    if sigma is not None:
        s = torch.as_tensor(sigma, dtype=torch.float64, device=extra.device).reshape(-1, 1, 1, 1)
        extra = extra * (0.5 + s)
    return (1.0 + extra).to(torch.float32).unsqueeze(1)


# --------------------------------------------------------------------------
# 安装：可选的 mask 输入通道（零初始化 conv）
# --------------------------------------------------------------------------

MASK_EMBEDDING_NAME = "arope_mask_embedding"
# 零参数 known 标签：不学 conv，把固定的单位向量×尺度按 κ 加到已知 token 的输入上（非持久 buffer，不进 state_dict）
KNOWN_TAG_NAME = "arope_known_tag"
# World Canvas 输入通道（actionrope/world_canvas.py）：48 latent + 1 有效位，同样零初始化
CANVAS_EMBEDDING_NAME = "arope_canvas_embedding"
CANVAS_IN_CHANNELS = 49


def _zero_conv(dit: WanModel, in_ch: int) -> nn.Conv3d:
    ref = dit.patch_embedding
    conv = nn.Conv3d(in_ch, dit.dim, kernel_size=dit.patch_size, stride=dit.patch_size)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    conv.to(device=ref.weight.device, dtype=ref.weight.dtype)
    return conv


def install_arope(dit: WanModel, mask_channel: bool = True, canvas_channel: bool = False, known_tag: float = 0.0) -> WanModel:
    """给 DiT 装 ARoPE。不改任何已有参数名。

    mask_channel=True ⇒ 新增 dit.arope_mask_embedding = Conv3d(1, dim, patch, stride=patch)，
    权重与偏置零初始化：装上的瞬间输出与原模型完全相同，训练时再学会用 known/new 信息。
    mask_channel=False ⇒ 严格零参数版（若之前装过则拆掉），state_dict 键集与原模型完全一致。
    canvas_channel=True ⇒ 再新增 dit.arope_canvas_embedding = Conv3d(49, dim, patch, stride=patch)（零初始化），
    吃 world_canvas.build_canvas_input 的 [b, 49, f, 30, 52]；False ⇒ 拆掉（若装过）。
    known_tag>0 ⇒ 零参数 known 标签：dit.arope_known_tag = 固定单位向量（seed 0）× known_tag（L2 范数），
    前向时按 κ 的 patch 均值加到已知 token 的 patch embedding 上（mask_input 走同一入口）；非持久 buffer，键集不变。
    """
    if mask_channel:
        if not hasattr(dit, MASK_EMBEDDING_NAME):
            dit.add_module(MASK_EMBEDDING_NAME, _zero_conv(dit, 1))
    elif hasattr(dit, MASK_EMBEDDING_NAME):
        delattr(dit, MASK_EMBEDDING_NAME)
    if canvas_channel:
        if not hasattr(dit, CANVAS_EMBEDDING_NAME):
            dit.add_module(CANVAS_EMBEDDING_NAME, _zero_conv(dit, CANVAS_IN_CHANNELS))
    elif hasattr(dit, CANVAS_EMBEDDING_NAME):
        delattr(dit, CANVAS_EMBEDDING_NAME)
    if hasattr(dit, KNOWN_TAG_NAME):
        delattr(dit, KNOWN_TAG_NAME)
    if known_tag and known_tag > 0:
        ref = dit.patch_embedding.weight
        v = torch.randn(dit.dim, generator=torch.Generator().manual_seed(0))
        v = (v / v.norm() * float(known_tag)).to(device=ref.device, dtype=ref.dtype)
        dit.register_buffer(KNOWN_TAG_NAME, v, persistent=False)
    return dit


def wants_mask_input(dit) -> bool:
    """前向要不要喂 mask_input（known 图）：学的 conv 通道或零参数固定标签任一装了就要。"""
    return hasattr(dit, MASK_EMBEDDING_NAME) or hasattr(dit, KNOWN_TAG_NAME)


# --------------------------------------------------------------------------
# 前向
# --------------------------------------------------------------------------

def _per_token_timestep(latents: torch.Tensor, timestep: torch.Tensor, n_tok_per_frame: int, first_frame_cond: bool,
                        clean_cells=None, timestep_map=None) -> torch.Tensor:
    """照抄 model_fn_wan_video 的 seperated_timestep 分支：帧 0 的 t=0，其余帧 t=timestep。

    原函数只支持 timestep 形状 [1]（t_mod 的 batch 维恒为 1，靠广播作用到 b 个样本）。
    这里放宽到 [b]：每个样本一个 timestep。dtype 跟 latents（训练链路里 timestep 本来就是 bf16）。
    返回 [b, f * n_tok_per_frame]。
    """
    b, f = latents.shape[0], latents.shape[2]
    if timestep_map is not None:
        # CNS：逐 token 的时间步 [b, f, tok_h, tok_w]（1000 制），已含坐标条件的噪声档
        assert timestep_map.shape[0] == b and timestep_map.shape[1] == f, (timestep_map.shape, b, f)
        t_tok = timestep_map.to(device=latents.device, dtype=latents.dtype).reshape(b, f, n_tok_per_frame).clone()
        if first_frame_cond:
            t_tok[:, 0] = 0
        for k in (clean_cells or ()):
            t_tok[:, int(k)] = 0
        return t_tok.reshape(b, f * n_tok_per_frame)
    timestep = torch.as_tensor(timestep, device=latents.device).to(latents.dtype).reshape(-1)
    if timestep.numel() == 1:
        timestep = timestep.expand(b)
    assert timestep.numel() == b, f"timestep 应为标量或 [b={b}]，得到 {timestep.numel()} 个"
    t_tok = timestep.view(b, 1, 1).expand(b, f, n_tok_per_frame).clone()
    if first_frame_cond:
        t_tok[:, 0] = 0
    # World Canvas 的"锚点 token"读法：除帧 0 外再把若干 cell 当干净条件（t=0），调用方负责在这些 cell 放干净 latent
    for k in (clean_cells or ()):
        t_tok[:, int(k)] = 0
    return t_tok.reshape(b, f * n_tok_per_frame)


def _dedup_rows(rows: torch.Tensor, ids=None):
    """把 [N, L, C] 里内容相同的行去重。返回 (unique_index [U], inverse [N])。

    有 ids（[N] 整数，相同 id ⇒ 同串）时直接按 id 分组；否则先用一个便宜的指纹分桶，
    桶内再用 torch.equal 确认，避免对 512×4096 的串做全量排序。
    """
    n = rows.shape[0]
    if ids is not None:
        keys = [int(k) for k in torch.as_tensor(ids).reshape(-1).tolist()]
        assert len(keys) == n
    else:
        fp = rows[:, :, ::61].float().sum(dim=(1, 2)).tolist()  # 61 与 4096 互素，采样均匀
        keys = []
        buckets: dict[float, list[int]] = {}
        for i in range(n):
            key = None
            for j in buckets.get(fp[i], []):
                if torch.equal(rows[i], rows[j]):
                    key = j
                    break
            if key is None:
                key = i
                buckets.setdefault(fp[i], []).append(i)
            keys.append(key)
    first: dict = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    uniq_keys = list(first)
    key_to_u = {k: u for u, k in enumerate(uniq_keys)}
    unique_index = torch.tensor([first[k] for k in uniq_keys], device=rows.device)
    inverse = torch.tensor([key_to_u[k] for k in keys], device=rows.device)
    return unique_index, inverse


def _embed_context(dit: WanModel, context: torch.Tensor, context_ids=None):
    """context [b, L, 4096] → [b, L, dim]；[b, f, L, 4096] → [(b f), L, dim]，text_embedding 只对唯一串算一次。"""
    if context.dim() == 3:
        return dit.text_embedding(context)
    assert context.dim() == 4, f"context 应为 [b, L, 4096] 或 [b, f, L, 4096]，得到 {tuple(context.shape)}"
    b, f, L, c = context.shape
    rows = context.reshape(b * f, L, c)
    unique_index, inverse = _dedup_rows(rows, context_ids)
    emb = dit.text_embedding(rows[unique_index])  # [U, L, dim]
    return emb[inverse]  # [(b f), L, dim]


def _block_forward(block, x, context, t_mod, freqs, feats=None, *, f: int, arm=None, idx: int = 0, grid=None):
    """照抄 DiTBlock.forward，只把 cross_attn 换成按 cell 分组的版本。

    context 为 [b, L, dim]（整条）时与 DiTBlock.forward 完全一样；
    为 [(b f), L, dim]（逐 cell）时 q 重排成 (b f) (h w) c 与之配对。

    arm / feats / idx / grid：baseline 动作臂的钩子（见 baseline/设计.md）。arm=None 时与原实现逐位相同。
    feats 作为张量参数传进来（而不是闭包）是为了梯度检查点下动作编码器的梯度能回传。
    """
    if arm is not None:
        x = arm.block_pre(idx, x, feats, grid)
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    x = block.gate(x, gate_msa, block.self_attn(input_x, freqs))
    if arm is not None:
        x = arm.block_mid(idx, x, feats, grid)  # 自注意力之后、文本 cross-attn 之前（Matrix-Game 式动作 cross-attn 放这里）

    q_in = block.norm3(x)
    if context.shape[0] == x.shape[0]:
        x = x + block.cross_attn(q_in, context)
    else:
        b = x.shape[0]
        assert context.shape[0] == b * f, f"逐 cell context 应有 b*f={b * f} 行，得到 {context.shape[0]}"
        q_in = rearrange(q_in, "b (f n) c -> (b f) n c", f=f)
        y = block.cross_attn(q_in, context)
        x = x + rearrange(y, "(b f) n c -> b (f n) c", f=f)
    if arm is not None:
        x = arm.block_pre_ffn(idx, x, feats, grid)  # 文本 cross-attn 之后、FFN 之前（Matrix-Game 3 上游的注入位置）

    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(input_x))
    if arm is not None:
        x = arm.block_post(idx, x, feats, grid)
    return x


def arope_forward(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    *,
    offset_px=None,
    first_frame_cond: bool = True,
    mask_input=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    context_ids=None,
    arm=None,
    action_inputs=None,
    canvas_input=None,
    clean_cells=None,
    t_index=None,
    timestep_map=None,
) -> torch.Tensor:
    """ARoPE 版 DiT 前向。latents [b, 48, f, 30, 52] → [b, 48, f, 30, 52]。

    offset_px:  [b, f, 2] (dx, dy) px；None ⇒ 普通 RoPE。
    first_frame_cond: 帧 0 为干净条件帧，其 timestep 置 0（TI2V 的 seperated_timestep 路径）。
    mask_input: [b, 1, f, 30, 52] float，known=1 / new=0；None ⇒ 不加。给了但没 install ⇒ 报错，
                避免"以为用了掩码其实没用"这种静默错误。
    canvas_input: [b, 49, f, 30, 52] float（world_canvas.build_canvas_input），None ⇒ 不加；给了但没 install ⇒ 报错。
    context:    [b, 512, 4096]（整条）或 [b, f, 512, 4096]（逐 cell）。
    context_ids: [b, f] 整数，相同 id ⇒ 同串，用于逐 cell 模式下去重；None ⇒ 按内容去重。
    arm / action_inputs: baseline 动作臂（baseline/设计.md 的 ActionArm）与它的输入 dict；None ⇒ 不注入。
    """
    b, _, f, lat_h, lat_w = latents.shape
    p_f, p_h, p_w = dit.patch_size
    n_tok_per_frame = (lat_h // p_h) * (lat_w // p_w)

    # 时间步：逐 token，帧 0 为 0（照抄 model_fn_wan_video）
    t_tok = _per_token_timestep(latents, timestep, n_tok_per_frame, first_frame_cond, clean_cells, timestep_map)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_tok.reshape(-1)).view(b, t_tok.shape[1], -1))
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

    grid = None
    feats = None
    if arm is not None:
        grid = dict(b=b, f=f, h=lat_h // p_h, w=lat_w // p_w, n_tok_per_frame=n_tok_per_frame)
        feats = arm.encode(action_inputs, grid)
        t_mod = arm.modify_t_mod(t_mod, feats, grid)

    # 文本
    context = _embed_context(dit, context, context_ids)
    if arm is not None:
        context = arm.extra_context(context, feats, grid)

    # patch embedding（+ 零初始化的 mask 通道）
    x = dit.patch_embedding(latents)
    if mask_input is not None:
        m = mask_input.to(device=x.device, dtype=x.dtype)
        if hasattr(dit, MASK_EMBEDDING_NAME):
            x = x + getattr(dit, MASK_EMBEDDING_NAME)(m)
        elif hasattr(dit, KNOWN_TAG_NAME):
            # 零参数 known 标签：κ 按 patch 取均值到 token 网格，乘固定向量加到输入（残差流全程带着这个标记）
            tag = getattr(dit, KNOWN_TAG_NAME).to(device=x.device, dtype=x.dtype)
            m_tok = torch.nn.functional.avg_pool3d(m, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
            x = x + m_tok * tag.view(1, -1, 1, 1, 1)
        else:
            raise ValueError("给了 mask_input 但 dit 既没有 arope_mask_embedding 也没有 arope_known_tag，请先 install_arope(dit, mask_channel=True) 或 known_tag>0")
    if canvas_input is not None:
        if not hasattr(dit, CANVAS_EMBEDDING_NAME):
            raise ValueError("给了 canvas_input 但 dit 没有 arope_canvas_embedding，请先 install_arope(dit, canvas_channel=True)")
        x = x + getattr(dit, CANVAS_EMBEDDING_NAME)(canvas_input.to(device=x.device, dtype=x.dtype))
    f, h, w = x.shape[2:]
    assert f * h * w == t_tok.shape[1]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    if arm is not None:
        x = arm.after_patch(x, feats, grid)

    # RoPE：普通查表 或 世界坐标现算
    if offset_px is None:
        freqs = build_world_freqs(dit, f, h, w, None, device=x.device, t_index=t_index)
    else:
        freqs = build_world_freqs(dit, f, h, w, offset_px.to(torch.float64) / PX_PER_TOKEN, device=x.device, t_index=t_index)

    # blocks（梯度检查点走 DiffSynth 自己的封装；f 用 partial 绑定，checkpoint 只收张量参数）
    for i, block in enumerate(dit.blocks):
        x = gradient_checkpoint_forward(
            functools.partial(_block_forward, block, f=f, arm=arm, idx=i, grid=grid),
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x, context, t_mod, freqs, feats,
        )

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))
