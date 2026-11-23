"""gc25d 模型：Wan2.2-TI2V-5B DiT + 逐像素数字语义图条件。

    x = dit.patch_embedding(latents) + dit.gc_sem_cond(sem_ids)      # [b, 3072, 21, H/32, W/32]

形状对分辨率通用（全部从输入张量取）；下面的数是 832×480（geometry 默认 GC_RES）的例子，1280×704 时
H/4×W/4 = 176×320、latent 44×80、token 22×40。
`gc_sem_cond`（SemCondEncoder）把 81×480×832 的 uint8 语义 id 视频压到 latent 网格：
  1. LUT：raw 0..6 → 0..6，10 → 7，11 → 8，其余（含 UNKNOWN=255）→ 9（K=10）；
  2. 逐类 4×4 平均池化 one-hot → [b, K, 81, 120, 208]（一类一类算，不物化全分辨率 one-hot）；
  3. 时间分组同 VAE：帧 0 重复 4 次 + 帧 1..80 → [b, 4K, 21, 120, 208]；
  4. pixel-unshuffle ×4 → [b, 64K=640, 21, 30, 52]；
  5. Conv3d(640, 512, 1) → SiLU → Conv3d(512, 512, (1,3,3), pad (0,1,1)) → SiLU →
     零初始化 Conv3d(512, dim, (1,2,2), stride (1,2,2))。
刚装上时最后一层权重与偏置全 0 ⇒ 输出恒为 0 ⇒ 前向与原模型逐位相同（tests/test_model.py::test_a）。

`gc_forward` 是 `actionrope.arope.arope_forward` 普通路径（offset_px=None, mask_input=None, arm=None）的照抄，
只在 patch embedding 后多加一项 sem 条件；block / RoPE 查表 / 逐 cell 文本 / 逐 token timestep 全部直接
import `actionrope.arope` 的函数，不复制。

实现上的取舍（设计 没写死、这里定下来的约定，infer 侧必须一致 —— 它直接 import 本文件即可）
------------------------------------------------------------------------------------------
* 通道顺序：时间分组后通道 = k·4 + τ（k 类别，τ 组内帧 0..3），unshuffle 后通道 = (k·4 + τ)·16 + p·4 + q
  （p 行内偏移、q 列内偏移），即 einops "b k (f t) (h p) (w q) -> b (k t p q) f h w"。
* 第 9 类（"其它 / UNKNOWN"）的池化值 = 1 − Σ_{k<9} 池化值。池化值都是 1/16 的整数倍，fp32 下精确，
  所以与逐像素判断"不属于前 9 类"再池化逐位相同，而且省一次全分辨率比较。
* one-hot 池化不依赖参数，放在 no_grad 里算（[b,10,81,120,208] bf16 ≈ 40 MB/样本，1280×704 ≈ 2.26×），再 cast 到模块 dtype。
* 前两层 conv 用 PyTorch 默认初始化，但在 fork_rng + 固定种子下建：各 rank、各次运行初始化相同，
  不吃全局 RNG（DeepSpeed 也会从 rank 0 广播，这里是双保险，且单卡 / 多卡结果可比）。

白模条件（cond_type="whitebox"；见文件中部"白模条件"一节）
--------------------------------------------------------------------------------------
另一种控制视频：pk 渲染的白模（灰度，像素对齐 RGB，255 留作 UNKNOWN）。模块挂在**另一个名字** `dit.gc_wb_cond`
（WhiteboxCondEncoder），结构与 SemCondEncoder 同构（逐帧 4×4 块特征 → 同一套 4 帧分组 + unshuffle → 1×1 → SiLU →
1×3×3 → SiLU → 零初始化 patch-stride conv），只有输入特征不同（5 通道：RGB 块均值、亮度块内标准差、已知像素比例）。
默认一律是 "ids"：不传 cond_type、dit 上没有 gc_wb_cond 时，所有函数的行为与加白模之前逐位相同。
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file

from actionrope.arope import (
    _block_forward,
    _embed_context,
    _per_token_timestep,
    build_world_freqs,
)
from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import WanModel, modulate, rope_apply, sinusoidal_embedding_1d

UNKNOWN_ID = 255
SEM_COND_NAME = "gc_sem_cond"
# raw id → 类别下标（K=10）。0..6 原样，10 玩家 → 7，11 boss → 8，其余（含 255）→ 9
RAW_TO_INDEX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 7, 11: 8}
NUM_CLASSES = 10
OTHER_INDEX = 9

# ----------------------------------------------------------------------------
# 词表（maliang / v10）
# ----------------------------------------------------------------------------
# "ids"   = 上面的 K=10（v3/v4/v8 的 gc25d，逐位不变）。
# "parts" = v10 胶囊白模角色按部位拆开（数值代理，不过 VAE）：玩家 / Boss 各 6 个部位 raw id，K=20。
#   raw 10 / 11 = 玩家 / Boss 躯干（pelvis_spine, spine_neck, head），保持与 "ids" 相同的值；
#   raw 12..16 = 玩家 左臂 / 右臂 / 左腿 / 右腿 / 武器，raw 17..21 = Boss 同序；左右按骨骼名（l_* / r_*），不按屏幕。
#   部位由代码对 json['v9']['characters'] 的胶囊逐像素射线求交得到（capsules.py），推理时代码世界用同一函数画。
#   collapse_parts()：12..16 → 10，17..21 → 11，即退回 "ids" 词表。
PART_NAMES = ("core", "l_arm", "r_arm", "l_leg", "r_leg", "weapon")
PLAYER_PART_IDS = (10, 12, 13, 14, 15, 16)
BOSS_PART_IDS = (11, 17, 18, 19, 20, 21)
RAW_TO_INDEX_PARTS = {**{i: i for i in range(7)},
                      **{raw: 7 + j for j, raw in enumerate(PLAYER_PART_IDS)},
                      **{raw: 13 + j for j, raw in enumerate(BOSS_PART_IDS)}}
# "parts_fx" = parts + BOSS 特效落点：raw 22..25 = 巨人轻 / 巨人重 / 守卫轻 / 守卫重，K=24；parts 各类下标不变，"other" 排最后。
FX_PART_IDS = (22, 23, 24, 25)
RAW_TO_INDEX_PARTS_FX = {**RAW_TO_INDEX_PARTS, **{raw: 19 + j for j, raw in enumerate(FX_PART_IDS)}}
VOCABS = {"ids": (RAW_TO_INDEX, NUM_CLASSES), "parts": (RAW_TO_INDEX_PARTS, 20), "parts_fx": (RAW_TO_INDEX_PARTS_FX, 24)}
DEFAULT_VOCAB = "ids"
assert all(len(m) == k - 1 and sorted(m.values()) == list(range(k - 1)) for m, k in VOCABS.values())


def vocab_spec(vocab: str) -> tuple[dict, int]:
    if vocab not in VOCABS:
        raise ValueError(f"vocab 只能是 {tuple(VOCABS)}，得到 {vocab!r}")
    return VOCABS[vocab]


def valid_raw_ids(vocab: str) -> frozenset:
    """条件图里允许出现的 raw 值（含 UNKNOWN）。其它值模型会静默当成 "other"，prep / 代码世界应断言。"""
    return frozenset(vocab_spec(vocab)[0]) | {UNKNOWN_ID}


def collapse_parts(ids):
    """parts 词表的 raw id → ids 词表（12..16 → 10，17..21 → 11）；numpy / torch 都行，返回同类型新数组。"""
    lut = np.arange(256, dtype=np.uint8)
    lut[list(PLAYER_PART_IDS)] = 10
    lut[list(BOSS_PART_IDS)] = 11
    if isinstance(ids, torch.Tensor):
        return torch.from_numpy(lut).to(ids.device)[ids.long()]
    return lut[ids]


def vocab_from_in_channels(in_channels: int) -> str:
    for name, (_, k) in VOCABS.items():
        if k * T_GROUP * UNSHUFFLE * UNSHUFFLE == in_channels:
            return name
    raise ValueError(f"gc_sem_cond.proj_in 输入通道 {in_channels} 不对应任何词表 {tuple(VOCABS)}")
# 空间：池化 4× + unshuffle 4× = 16× = VAE 空间压缩；时间：VAE 4×（帧 0 单独成组，这里重复 4 次补齐）
POOL = 4
UNSHUFFLE = 4
T_GROUP = 4
HIDDEN = 512
SEM_COND_INIT_SEED = 20261010


def _lut(vocab: str = DEFAULT_VOCAB) -> torch.Tensor:
    m, k = vocab_spec(vocab)
    lut = torch.full((256,), k - 1, dtype=torch.uint8)
    for raw, i in m.items():
        lut[raw] = i
    return lut


def sem_ids_to_index(sem_ids: torch.Tensor, vocab: str = DEFAULT_VOCAB) -> torch.Tensor:
    """raw id → 类别下标（uint8，同形状）。只给测试 / 可视化用；前向不走这条（见 pooled_onehot）。"""
    ids = sem_ids.long()
    if ids.min() < 0 or ids.max() > 255:
        raise ValueError(f"sem_ids 超出 uint8 范围：[{int(ids.min())}, {int(ids.max())}]")
    return _lut(vocab).to(sem_ids.device)[ids]


@torch.no_grad()
def pooled_onehot(sem_ids: torch.Tensor, dtype=torch.float32, vocab: str = DEFAULT_VOCAB) -> torch.Tensor:
    """sem_ids [b, T, H, W]（uint8 或整数）→ 逐类 4×4 平均池化 one-hot [b, K, T, H/4, W/4]（dtype）。

    一类一类算：每次只有一张 [b·T, 1, H, W] 的 0/1 图（fp32 ≈ 128 MB/样本，用完即弃），
    从不物化 [b, K, T, H, W] 的全分辨率 one-hot。"""
    assert sem_ids.dim() == 4, f"sem_ids 应为 [b, T, H, W]，得到 {tuple(sem_ids.shape)}"
    if sem_ids.dtype.is_floating_point or sem_ids.dtype == torch.bool:
        raise TypeError(f"sem_ids 必须是整数 id（uint8 / long），得到 {sem_ids.dtype}")
    b, T, H, W = sem_ids.shape
    assert H % POOL == 0 and W % POOL == 0, (H, W)
    flat = sem_ids.reshape(b * T, 1, H, W)
    if os.environ.get("GC_POOL_MODE", "frac") == "nearest":   # paper encoder ablation: one id per 4x4 block (no fractions)
        c = POOL // 2
        flat = flat[:, :, c::POOL, c::POOL].repeat_interleave(POOL, 2).repeat_interleave(POOL, 3)
    raw_to_index, num_classes = vocab_spec(vocab)
    raw_of = {k: raw for raw, k in raw_to_index.items()}
    planes = []
    for k in range(num_classes - 1):
        m = (flat == raw_of[k]).to(torch.float32)
        planes.append(F.avg_pool2d(m, POOL).view(b, T, H // POOL, W // POOL))
        del m
    # 最后一类（"other"，ids 词表是第 9 类）= 1 − 其余之和（池化值是 1/16 的整数倍，fp32 精确，见文件头）
    other = 1.0 - torch.stack(planes, 0).sum(0)
    planes.append(other)
    out = torch.stack(planes, 1)  # [b, K, T, H/4, W/4]
    return out.to(dtype)


def group_and_unshuffle(pooled: torch.Tensor) -> torch.Tensor:
    """[b, K, T=4f−3, h4, w4] → 帧 0 重复 4 次后按 4 帧分组 → pixel-unshuffle ×4 → [b, K·4·16, f, h4/4, w4/4]。"""
    b, K, T, h4, w4 = pooled.shape
    assert (T - 1) % T_GROUP == 0, f"帧数应为 4k+1，得到 {T}"
    x = torch.cat([pooled[:, :, :1].expand(b, K, T_GROUP - 1, h4, w4), pooled], dim=2)  # [b, K, 4f, h4, w4]
    return rearrange(x, "b k (f t) (h p) (w q) -> b (k t p q) f h w", t=T_GROUP, p=UNSHUFFLE, q=UNSHUFFLE)


class SemCondEncoder(nn.Module):
    """语义 id 视频 → patch-embedding 空间的加性条件。见文件头。"""

    def __init__(self, dim: int, patch_size=(1, 2, 2), hidden: int = HIDDEN, vocab: str = DEFAULT_VOCAB):
        super().__init__()
        _, num_classes = vocab_spec(vocab)
        self.vocab = vocab
        self.num_classes = num_classes
        self.in_channels = num_classes * T_GROUP * UNSHUFFLE * UNSHUFFLE  # ids 640 / parts 1280
        self.dim = dim
        self.proj_in = nn.Conv3d(self.in_channels, hidden, kernel_size=1)
        self.conv_mid = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.proj_out = nn.Conv3d(hidden, dim, kernel_size=tuple(patch_size), stride=tuple(patch_size))
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def features(self, sem_ids: torch.Tensor) -> torch.Tensor:
        """sem_ids [b, 81, H, W] → 卷积前的输入 [b, 64K, 21, H/16, W/16]（模块 dtype；832×480：[b, 640, 21, 30, 52]）。"""
        dtype = self.proj_in.weight.dtype
        x = group_and_unshuffle(pooled_onehot(sem_ids.to(self.proj_in.weight.device), torch.float32, self.vocab))
        return x.to(dtype)

    def forward(self, sem_ids: torch.Tensor) -> torch.Tensor:
        x = self.features(sem_ids)
        assert x.shape[1] == self.in_channels, (x.shape, self.in_channels)
        x = F.silu(self.proj_in(x))
        x = F.silu(self.conv_mid(x))
        return self.proj_out(x)

    def forward_pooled(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pooled one-hot [b, K, T, H/4, W/4] (fp32) → (patch-embedding term [b, dim, f, h, w], trunk features
        [b, 512, f, 2h, 2w]). Same numbers as forward(); the trunk features feed the v2 deep adapter."""
        dtype = self.proj_in.weight.dtype
        x = group_and_unshuffle(pooled).to(dtype)
        x = F.silu(self.proj_in(x))
        h = F.silu(self.conv_mid(x))
        return self.proj_out(h), h

    def is_zero_output(self) -> bool:
        """最后一层权重与偏置全 0（刚装上 / 从无此模块的 ckpt 热启动）⇒ 输出恒为 0。"""
        return bool((self.proj_out.weight == 0).all() and (self.proj_out.bias == 0).all())


# ----------------------------------------------------------------------------
# 白模条件（cond_type="whitebox"）
# ----------------------------------------------------------------------------
# 跑酷数据的控制视频是 pk 渲染的白模：只有方块 / 圆柱 / 胶囊、固定光照（亮度 ∝ 面朝向）、黑天空、1–3 px 的
# 暗轮廓线（深度 / 法线突变处；窗框、窗台、障碍物顶面与屋顶的分界只靠它）。数据里是灰度（R=G=B），prep 存 uint8
# [81,480,832] 并把亮度截到 ≤254，255 留作 UNKNOWN（整段条件 dropout / CFG 负支 = 全 255，与语义 id 同一约定）。
#
# 编码器与 SemCondEncoder 同构，只换"逐帧 4×4 块特征"这一步（K=10 的池化 one-hot → WB_FEATURES=5 个通道）：
#   0..2 R, G, B 块均值（x/127.5 − 1）；3 亮度块内标准差（轮廓线在 4×4 均值里几乎消失，在标准差里是强信号）；
#   4 已知像素比例（全 255 ⇒ 0；没有它，"未知"会与均值 ≈ 0 的中灰立面混在一起）。
# 均值 / 标准差只在已知像素上统计。之后帧 0 重复 4 次 + 4 帧分组 + unshuffle ×4（group_and_unshuffle，对通道数通用）
# → [b, 5·4·16 = 320, 21, 30, 52] → 1×1 → SiLU → 1×3×3 → SiLU → 零初始化 Conv3d(512, dim, (1,2,2), stride (1,2,2))。
# 时间方向的核全是 1 ⇒ cell j 的条件只依赖 cell j 自己的帧（逐 latent cell 因果，可蒸馏成因果学生，
# tests/test_wb_cond.py 验证）。输入也接受 RGB uint8 [b,T,H,W,3]（三通道都是 255 才算 UNKNOWN），给以后的彩色控制视频用。
WB_COND_NAME = "gc_wb_cond"
WB_COND_INIT_SEED = 20261012
WB_FEATURES = 5
WB_FEATURE_NAMES = ("mean_r", "mean_g", "mean_b", "luma_std", "known_frac")
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
COND_TYPES = ("ids", "whitebox")
COND_MODULE = {"ids": SEM_COND_NAME, "whitebox": WB_COND_NAME}


@torch.no_grad()
def wb_pooled_features(wb: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """白模视频 uint8 [b, T, H, W]（灰度）或 [b, T, H, W, 3]（RGB）→ 逐帧 4×4 块特征 [b, 5, T, H/4, W/4]（dtype）。

    通道见上（WB_FEATURE_NAMES）。255 = UNKNOWN：灰度看该像素，RGB 看三通道是否都是 255。块内一个已知像素都没有时
    前 4 个通道为 0、第 5 个为 0。逐帧独立计算（不跨帧），fp32。"""
    if wb.dtype != torch.uint8:
        raise TypeError(f"白模必须是 uint8 像素值（255 = UNKNOWN），得到 {wb.dtype}")
    if wb.dim() == 4:
        rgb = False
    elif wb.dim() == 5 and wb.shape[-1] == 3:
        rgb = True
    else:
        raise AssertionError(f"白模应为 [b, T, H, W] 或 [b, T, H, W, 3]，得到 {tuple(wb.shape)}")
    b, T, H, W = wb.shape[:4]
    assert H % POOL == 0 and W % POOL == 0, (H, W)
    h, w = H // POOL, W // POOL
    if rgb:
        x = wb.reshape(b * T, H, W, 3).permute(0, 3, 1, 2)                     # [bT, 3, H, W]
        known = (x != UNKNOWN_ID).all(1, keepdim=True)
        v = x.float().div_(127.5).sub_(1.0).mul_(known)
        luma = sum(c * v[:, i:i + 1] for i, c in enumerate(LUMA_WEIGHTS))
    else:
        x = wb.reshape(b * T, 1, H, W)
        known = x != UNKNOWN_ID
        v = x.float().div_(127.5).sub_(1.0).mul_(known)
        luma = v
    kf = F.avg_pool2d(known.float(), POOL)                                     # [bT, 1, h, w]，1/16 的整数倍
    denom = kf.clamp_min(1.0 / (POOL * POOL))
    mean = F.avg_pool2d(v, POOL) / denom                                       # [bT, 1 或 3, h, w]
    lmean = mean if not rgb else F.avg_pool2d(luma, POOL) / denom
    dev = (luma - lmean.repeat_interleave(POOL, 2).repeat_interleave(POOL, 3)).mul_(known)
    std = (F.avg_pool2d(dev.square_(), POOL) / denom).sqrt_()                  # 两遍法：常数块的 std 恰为 0
    if not rgb:
        mean = mean.expand(-1, 3, -1, -1)
    out = torch.cat([mean, std, kf], 1)                                        # [bT, 5, h, w]
    return out.view(b, T, WB_FEATURES, h, w).transpose(1, 2).to(dtype)


class WhiteboxCondEncoder(nn.Module):
    """白模视频 → patch-embedding 空间的加性条件（与 SemCondEncoder 同构、同层名，只换输入特征，见上）。"""

    def __init__(self, dim: int, patch_size=(1, 2, 2), hidden: int = HIDDEN):
        super().__init__()
        self.n_features = WB_FEATURES
        self.in_channels = WB_FEATURES * T_GROUP * UNSHUFFLE * UNSHUFFLE  # 320
        self.dim = dim
        self.proj_in = nn.Conv3d(self.in_channels, hidden, kernel_size=1)
        self.conv_mid = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.proj_out = nn.Conv3d(hidden, dim, kernel_size=tuple(patch_size), stride=tuple(patch_size))
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def features(self, wb: torch.Tensor) -> torch.Tensor:
        """白模 [b, 81, H, W(, 3)] uint8 → 卷积前的输入 [b, 320, 21, H/16, W/16]（模块 dtype）。"""
        x = group_and_unshuffle(wb_pooled_features(wb.to(self.proj_in.weight.device), torch.float32))
        return x.to(self.proj_in.weight.dtype)

    def forward(self, wb: torch.Tensor) -> torch.Tensor:
        x = self.features(wb)
        assert x.shape[1] == self.in_channels, (x.shape, self.in_channels)
        x = F.silu(self.proj_in(x))
        x = F.silu(self.conv_mid(x))
        return self.proj_out(x)

    def forward_pooled(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wb_pooled_features 的输出 [b, 5, T, H/4, W/4] → (patch-embedding 项, trunk)。与 forward() 同一组数。"""
        x = group_and_unshuffle(pooled).to(self.proj_in.weight.dtype)
        x = F.silu(self.proj_in(x))
        h = F.silu(self.conv_mid(x))
        return self.proj_out(h), h

    def is_zero_output(self) -> bool:
        return bool((self.proj_out.weight == 0).all() and (self.proj_out.bias == 0).all())


# ----------------------------------------------------------------------------
# v2 architecture : deep semantic injection + semantic attention bias. Both zero-initialised ⇒ a model with
# them freshly installed computes exactly what v1 computes (warm start from any v1 export).
# ----------------------------------------------------------------------------
ADAPTER_NAME = "adapter"            # dit.gc_sem_cond.adapter
ATTN_GAIN_NAME = "attn_qk"          # dit.gc_sem_cond.attn_qk  [n_blocks, 2, K, dim]
ADAPTER_HIDDEN = 1024


class SemAdapter(nn.Module):
    """Deep injection (VACE / ControlNet-style, but light): the encoder's trunk features are patchified to the token grid,
    mixed by a per-cell 3×3 conv, and added to the residual stream before each block in `layers` through a per-layer
    zero-init Linear. In v1 the ids enter only once at the patch embedding (sem term ≈ 12 % of its RMS) and the
    cond / no-cond val gap stays tiny, i.e. the network barely relies on the map deeper in."""

    def __init__(self, dim: int, layers, hidden_in: int = HIDDEN, hidden: int = ADAPTER_HIDDEN):
        super().__init__()
        self.patch = nn.Conv3d(hidden_in, hidden, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # spatial only: every latent cell sees only its own ids ⇒ the condition stays causal / chunk-local, so the model
        # can be distilled to an autoregressive (chunked, KV-cached) student where future ids are not known yet
        self.mix = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.inj = nn.ModuleDict({str(int(i)): nn.Linear(hidden, dim) for i in layers})
        for lin in self.inj.values():
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    @property
    def layers(self) -> list[int]:
        return sorted(int(k) for k in self.inj.keys())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """trunk [b, 512, f, H/16, W/16] → token features [b, f·(H/32)·(W/32), hidden]（832×480：f·15·26）."""
        t = F.silu(self.patch(h))
        t = F.silu(self.mix(t))
        t = rearrange(t, "b c f h w -> b (f h w) c")
        return F.layer_norm(t, (t.shape[-1],))    # unit-RMS tokens: the zero-init injections become effective quickly

    def is_zero_output(self) -> bool:
        return all(bool((l.weight == 0).all() and (l.bias == 0).all()) for l in self.inj.values())


def token_class_dist(pooled: torch.Tensor) -> torch.Tensor:
    """pooled one-hot [b, K, 4f−3, H/4, W/4] → per-token class fractions [b, f·(H/32)·(W/32), K] (fp32), the
    'other / UNKNOWN' class zeroed (unknown tokens neither give nor get the same-class bonus)."""
    b, K, T, h4, w4 = pooled.shape
    x = torch.cat([pooled[:, :, :1].expand(b, K, T_GROUP - 1, h4, w4), pooled], dim=2)       # [b, K, 4f, h4, w4]
    f = x.shape[2] // T_GROUP
    x = x.view(b, K, f, T_GROUP, h4, w4).mean(3)                                            # [b, K, f, h4, w4]
    tok = UNSHUFFLE * 2                       # a DiT token = 2×2 latent px = 8×8 pooled px = 32×32 image px
    x = F.avg_pool2d(x.reshape(b, K * f, h4, w4), tok)
    x = x.view(b, K, f, h4 // tok, w4 // tok).clone()
    x[:, K - 1] = 0
    return rearrange(x, "b k f h w -> b (f h w) k")


def _self_attn(sa, x, freqs, p_tok=None, e_qk=None):
    """WanSelfAttention.forward; with (p_tok [b, L, K], e_qk [2, K, dim]) learned class embeddings are added to q and k
    after RoPE (q += p·E_q, k += p·E_k). Δq·Δk is a learnable class-pair attention bias (per head a rank-≤128 K×K
    matrix), q·Δk lets content look for a class. Head dim stays 128 ⇒ same SDPA kernel as v1; E = 0 ⇒ bit-identical."""
    if e_qk is None:
        return sa(x, freqs)
    n = sa.num_heads
    q = rope_apply(sa.norm_q(sa.q(x)), freqs, n)
    k = rope_apply(sa.norm_k(sa.k(x)), freqs, n)
    p = p_tok.to(q.dtype)
    q = q + p @ e_qk[0].to(q.dtype)
    k = k + p @ e_qk[1].to(k.dtype)
    return sa.o(sa.attn(q, k, sa.v(x)))


def gc_block_forward(block, x, context, t_mod, freqs, p_tok=None, e_qk=None, *, f: int):
    """actionrope.arope._block_forward with arm=None (DiTBlock.forward + per-cell text cross-attn), self-attention
    through _self_attn. e_qk=None ⇒ identical to _block_forward."""
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
    x = block.gate(x, gate_msa, _self_attn(block.self_attn, input_x, freqs, p_tok, e_qk))
    q_in = block.norm3(x)
    if context.shape[0] == x.shape[0]:
        x = x + block.cross_attn(q_in, context)
    else:
        b = x.shape[0]
        assert context.shape[0] == b * f, f"逐 cell context 应有 b*f={b * f} 行，得到 {context.shape[0]}"
        q_in = rearrange(q_in, "b (f n) c -> (b f) n c", f=f)
        y = block.cross_attn(q_in, context)
        x = x + rearrange(y, "(b f) n c -> b (f n) c", f=f)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(input_x))
    return x


def install_v2(dit: WanModel, adapter_every: int = 0, sem_attn: bool = False) -> WanModel:
    """Add the zero-init v2 modules under dit.gc_sem_cond (idempotent). adapter_every=k ⇒ inject before blocks 0, k, 2k…"""
    if not hasattr(dit, SEM_COND_NAME):
        install_sem_cond(dit)
    enc = getattr(dit, SEM_COND_NAME)
    ref = dit.patch_embedding.weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEM_COND_INIT_SEED + 1)
        if adapter_every and not hasattr(enc, ADAPTER_NAME):
            ad = SemAdapter(dit.dim, range(0, len(dit.blocks), int(adapter_every)))
            enc.add_module(ADAPTER_NAME, ad.to(device=ref.device, dtype=ref.dtype))
    if sem_attn and not hasattr(enc, ATTN_GAIN_NAME):
        enc.register_parameter(ATTN_GAIN_NAME, nn.Parameter(
            torch.zeros(len(dit.blocks), 2, enc.num_classes, dit.dim, device=ref.device, dtype=ref.dtype)))
    return dit


def v2_config_from_keys(keys) -> dict:
    """Which v2 modules a checkpoint carries (from its key names)."""
    pre = SEM_COND_NAME + "."
    inj = sorted({int(k.split(".")[3]) for k in keys if k.startswith(f"{pre}{ADAPTER_NAME}.inj.")})
    every = (inj[1] - inj[0]) if len(inj) > 1 else (1 if inj else 0)
    return {"adapter_every": every, "sem_attn": f"{pre}{ATTN_GAIN_NAME}" in set(keys)}


# ----------------------------------------------------------------------------
# 加载 / 安装
# ----------------------------------------------------------------------------

def load_dit(model_dir: str, device: str = "cpu") -> WanModel:
    """只加载 DiT（bf16），与 actionrope/train.py::load_dit 同一条路径（逐行照抄，免得 import actionrope.train 拖进
    baseline / dataset；maliang 不带 actionrope/train.py）。"""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    files = [os.path.join(model_dir, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device, model_configs=[ModelConfig(path=files)],
        tokenizer_config=None, redirect_common_files=False,
    )
    return pipe.dit


def install_sem_cond(dit: WanModel, vocab: str = DEFAULT_VOCAB) -> WanModel:
    """给 DiT 装零初始化的 `dit.gc_sem_cond`（幂等：已装则原样返回，不重置权重；已装的词表与 vocab 不同则报错）。"""
    if hasattr(dit, SEM_COND_NAME):
        enc = getattr(dit, SEM_COND_NAME)
        assert isinstance(enc, SemCondEncoder), f"dit.{SEM_COND_NAME} 不是 SemCondEncoder：{type(enc)}"
        if enc.vocab != vocab:
            raise ValueError(f"dit.{SEM_COND_NAME} 已按词表 {enc.vocab!r} 安装，要求的是 {vocab!r}")
        return dit
    ref = dit.patch_embedding.weight
    # 固定种子建模块：不消耗、也不依赖全局 RNG（各 rank 一致，见文件头）
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEM_COND_INIT_SEED)
        enc = SemCondEncoder(dit.dim, patch_size=tuple(dit.patch_size), vocab=vocab)
    enc.to(device=ref.device, dtype=ref.dtype)
    dit.add_module(SEM_COND_NAME, enc)
    return dit


def install_wb_cond(dit: WanModel) -> WanModel:
    """给 DiT 装零初始化的 `dit.gc_wb_cond`（白模条件；幂等）。与 gc_sem_cond 互斥（双条件以后再说）。"""
    if hasattr(dit, SEM_COND_NAME):
        raise ValueError(f"dit 上已有 {SEM_COND_NAME}（语义 id 条件）；白模模型只装 {WB_COND_NAME}")
    if hasattr(dit, WB_COND_NAME):
        enc = getattr(dit, WB_COND_NAME)
        assert isinstance(enc, WhiteboxCondEncoder), f"dit.{WB_COND_NAME} 不是 WhiteboxCondEncoder：{type(enc)}"
        return dit
    ref = dit.patch_embedding.weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(WB_COND_INIT_SEED)
        enc = WhiteboxCondEncoder(dit.dim, patch_size=tuple(dit.patch_size))
    enc.to(device=ref.device, dtype=ref.dtype)
    dit.add_module(WB_COND_NAME, enc)
    return dit


def install_cond(dit: WanModel, cond_type: str = "ids") -> WanModel:
    """按 cond_type 装条件模块：ids ⇒ install_sem_cond，whitebox ⇒ install_wb_cond。"""
    if cond_type not in COND_TYPES:
        raise ValueError(f"cond_type 只能是 {COND_TYPES}，得到 {cond_type!r}")
    return install_wb_cond(dit) if cond_type == "whitebox" else install_sem_cond(dit)


def resolve_cond_type(dit, cond_type: str | None = None) -> str:
    """显式给了就用（并核对合法）；None ⇒ 看 dit 上装了哪个：只有 gc_wb_cond ⇒ "whitebox"，否则 "ids"（旧行为）。"""
    if cond_type is not None:
        if cond_type not in COND_TYPES:
            raise ValueError(f"cond_type 只能是 {COND_TYPES}，得到 {cond_type!r}")
        return cond_type
    has_sem, has_wb = hasattr(dit, SEM_COND_NAME), hasattr(dit, WB_COND_NAME)
    if has_sem and has_wb:
        raise ValueError(f"dit 上同时有 {SEM_COND_NAME} 和 {WB_COND_NAME}，请显式传 cond_type")
    return "whitebox" if has_wb else "ids"


def sem_vocab(dit) -> str:
    """dit 上已装的 gc_sem_cond 的词表。"""
    return getattr(dit, SEM_COND_NAME).vocab


def vocab_from_state(sd_or_header) -> str | None:
    """state dict / safetensors 头（键 → 张量或 {'shape': ...}）→ gc_sem_cond 的词表；没有 gc_sem_cond 键 ⇒ None。"""
    for k in (f"{SEM_COND_NAME}.proj_in.weight", f"dit.{SEM_COND_NAME}.proj_in.weight"):
        if k in sd_or_header:
            v = sd_or_header[k]
            shape = v["shape"] if isinstance(v, dict) else tuple(v.shape)
            return vocab_from_in_channels(int(shape[1]))
    return None


def cond_type_from_keys(keys) -> str:
    """ckpt 的键名 → 它是哪种条件的导出（有 gc_wb_cond.* / dit.gc_wb_cond.* ⇒ whitebox，否则 ids）。"""
    pre = (WB_COND_NAME + ".", "dit." + WB_COND_NAME + ".")
    return "whitebox" if any(k.startswith(pre) for k in keys) else "ids"


def _read_safetensors(path) -> tuple[dict, dict]:
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    sd, meta = {}, {}
    for p in paths:
        part = load_file(str(p))
        dup = set(part) & set(sd)
        if dup:
            raise RuntimeError(f"{p} 与前面的分片有重复键：{sorted(dup)[:5]}")
        sd.update(part)
        with safe_open(str(p), framework="pt") as fh:
            meta.update(fh.metadata() or {})
    return sd, meta


def load_gc_weights(dit: WanModel, path, cond_type: str = "ids") -> dict:
    """从导出的 safetensors（或分片列表）加载 DiT 权重。先 install_sem_cond（幂等），再按 ckpt 里出现的键装上 v2 模块
    （gc_sem_cond.adapter.* / gc_sem_cond.attn_gain）。

    严格匹配 dit.state_dict() 的键集；放行的缺失只有两类，且对应模块必须仍是零输出（刚装上）：
    * 全部 `gc_sem_cond.*`（从普通 Wan / ARoPE plain 导出热启动）；
    * v2 模块（调用方先 install_v2 再从 v1 导出热启动）。
    返回 info：missing / n_tensors / has_sem_cond / v2 / metadata。
    cond_type="whitebox" ⇒ 走 _load_wb_weights（gc_wb_cond，见那里）；默认 "ids" 的行为与原来完全相同。"""
    if cond_type != "ids":
        if cond_type != "whitebox":
            raise ValueError(f"cond_type 只能是 {COND_TYPES}，得到 {cond_type!r}")
        return _load_wb_weights(dit, path)
    sd, meta = _read_safetensors(path)
    # 词表：ckpt 里有 gc_sem_cond ⇒ 按它的输入通道定；没有（Wan / plain 热启动）⇒ 用 dit 上已装的，未装则默认 ids
    ck_vocab = vocab_from_state(sd)
    have = sem_vocab(dit) if hasattr(dit, SEM_COND_NAME) else None
    if ck_vocab and have and ck_vocab != have:
        raise RuntimeError(f"{path} 的 gc_sem_cond 是 {ck_vocab!r} 词表，dit 上装的是 {have!r}")
    install_sem_cond(dit, ck_vocab or have or DEFAULT_VOCAB)
    v2 = v2_config_from_keys(sd.keys())
    install_v2(dit, **v2)
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    prefix = SEM_COND_NAME + "."
    v2_prefixes = (f"{prefix}{ADAPTER_NAME}.", f"{prefix}{ATTN_GAIN_NAME}")
    bad_missing = [k for k in missing if not k.startswith(prefix)]
    if bad_missing or unexpected:
        raise RuntimeError(f"{path} 与模型键集不符：缺 {bad_missing[:5]}…({len(bad_missing)})，"
                           f"多 {list(unexpected)[:5]}…({len(unexpected)})")
    enc = getattr(dit, SEM_COND_NAME)
    sem_missing = [k for k in missing if k.startswith(prefix) and not k.startswith(v2_prefixes)]
    v2_missing = [k for k in missing if k.startswith(v2_prefixes)]
    n_sem_total = sum(1 for k in dit.state_dict() if k.startswith(prefix) and not k.startswith(v2_prefixes))
    if sem_missing and len(sem_missing) != n_sem_total:
        raise RuntimeError(f"{path} 只含部分 {prefix}* 键（缺 {sem_missing}），ckpt 不完整")
    if sem_missing and not enc.is_zero_output():
        raise RuntimeError(f"{path} 没有 {prefix}* 键，但 dit 上的 {SEM_COND_NAME} 已不是零初始化状态；"
                           f"请在全新 install 的模型上热启动")
    if v2_missing:
        ad, g = getattr(enc, ADAPTER_NAME, None), getattr(enc, ATTN_GAIN_NAME, None)
        if (ad is not None and not ad.is_zero_output()) or (g is not None and bool((g != 0).any())):
            raise RuntimeError(f"{path} 缺 v2 键 {v2_missing[:3]}…，但 v2 模块已不是零初始化状态")
    return {"missing": list(missing), "n_tensors": len(sd), "has_sem_cond": not sem_missing, "v2": v2, "vocab": enc.vocab,
            "v2_missing": len(v2_missing), "metadata": meta, "path": str(path)}


def _load_wb_weights(dit: WanModel, path) -> dict:
    """白模模型（dit.gc_wb_cond）的加载：install_wb_cond（幂等）后严格匹配键集。

    * ckpt 里的 `gc_sem_cond.*`（含 v2 子模块）一律丢弃并记进 info["dropped_sem_keys"] —— 从语义 id 模型
      （如 gc25d_v3v4/step-6000，825 个 Wan 键 + 6 个 gc_sem_cond 键）热启动时只取 825 个 DiT 键；
    * 唯一放行的缺失：6 个 `gc_wb_cond.*` 全缺（热启动），且模块仍是零输出（刚装上）；只缺一部分 ⇒ 报错；
    * 缺任何 DiT 键 / 有其它多余键 ⇒ 报错；同时带 gc_sem_cond.* 与 gc_wb_cond.* 的 ckpt ⇒ 报错。"""
    install_wb_cond(dit)
    sd, meta = _read_safetensors(path)
    n_file = len(sd)
    sem_prefix, prefix = SEM_COND_NAME + ".", WB_COND_NAME + "."
    dropped = sorted(k for k in sd if k.startswith(sem_prefix))
    has_wb = any(k.startswith(prefix) for k in sd)
    if dropped and has_wb:
        raise RuntimeError(f"{path} 同时有 {sem_prefix}* 和 {prefix}* 键，不知道该用哪个")
    for k in dropped:
        del sd[k]
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith(prefix)]
    if bad_missing or unexpected:
        raise RuntimeError(f"{path} 与白模模型键集不符：缺 {bad_missing[:5]}…({len(bad_missing)})，"
                           f"多 {list(unexpected)[:5]}…({len(unexpected)})")
    enc = getattr(dit, WB_COND_NAME)
    wb_missing = [k for k in missing if k.startswith(prefix)]
    n_wb_total = sum(1 for k in dit.state_dict() if k.startswith(prefix))
    if wb_missing and len(wb_missing) != n_wb_total:
        raise RuntimeError(f"{path} 只含部分 {prefix}* 键（缺 {wb_missing}），ckpt 不完整")
    if wb_missing and not enc.is_zero_output():
        raise RuntimeError(f"{path} 没有 {prefix}* 键，但 dit 上的 {WB_COND_NAME} 已不是零初始化状态；"
                           f"请在全新 install 的模型上热启动")
    return {"missing": list(missing), "n_tensors": n_file, "n_loaded": len(sd), "cond_type": "whitebox",
            "has_wb_cond": not wb_missing, "has_sem_cond": False, "dropped_sem_keys": dropped,
            "v2": {"adapter_every": 0, "sem_attn": False}, "v2_missing": 0, "metadata": meta, "path": str(path)}


# ----------------------------------------------------------------------------
# 前向
# ----------------------------------------------------------------------------

def gc_forward(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    sem_ids,
    *,
    first_frame_cond: bool = True,
    context_ids=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    aux: dict | None = None,
    cond_type: str | None = None,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """gc25d 的 DiT 前向。latents [b, 48, f, H/16, W/16] → v_pred 同形（832×480：[b, 48, f, 30, 52]）。

    cond_type："ids"（gc_sem_cond，语义 id）/ "whitebox"（gc_wb_cond，sem_ids 参数里放白模 uint8 [b, 4f−3, H, W(, 3)]）；
              None ⇒ resolve_cond_type（只装了 gc_wb_cond ⇒ whitebox，否则 ids —— 不传时与原来逐位相同）。

    = arope_forward(offset_px=None, mask_input=None, arm=None) 的逐行照抄 + patch embedding 后的 sem 条件项。
    sem_ids:  [b, 4f−3, 16·lat_h, 16·lat_w] uint8 / 整数 raw id（255 = UNKNOWN）；None ⇒ 不加 sem 项（≡ 原模型）。
              给了但 dit 没装 gc_sem_cond ⇒ 报错（避免"以为用了条件其实没用"）。
    context:  [b, 512, 4096]（整条）或 [b, f, 512, 4096]（逐 cell）；context_ids [b, f] 用于逐 cell 去重。
    timestep: 标量 / [1] / [b]（1000 制）；first_frame_cond ⇒ 帧 0 的 t=0。
    aux:      给一个 dict ⇒ 填入诊断量（sem 项与 patch embedding 的 RMS），不影响计算。
    """
    b, _, f, lat_h, lat_w = latents.shape
    p_f, p_h, p_w = dit.patch_size
    n_tok_per_frame = (lat_h // p_h) * (lat_w // p_w)

    # 时间步：逐 token，帧 0 为 0（照抄 arope_forward / model_fn_wan_video）
    t_tok = _per_token_timestep(latents, timestep, n_tok_per_frame, first_frame_cond)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_tok.reshape(-1)).view(b, t_tok.shape[1], -1))
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

    # 文本（逐 cell 时只对唯一串算一次 text_embedding）
    context = _embed_context(dit, context, context_ids)

    # patch embedding + 语义条件
    x = dit.patch_embedding(latents)
    adapter = attn_gain = tok = p_tok = None
    if sem_ids is not None and resolve_cond_type(dit, cond_type) == "whitebox":
        enc = getattr(dit, WB_COND_NAME, None)
        if enc is None:
            raise ValueError(f"cond_type=whitebox 但 dit 没有 {WB_COND_NAME}，请先 install_wb_cond(dit) / "
                             f"load_gc_weights(..., cond_type='whitebox')")
        exp = (b, T_GROUP * (f - 1) + 1, lat_h * POOL * UNSHUFFLE, lat_w * POOL * UNSHUFFLE)
        assert tuple(sem_ids.shape[:4]) == exp and sem_ids.dim() in (4, 5), \
            f"白模形状应为 {exp}（灰度）或 {exp + (3,)}（RGB），得到 {tuple(sem_ids.shape)}"
        s, _ = enc.forward_pooled(wb_pooled_features(sem_ids.to(enc.proj_in.weight.device), torch.float32))
        assert s.shape == x.shape, f"白模项 {tuple(s.shape)} ≠ patch embedding {tuple(x.shape)}"
        if aux is not None:
            aux["sem_rms"] = s.detach().float().pow(2).mean().sqrt()
            aux["x_rms"] = x.detach().float().pow(2).mean().sqrt()
        x = x + s.to(x.dtype)
    elif sem_ids is not None:
        enc = getattr(dit, SEM_COND_NAME, None)
        if enc is None:
            raise ValueError(f"给了 sem_ids 但 dit 没有 {SEM_COND_NAME}，请先 install_sem_cond(dit) / load_gc_weights")
        exp = (b, T_GROUP * (f - 1) + 1, lat_h * POOL * UNSHUFFLE, lat_w * POOL * UNSHUFFLE)
        assert tuple(sem_ids.shape) == exp, f"sem_ids 形状应为 {exp}，得到 {tuple(sem_ids.shape)}"
        adapter = getattr(enc, ADAPTER_NAME, None)
        attn_gain = getattr(enc, ATTN_GAIN_NAME, None)
        pooled = pooled_onehot(sem_ids.to(enc.proj_in.weight.device), torch.float32, enc.vocab)
        s, trunk = enc.forward_pooled(pooled)
        assert s.shape == x.shape, f"sem 项 {tuple(s.shape)} ≠ patch embedding {tuple(x.shape)}"
        if aux is not None:
            aux["sem_rms"] = s.detach().float().pow(2).mean().sqrt()
            aux["x_rms"] = x.detach().float().pow(2).mean().sqrt()
        x = x + s.to(x.dtype)
        if adapter is not None:
            tok = adapter(trunk)                                                  # [b, L, 1024]
        if attn_gain is not None:
            p_tok = token_class_dist(pooled)                                      # [b, L, K] fp32
        del pooled
    if extra is not None:                       # paper ablation (color_vae): precomputed condition term
        assert extra.shape == x.shape, (tuple(extra.shape), tuple(x.shape))
        if aux is not None:
            aux["sem_rms"] = extra.detach().float().pow(2).mean().sqrt()
            aux["x_rms"] = x.detach().float().pow(2).mean().sqrt()
        x = x + extra.to(x.dtype)
    f, h, w = x.shape[2:]
    assert f * h * w == t_tok.shape[1]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    # 普通 RoPE 查表（与 model_fn_wan_video 逐位相同）
    freqs = build_world_freqs(dit, f, h, w, None, device=x.device)

    # blocks：v1 modules only ⇒ actionrope._block_forward (bit-identical to v1); v2 ⇒ gc_block_forward + injections
    inj_rms = []
    for i, block in enumerate(dit.blocks):
        if adapter is not None and str(i) in adapter.inj:
            r = adapter.inj[str(i)](tok)
            if aux is not None:
                inj_rms.append(r.detach().float().pow(2).mean())
            x = x + r.to(x.dtype)
        if attn_gain is not None:
            x = gradient_checkpoint_forward(
                functools.partial(gc_block_forward, block, f=f),
                use_gradient_checkpointing, use_gradient_checkpointing_offload,
                x, context, t_mod, freqs, p_tok, attn_gain[i],
            )
        else:
            x = gradient_checkpoint_forward(
                functools.partial(_block_forward, block, f=f, arm=None, idx=i, grid=None),
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs, None,
            )
    if aux is not None and inj_rms:
        aux["inj_rms"] = torch.stack(inj_rms).mean().sqrt()
    if aux is not None and attn_gain is not None:
        aux["attn_bias_absmax"] = attn_gain.detach().float().pow(2).mean().sqrt()   # RMS of the class embeddings

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))
