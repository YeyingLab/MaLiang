"""蒸馏用的 batch 工具：mask / 权重图 / 偏移与文本的延长和开窗。

数据集本身**一行不改**：`actionrope.dataset.AropeLatentDataset` + `arope_collate` 已经给齐
了三个阶段需要的一切（`input_latents [b,48,21,30,52] bf16`、`context [b,21,512,4096] bf16`、
`context_ids [b,21] i64`、`offset_px [b,21,2] f32`、`offset_tok`、`delta_tok`、`action_idx`、
`clip_id`、`kind`）。本文件只做 batch 之上的纯函数，不碰 IO、不碰随机数据增强。

四类工具
--------
1. `to_device` —— 与 `actionrope/train.py:344` 同一份语义（张量搬走，字符串 list 原样）。
2. `make_mask_input` / `make_weight_map` —— stage1/2 的 `mask_input` 与 loss 权重图。
   `mask_input` 直接用 `known_mask(offset_px / 16, 30, 52)`，**不能**写成
   `loss_weight_map(...).eq(1)`：那样一来模型的输入通道就被 loss 配方绑架了——
   `new_weight=1.0`（想关掉 new 区加权时的自然写法）会让权重图恒等于 1，掩码整幅变成
   "全 known"，模型输入被悄悄改掉而不报错；`sigma`/`ramp_latent` 也各自能把这条等价关系
   推到 float32 的舍入边缘（dist 模式下 `d == 0 ⟺ known` 只是巧合，紧贴边界的
   `1 + 5e-10` 在 float32 里就是 1.0）。掩码是**模型输入**，权重图是**loss 配方**，
   两件事必须各自从 offset 现算。
3. `window_offsets` / `window_context` —— stage3 滑窗 KV 的**锚定切片**。绝对 cell k、
   窗口最老保留 cell a 时，窗口内的世界坐标一律减去 `offset_px[a]`，于是整段窗口在 RoPE
   看来就是一条以自身 cell 0 为原点的 21-cell 训练 clip（RoPE 是相对编码，整体平移不变）。
   两者索引口径完全一致：`[:, a:k+1]`。
4. `extend_offsets` / `extend_context` —— rollout 超过 21 cell 时的延长。
   `--rollout_max 26` 意味着 cell 21..26 在数据里根本不存在：`parse_actions` 硬要求 21 个
   cell（`infer.py:127`），`AropeLatentDataset.__getitem__` 硬要求 21 条 prompt。两条流必须
   **同时**延长而且逐下标对齐，否则要么在 `context.shape[1] == 1+N` 的断言处崩，要么更糟：
   几何往前走、文本停在 cell 20，rollout 照跑、loss 照降，但训的是错的东西。

   两条流的延长策略**故意不同**，这不是笔误：
   * 文本可以"保持"最后一条（`_wm/an earlier internal project/distill/code/stage3_dmd/dmd_gm.py:628-637`
     的 `last = context[:, Fl-1:Fl]; pad = last.expand(...)`）。ARoPE 臂的 prompt 经
     `prompts.strip_action` 剥掉了动作从句，只剩场景描述，而 rollout 不换场景，所以
     "保持最后一条" 在这里是**语义正确**的，不是凑合。
   * 偏移不能"保持"。偏移保持不动 = "玩家站住了"，那是**另一个动作**，不是中性动作
     （V Rising 的 null action 陷阱，GraphGWM/README 记过一次）。所以偏移必须继续积分一个
     速度：`continue`（沿用最后一段速度）或 `resample`（默认，每 5 个 cell 换一次方向，
     方向取 `geometry.ACTION_DIRS` 9 类、速度取 sidecar 实测的 `walk_speed_px_s`）。

速度 → 逐 cell 偏移的换算（不是拍脑袋的 v/4）
------------------------------------------
Wan2.2 因果 VAE 的分组是 cell 0 = 帧 0、cell k = 帧 4k−3…4k（`geometry.cell_frames`），
逐 cell 偏移是 cell 内各帧偏移的**均值**（`geometry.frames_to_cells`），而逐帧偏移按
`frame_off[i] = frame_off[i−1] + v[cell(i)] / FPS` 积分（`infer.py::parse_actions`）。
把这两步合起来，逐 cell 增量是（v 单位 px/s，FPS=16）：

    delta[1]   = 2.5 · v[1] / FPS                      （cell 0 只有帧 0，起步项不同）
    delta[k]   = (1.5 · v[k−1] + 2.5 · v[k]) / FPS     （k ≥ 2）

匀速时退化成 4·v/FPS = v/4 px per cell（一个 cell = 4 帧 = 0.25 s）。换方向的那个 cell 上
1.5/2.5 的加权才是真值，直接写 v/4 会在每个 chunk 边界差 1.5·|Δv|/16（最大 ~13 px ≈ 0.8 个
latent 格）。`test_data.py::test_extend_matches_frame_integration` 用帧级积分逐位核对了这条。
"""
from __future__ import annotations

import math

import numpy as np
import torch

from actionrope.arope import known_mask, loss_weight_map
from actionrope.geometry import ACTION_DIRS, CELLS, FPS, LAT_H, LAT_W, PX_PER_LATENT

# 一个 cell 覆盖 4 帧（cell 0 除外），见 geometry.cell_frames
CELL_FRAMES = 4
# 帧级积分 + cell 内均值合出来的两个权重（推导见模块 docstring）
_W_PREV, _W_CUR = 1.5, 2.5

# sidecar 实测步速（data/latent/arope_sidecar.pt 的 meta["walk_speed_px_s"]，已乘 |gain|）。
# 横纵不同是透视造成的，不是噪声；`infer.py::parse_actions` 也是按轴分别乘这个数。
# 只作为默认值：真跑的时候优先用 walk_speed_from_sidecar() 从当前 sidecar 现读。
WALK_SPEED_PX_S = (69.24106392992304, 46.012630371093756)

# rollout 延长的默认档位，与 stage3 的 --offset_chunk_cells 5 对齐
DEFAULT_CHUNK_CELLS = 5
DEFAULT_SPEED_MULT = (0.8, 1.2)


# --------------------------------------------------------------------------
# batch 搬运
# --------------------------------------------------------------------------

def to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    """张量搬到 device，其余（clip_id / kind 这种 list[str]）原样带过去。

    与 `actionrope/train.py:344-345` 同一份实现，蒸馏侧不再各写一遍。
    """
    return {
        k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


# --------------------------------------------------------------------------
# mask 通道 / loss 权重图
# --------------------------------------------------------------------------

def make_mask_input(offset_px: torch.Tensor, lat_h: int = LAT_H, lat_w: int = LAT_W,
                    angle=None) -> torch.Tensor:
    """known/new 掩码通道 → float32 [b, 1, f, lat_h, lat_w]，喂给 `dit.arope_mask_embedding`。

    offset_px [b, f, 2] px。除以 PX_PER_LATENT=16 换成 latent 格单位，与
    `actionrope/train.py:243-245` 逐位一致（known=1、new=0）。
    stage3 传的是**锚定后**的窗口偏移（见 window_offsets），于是 "known" 的含义变成
    "落在窗口最老那一 cell 的足迹内"——正是模型训练时见过的那种。
    """
    off = _as_bf2(offset_px, "offset_px")
    if angle is not None and bool(torch.any(torch.as_tensor(angle) != 0)):
        known = _known_rotated(off / PX_PER_LATENT, angle, lat_h, lat_w)
    else:
        known = known_mask(off / PX_PER_LATENT, lat_h, lat_w)  # bool [b, f, h, w]
    return known.to(torch.float32).unsqueeze(1)


def _known_rotated(off: torch.Tensor, angle, lat_h: int, lat_w: int) -> torch.Tensor:
    """相机在锚点与当前 cell 之间转了 angle 弧度时的 known 掩码（angle = 锚点朝向 − 当前朝向）。

    当前格点 p 先按原口径加上偏移，再绕画幅中心转 −angle 回到**锚点相机系**，落在锚点足迹
    [−0.5, h−0.5] × [−0.5, w−0.5] 内 ⇒ known。angle=0 时逐位退化成 `known_mask`（旋转成恒等）。
    旋转之后 known 区是一块**斜的**多边形 —— 训练里只出现过轴对齐的边界，这一条是已知的分布外。
    """
    from distill.arope_dit import rot2d
    b, f = off.shape[:2]
    dev, dt = off.device, torch.float64
    ang = torch.as_tensor(angle, device=dev, dtype=dt).view(b, f, 1, 1)
    gi = torch.arange(lat_h, device=dev, dtype=dt).view(1, 1, lat_h, 1).expand(b, f, lat_h, lat_w)
    gj = torch.arange(lat_w, device=dev, dtype=dt).view(1, 1, 1, lat_w).expand(b, f, lat_h, lat_w)
    ctr_h, ctr_w = (lat_h - 1) / 2.0, (lat_w - 1) / 2.0
    o = off.to(dt).view(b, f, 1, 1, 2)
    p = torch.stack([gj + o[..., 0] - ctr_w, gi + o[..., 1] - ctr_h], dim=-1)
    q = rot2d(p, -ang)
    wj, wi = q[..., 0] + ctr_w, q[..., 1] + ctr_h
    return (wi >= -0.5) & (wi <= lat_h - 0.5) & (wj >= -0.5) & (wj <= lat_w - 0.5)


def make_weight_map(offset_px: torch.Tensor, lat_h: int = LAT_H, lat_w: int = LAT_W,
                    new_weight: float = 2.0, ramp_latent: float = 0.0, sigma=None) -> torch.Tensor:
    """loss 权重图 → float32 [b, 1, f, lat_h, lat_w]（known=1，new=new_weight）。

    直接转手 `actionrope.arope.loss_weight_map`，把 10k 检查点训练时用的 new 区 ×2 权重
    原样带进 stage1。stage2 的 CD loss **故意不用**这张图（见 PORT_设计）。
    """
    off = _as_bf2(offset_px, "offset_px")
    return loss_weight_map(off, lat_h, lat_w, new_weight=new_weight,
                           ramp_latent=ramp_latent, sigma=sigma)


# --------------------------------------------------------------------------
# 滑窗锚定
# --------------------------------------------------------------------------

def window_offsets(offset_px: torch.Tensor, a: int, k: int) -> torch.Tensor:
    """窗口 [a, k] 的锚定偏移 → [b, k−a+1, 2]，第 0 项恒为 0。

    a = max(0, k − kv_window) 是窗口里最老的保留 cell。RoPE 是相对编码，整窗同时平移
    (t, offset) 不改变任何注意力分数，所以 RoPE 重锚定是精确的（`tests/test_world_rope_window.py` a1/a3），
    当前查询 cell 的 known 掩码也精确（d1）。但**缓存里各 cell 的 known 掩码通道是按它们各自提交时的锚点
    a_j = max(0, j−20) < a 固化的**，所以 k>20 的历史与一条锚在 k−20 的训练 clip 并不逐位相同 ——
    这是 stage2→stage3 的输入分布移位，靠 self-forcing 训练穿过（审计 GEO-2）。
    """
    off = _as_bf2(offset_px, "offset_px")
    f = off.shape[1]
    if not 0 <= a <= k < f:
        raise ValueError(f"窗口下标越界: a={a}, k={k}, f={f}")
    return off[:, a:k + 1] - off[:, a:a + 1]


def window_context(context: torch.Tensor, context_ids: torch.Tensor, a: int, k: int):
    """`window_offsets` 的文本孪生：同一段 [a, k] 的 context / context_ids。

    单独写出来只有一个理由——把 "偏移窗口" 和 "文本窗口" 的下标口径钉在同一个函数里，
    stage3 的 scored tail 才不会两边差一格。
    """
    ctx, ids = _check_context(context, context_ids)
    f = ctx.shape[1]
    if not 0 <= a <= k < f:
        raise ValueError(f"窗口下标越界: a={a}, k={k}, f={f}")
    return ctx[:, a:k + 1], ids[:, a:k + 1]


# --------------------------------------------------------------------------
# rollout 延长：偏移
# --------------------------------------------------------------------------

def walk_speed_from_sidecar(sidecar) -> tuple[float, float]:
    """从 sidecar（路径或已加载的 dict）取实测步速 (px/s)。默认常量只是它的冻结副本。"""
    from actionrope.dataset import load_sidecar

    meta = load_sidecar(sidecar)["meta"]
    ws = meta["walk_speed_px_s"]
    if len(ws) != 2:
        raise ValueError(f"sidecar meta 的 walk_speed_px_s 应为 2 个数，得到 {ws!r}")
    return float(ws[0]), float(ws[1])


def append_cell_velocity(offset_px: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    """把逐 cell 速度 vel [b, m, 2]（px/s）接到 offset_px [b, F0, 2] 后面 → [b, F0+m, 2]。

    用模块 docstring 的 (1.5·v_prev + 2.5·v_cur)/FPS 增量积分，即帧级积分 + cell 内均值的
    闭式解。第一个新 cell 需要"上一 cell 的速度"，而数据只给了偏移，所以从最后一段已知增量
    反推匀速等效值 v_prev（匀速时精确；变速时那一格的误差 ≤ 1.5·|Δv|/16）。
    """
    off = _as_bf2(offset_px, "offset_px")
    if vel.dim() != 3 or vel.shape[0] != off.shape[0] or vel.shape[2] != 2:
        raise ValueError(f"vel 形状应为 [{off.shape[0]}, m, 2]，得到 {tuple(vel.shape)}")
    b, f0 = off.shape[:2]
    if f0 < 1:
        raise ValueError("offset_px 至少要有 cell 0（形状 [b, ≥1, 2]）")
    m = vel.shape[1]
    if m == 0:
        return off.clone()
    vel = vel.to(device=off.device, dtype=off.dtype)

    v_prev = _implied_last_velocity(off)  # [b, 2] px/s
    out = [off]
    cur = off[:, -1]                       # [b, 2] 最后一个已知 cell 的累计偏移
    k = f0                                 # 即将写入的绝对 cell 下标
    for j in range(m):
        v = vel[:, j]
        if k == 0:
            raise ValueError("cell 0 的偏移恒为 0，不能由速度生成")
        if k == 1:
            # cell 0 只有帧 0，起步项只有 2.5 那一半
            delta = (_W_CUR * v) / FPS
        else:
            delta = (_W_PREV * v_prev + _W_CUR * v) / FPS
        cur = cur + delta
        out.append(cur.unsqueeze(1))
        v_prev = v
        k += 1
    return torch.cat(out, dim=1)


def extend_offsets(offset_px: torch.Tensor, N: int, mode: str = "resample",
                   chunk: int = DEFAULT_CHUNK_CELLS, rng: torch.Generator | None = None,
                   walk_speed=None, speed_mult: tuple[float, float] = DEFAULT_SPEED_MULT,
                   include_idle: bool = True) -> torch.Tensor:
    """把 [b, F0, 2] 的逐 cell 偏移延长/裁剪到 rollout 需要的 [b, N+1, 2]。

    N = rollout 生成的 cell 数（不含 cell 0），所以长度是 1+N，与
    `dmd_gm.py:628-637` 的 `prompts.shape[1] == 1 + N` 口径一致。
    1+N ≤ F0 时就是切片（原样保留真值），不会引入任何合成运动。

    mode:
      * "resample"（默认）—— 每 `chunk` 个 cell 抽一次动作：方向取 `geometry.ACTION_DIRS`
        的 9 类之一（含 idle），速度按轴取 walk_speed × 倍率（[0.8, 1.2] 均匀），
        与 `infer.py::parse_actions` 的 `ACTION_DIRS[i] * walk_speed * mult` 完全同构，
        对应 ReactiveGWM 的 `dmd_action_extend_mode: preset_chunks`。逐样本独立抽。
      * "continue" —— 沿用最后一段已知增量隐含的匀速（等价于把 delta_tok[-1] 一路重复）。
      * "loop" —— 合成的走出去再原路走回来（见分支内注释）。
      * "retrace" —— 沿真实轨迹倒放回 cell 0 再顺放，回访 clip 自己的 cell（画布记忆训练用）。

    rng: CPU 上的 torch.Generator。抽样一律在 CPU 上做再搬到 offset_px.device，
    这样 4 张卡上的种子行为与设备无关；各 rank 抽到不同方向没关系（数据本来就不同），
    **形状**由 rank-synced 的 N 决定，FSDP 的 all-gather 不会错位。
    """
    off = _as_bf2(offset_px, "offset_px")
    b, f0 = off.shape[:2]
    if N < 0:
        raise ValueError(f"N 应 ≥ 0，得到 {N}")
    want = N + 1
    if want <= f0:
        return off[:, :want].contiguous()
    m = want - f0

    if mode == "continue":
        v = _implied_last_velocity(off).unsqueeze(1).expand(b, m, 2)  # [b, m, 2]
    elif mode == "resample":
        v = _sample_chunk_velocity(b, m, chunk=chunk, rng=rng, walk_speed=walk_speed,
                                   speed_mult=speed_mult, include_idle=include_idle)
        v = v.to(device=off.device, dtype=off.dtype)
    elif mode == "loop":
        # 会**回访**的合成动作（World-Canvas Memory 训练用）：前一半按 resample 抽 chunk 速度走出去，
        # 后一半把这些速度**倒序取反**走回来 —— 回程的每个 cell 都精确落在去程某个 cell 的世界坐标上，
        # 而且间隔 ≥ 窗口长度时正是"被驱逐后再回访"的情形。自然数据（15 s 长 clip）平均只有 4/40 个回访 cell，不够。
        half = (m + 1) // 2
        v_out = _sample_chunk_velocity(b, half, chunk=chunk, rng=rng, walk_speed=walk_speed,
                                       speed_mult=speed_mult, include_idle=False)
        v_back = -torch.flip(v_out, dims=[1])
        v = torch.cat([v_out, v_back], dim=1)[:, :m].to(device=off.device, dtype=off.dtype)
    elif mode == "retrace":
        # 沿**真实**轨迹来回走（World-Canvas Memory 训练用）：延长段先把 clip 的坐标倒序走回去
        # （cell F0−1+i **精确**等于 cell F0−1−i 的世界坐标，时间距离 2i），走到 cell 0 的位置后再顺着原路
        # 走出去，如此往复。直接抄坐标而不走 append_cell_velocity 的速度积分：那条公式带 (1.5·v_prev+2.5·v_cur)
        # 的跨 cell 平滑，回程会与去程错开几个 px，而回访一致性 loss 与画布选格都要求逐位相同的世界坐标。
        # 与 "loop" 的区别：回访的是 clip 自己的 cell（含真实首帧 cell 0），而且 F0=21 时回程 20 格里有 12 格
        # 与其伙伴的时间距离 > 17（被 recency 窗口驱逐、只能由画布供给），"loop" 要 N≈60 才有同样多。
        assert f0 >= 2, "retrace 至少要 2 个真实 cell"
        back_leg = torch.flip(off[:, :-1], dims=[1])              # [b, F0−1, 2]: cell F0−2, …, 0
        fwd_leg = off[:, 1:]                                      # [b, F0−1, 2]: cell 1, …, F0−1
        legs, n_have, back = [], 0, True
        while n_have < m:
            leg = back_leg if back else fwd_leg
            legs.append(leg); n_have += leg.shape[1]; back = not back
        ext = torch.cat([off, torch.cat(legs, dim=1)[:, :m]], dim=1).contiguous()
        assert ext.shape == (b, want, 2), f"延长失败: {tuple(ext.shape)} != {(b, want, 2)}"
        return ext
    else:
        raise ValueError(f"未知 mode {mode!r}，可选 'resample' | 'continue' | 'loop' | 'retrace'")
    ext = append_cell_velocity(off, v)
    assert ext.shape == (b, want, 2), f"延长失败: {tuple(ext.shape)} != {(b, want, 2)}"
    return ext


# --------------------------------------------------------------------------
# rollout 延长：文本
# --------------------------------------------------------------------------

def extend_context(context: torch.Tensor, context_ids: torch.Tensor, N: int, mode: str = "hold"):
    """把 [b, F0, L, C] 的逐 cell 文本与 [b, F0] 的 id 延长/裁剪到 1+N 个 cell。

    返回 (context_ext [b, 1+N, L, C], ids_ext [b, 1+N])，与 `extend_offsets` 逐下标对齐。

    mode="hold"（唯一实现）：cell ≥ F0 一律复制**最后一条已写好的** cell 文本及其 id，
    照抄 `_wm/an earlier internal project/distill/code/stage3_dmd/dmd_gm.py:628-637`。ARoPE 臂的 prompt 已被
    `prompts.strip_action` 剥成"只剩场景"，rollout 又不换场景，所以保持是语义正确的；
    id 必须跟着一起保持，`arope_forward` 靠它去重 text_embedding，id 错了就会拿错一行文本。
    转场语料（burn / hardcut）的 cell 20 已经是烧完之后的目标场景，保持它同样正确。

    注意：文本"保持"而偏移"继续走"，两者策略不同是**有意的**（见模块 docstring），
    不是这里漏了一种模式。
    """
    ctx, ids = _check_context(context, context_ids)
    b, f0 = ctx.shape[:2]
    if N < 0:
        raise ValueError(f"N 应 ≥ 0，得到 {N}")
    if mode != "hold":
        raise ValueError(f"未知 mode {mode!r}：文本只实现 'hold'（保持最后一条已写好的 cell 文本）")
    want = N + 1
    if want <= f0:
        return ctx[:, :want].contiguous(), ids[:, :want].contiguous()
    pad_n = want - f0
    ctx_pad = ctx[:, f0 - 1:f0].expand(b, pad_n, *ctx.shape[2:])
    ids_pad = ids[:, f0 - 1:f0].expand(b, pad_n)
    ctx_ext = torch.cat([ctx, ctx_pad], dim=1)
    ids_ext = torch.cat([ids, ids_pad], dim=1)
    assert ctx_ext.shape[1] == want and ids_ext.shape[1] == want, (
        f"文本延长失败: {ctx_ext.shape[1]}/{ids_ext.shape[1]} != {want}")
    return ctx_ext, ids_ext


def assert_rollout_aligned(offset_ext: torch.Tensor, context_ext: torch.Tensor,
                           ids_ext: torch.Tensor, N: int, latent=None) -> None:
    """rollout 前的对齐闸：三条（可选四条）流的 cell 数必须都是 1+N，且 batch 一致。

    stage3 每次 rollout 之前、每次 teacher/critic forward 之前都调一下——这类错位
    不会报错、只会让 loss 照降而训错东西，所以要用断言把它变响。
    """
    want = N + 1
    b = offset_ext.shape[0]
    if offset_ext.shape != (b, want, 2):
        raise AssertionError(f"offset 形状 {tuple(offset_ext.shape)} != {(b, want, 2)}")
    if context_ext.shape[:2] != (b, want):
        raise AssertionError(f"context 形状 {tuple(context_ext.shape)} 前两维 != {(b, want)}")
    if tuple(ids_ext.shape) != (b, want):
        raise AssertionError(f"context_ids 形状 {tuple(ids_ext.shape)} != {(b, want)}")
    if latent is not None and (latent.shape[0] != b or latent.shape[2] != want):
        raise AssertionError(f"latent 形状 {tuple(latent.shape)} 与 [b={b}, ·, f={want}, ·, ·] 不符")


# --------------------------------------------------------------------------
# 内部
# --------------------------------------------------------------------------

def _as_bf2(off: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(off, torch.Tensor):
        raise TypeError(f"{name} 应为 Tensor，得到 {type(off)}")
    if off.dim() != 3 or off.shape[-1] != 2:
        raise ValueError(f"{name} 形状应为 [b, f, 2]，得到 {tuple(off.shape)}")
    return off


def _check_context(context: torch.Tensor, context_ids: torch.Tensor):
    if context.dim() != 4:
        raise ValueError(f"context 形状应为 [b, f, L, C]，得到 {tuple(context.shape)}")
    if context_ids.dim() != 2:
        raise ValueError(f"context_ids 形状应为 [b, f]，得到 {tuple(context_ids.shape)}")
    if context.shape[:2] != context_ids.shape:
        raise ValueError(f"context {tuple(context.shape)[:2]} 与 context_ids "
                         f"{tuple(context_ids.shape)} 的 (b, f) 不一致")
    return context, context_ids


def _implied_last_velocity(off: torch.Tensor) -> torch.Tensor:
    """从最后一段已知增量反推匀速等效速度 [b, 2]（px/s）。

    匀速时 delta[k] = 4·v/FPS（k ≥ 2），delta[1] = 2.5·v/FPS；F0 = 1（只有 cell 0）时
    没有任何增量可用，只能取 0（此时 'continue' 退化成"站住不动"，调用方应当用 'resample'）。
    """
    b, f0 = off.shape[:2]
    if f0 < 2:
        return off.new_zeros(b, 2)
    delta = off[:, -1] - off[:, -2]
    w = _W_CUR if f0 == 2 else (_W_PREV + _W_CUR)  # 2.5（cell 1） / 4.0（cell ≥ 2）
    return delta * (FPS / w)


def _sample_chunk_velocity(b: int, m: int, chunk: int, rng: torch.Generator | None,
                           walk_speed, speed_mult: tuple[float, float],
                           include_idle: bool) -> torch.Tensor:
    """逐样本、每 chunk 个 cell 抽一次动作 → 逐 cell 速度 [b, m, 2]（px/s，float32 CPU）。"""
    if chunk < 1:
        raise ValueError(f"chunk 应 ≥ 1，得到 {chunk}")
    if rng is not None and rng.device.type != "cpu":
        raise ValueError("rng 必须是 CPU 上的 torch.Generator（抽样固定在 CPU 做，保证跨设备一致）")
    lo, hi = float(speed_mult[0]), float(speed_mult[1])
    if not 0.0 <= lo <= hi:
        raise ValueError(f"speed_mult 应为 0 ≤ lo ≤ hi，得到 {speed_mult!r}")
    ws = WALK_SPEED_PX_S if walk_speed is None else walk_speed
    ws = torch.tensor([float(ws[0]), float(ws[1])], dtype=torch.float32)
    dirs = torch.from_numpy(np.asarray(ACTION_DIRS, dtype=np.float32))  # [9, 2] 单位方向，0 = idle
    n_dir = dirs.shape[0]
    lo_idx = 0 if include_idle else 1

    n_chunk = math.ceil(m / chunk)
    idx = torch.randint(lo_idx, n_dir, (b, n_chunk), generator=rng)          # [b, n_chunk]
    mult = torch.rand((b, n_chunk, 1), generator=rng) * (hi - lo) + lo       # [b, n_chunk, 1]
    v_chunk = dirs[idx] * ws.view(1, 1, 2) * mult                            # [b, n_chunk, 2]
    # 每个 chunk 的速度铺满它覆盖的 cell（最后一个 chunk 可能不满，截断）
    return v_chunk.repeat_interleave(chunk, dim=1)[:, :m].contiguous()


__all__ = [
    "ACTION_DIRS", "CELLS", "CELL_FRAMES", "DEFAULT_CHUNK_CELLS", "DEFAULT_SPEED_MULT",
    "WALK_SPEED_PX_S", "append_cell_velocity", "assert_rollout_aligned", "extend_context",
    "extend_offsets", "make_mask_input", "make_weight_map", "to_device",
    "walk_speed_from_sidecar", "window_context", "window_offsets",
]
