"""几何工具：帧/cell 映射、日志 → 屏幕偏移采样、known/new 掩码。

sidecar 生成（actionrope/sidecar.py）和推理造指令（actionrope/infer.py）共用这一套，
避免两处各写一套时符号或分组规则悄悄不一致。

坐标约定（与 设计 一致）：
    屏幕 u 向右、v 向下，单位 px（832×480 画幅）
    p_px1280 = [x, y, z] @ M_px1280     玩家的"屏幕投影"，含高度项 y
    p_px832  = p_px1280 * [832/1280, 480/720]
    frame_offset_px[i] = p_px832(frame i) − p_px832(frame 0)
相机刚性钉在玩家上，所以背景相对屏幕的位移 = −frame_offset_px。这个符号由
actionrope/verify_sidecar.py 对视频实测，不是照抄日志。
"""
from __future__ import annotations

import json
import os

import numpy as np

FRAMES, CELLS, W, H = 81, 21, 832, 480
LAT_H, LAT_W, TOK_H, TOK_W = 30, 52, 15, 26
PX_PER_TOKEN, PX_PER_LATENT = 32, 16
FPS = 16.0

# 日志里的 M 是 1280×720 画幅下解出来的；clip 是直接 scale 到 832×480（不裁不补），
# 所以横纵各自按比例缩，横向 0.65、纵向 0.6667 —— 2.5% 的横向压缩要保留。
SCALE_1280_TO_832 = np.array([W / 1280.0, H / 720.0], np.float64)

# 采样有效性判定。日志 20 Hz，最近邻误差正常 ≤ 0.025 s；超过 0.15 s 说明日志断档。
# 玩家跑动上限 ~2.5 m/s × 27 px/m ≈ 70 px/s ≈ 4 px/帧，单帧 200 px 只可能是传送。
MAX_SAMPLE_DT = 0.15
MAX_STEP_PX = 200.0
# 相邻两帧所选日志样本的 vt 差：正常 ≤ 1/16 + 2×max_sample_dt（0.15 s 规则下 ≈ 0.36 s）。
# 远超这个量（或倒退）只会发生在 offsets.json 的 offset 台阶处（20260823 的 vt=14400，
# vid_t 局部非单调），那里的最近邻会跨过 4 s 的日志跳到另一侧，逐帧偏移出现 ~150 px 的
# 假跳变但没到 MAX_STEP_PX。阈值取 max(MAX_LOG_JUMP_S, 1/16 + 2×max_sample_dt + 0.1)，
# 放宽 max_sample_dt 时随之放宽，台阶的 4 s 总是被抓住。
MAX_LOG_JUMP_S = 0.5

# 9 类动作 → 屏幕单位方向 (du, dv)，索引与 data/code/scenes.py 的 ACTIONS 一致：
# 0 idle, 1 up, 2 up-left, 3 left, 4 down-left, 5 down, 6 down-right, 7 right, 8 up-right。
_S = np.sqrt(0.5)
ACTION_DIRS = np.array([
    [0.0, 0.0], [0.0, -1.0], [-_S, -_S], [-1.0, 0.0], [-_S, _S],
    [0.0, 1.0], [_S, _S], [1.0, 0.0], [_S, -_S],
], np.float64)


def cell_frames(k: int) -> list[int]:
    """cell k 覆盖的帧下标：cell 0 = 帧 0；cell k = 帧 4k−3 … 4k（Wan2.2 因果 VAE 的分组）。"""
    if not 0 <= k < CELLS:
        raise ValueError(f"cell 下标越界: {k}")
    return [0] if k == 0 else list(range(4 * k - 3, 4 * k + 1))


def frames_to_cells(frame_offset_px: np.ndarray) -> np.ndarray:
    """逐帧偏移 [81,2] → 逐 cell 偏移 [21,2]：cell 内各帧取均值。cell 0 就是帧 0。"""
    off = np.asarray(frame_offset_px, np.float64)
    if off.shape != (FRAMES, 2):
        raise ValueError(f"frame_offset_px 形状应为 ({FRAMES}, 2)，得到 {off.shape}")
    return np.stack([off[cell_frames(k)].mean(0) for k in range(CELLS)], 0)


def load_session_tracks(logs_dir: str, session: str) -> dict:
    """读一个 session 的日志：按 vt 排序去重的世界轨迹 + offset 分段 + 两种画幅下的 M。

    与 data/code/make_clips.py 的 build_tracks 同一套处理：日志有重复时间戳，
    不去重的话 gradient / 最近邻都会出问题；这里只保留 diff(vt) > 1e-6 的样本。
    make_clips 用的是默认 argsort（不稳定排序），这里用 stable —— 两个 session 上核对过
    保留下来的样本下标逐一相同（重复 vt 的样本本来就按写入顺序排在一起）。
    """
    d = os.path.join(logs_dir, session)
    st = np.load(os.path.join(d, "state.npz"))
    o = np.argsort(st["vt"], kind="stable")
    vt = st["vt"][o].astype(np.float64)
    x = st["x"][o].astype(np.float64)
    y = st["y"][o].astype(np.float64)
    z = st["z"][o].astype(np.float64)
    keep = np.concatenate([[True], np.diff(vt) > 1e-6])
    vt, x, y, z = vt[keep], x[keep], y[keep], z[keep]

    with open(os.path.join(d, "offsets.json"), encoding="utf-8") as fh:
        align = json.load(fh)
    segs = align["segments"]
    off_los = np.array([s["vt_lo"] for s in segs], np.float64)
    off_vals = np.array([s["offset"] for s in segs], np.float64)
    if align.get("M_px1280") is None:
        raise ValueError(f"{session} 的 offsets.json 没有 M_px1280")
    M_px1280 = np.array(align["M_px1280"], np.float64)
    if M_px1280.shape != (3, 2):
        raise ValueError(f"M_px1280 形状应为 (3, 2)，得到 {M_px1280.shape}")
    M_px832 = M_px1280 * SCALE_1280_TO_832[None, :]

    tracks = dict(session=session, vt=vt, x=x, y=y, z=z,
                  off_los=off_los, off_vals=off_vals,
                  M_px1280=M_px1280, M_px832=M_px832)
    # video_t = vt + offset(vt)。20260823 在 vt=14400 有 −4.13 s 的台阶，vid_t 在那里
    # 局部不单调；make_clips 就是在这条数组上 searchsorted 的，为了逐帧真值一致这里照做，
    # 落在台阶附近的 clip 靠 MAX_SAMPLE_DT / MAX_LOG_JUMP_S 判 invalid。
    tracks["vid_t"] = vt + offset_at(tracks, vt)
    return tracks


def offset_at(tracks: dict, vt):
    """分段常数 offset(vt)。段以 vt_lo 为界，右闭左开与 make_clips 一致。"""
    i = np.clip(np.searchsorted(tracks["off_los"], vt, side="right") - 1,
                0, tracks["off_vals"].size - 1)
    return tracks["off_vals"][i]


def project_px832(tracks: dict, x, y, z):
    """世界坐标 → 832×480 屏幕投影 (u, v)。支持标量或同形数组。"""
    M = tracks["M_px832"]
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); z = np.asarray(z, np.float64)
    u = x * M[0, 0] + y * M[1, 0] + z * M[2, 0]
    v = x * M[0, 1] + y * M[1, 1] + z * M[2, 1]
    return u, v


def nearest_log_index(tracks: dict, video_t: np.ndarray) -> np.ndarray:
    """按 video_t 取最近邻日志样本下标（与 make_clips.process_segment 逐帧一致）。"""
    vid_t = tracks["vid_t"]
    i = np.searchsorted(vid_t, video_t)
    i = np.clip(i, 1, vid_t.size - 1)
    prev = i - 1
    return np.where(np.abs(vid_t[i] - video_t) <= np.abs(vid_t[prev] - video_t), i, prev)


def sample_frame_offsets(tracks: dict, video_t0: float, n_frames: int = FRAMES,
                         return_valid: bool = False, max_sample_dt: float = MAX_SAMPLE_DT):
    """clip 的逐帧相机偏移 [n_frames, 2]（dx, dy），帧 0 恒为 (0, 0)。

    帧 i 的 video_t = video_t0 + i/16，日志按 video_t 最近邻取样。
    valid 判定（任一条不满足 ⇒ False）：
        · 每帧最近邻日志样本的时间差 ≤ max_sample_dt（否则是日志断档）
        · 坐标无 NaN
        · 所选日志样本的 vt 单调且相邻差 ≤ max(MAX_LOG_JUMP_S, 1/16 + 2·max_sample_dt + 0.1)
          （否则跨过了 offset 台阶）
        · 单帧位移 ≤ MAX_STEP_PX（否则是传送）
    默认只返回偏移数组，且 invalid 时整个数组填 NaN —— 调用方不检查也用不到脏值。
    return_valid=True 时返回 (offsets_raw, valid, reason)，raw 里保留实际算出的值供排查。
    """
    want_t = float(video_t0) + np.arange(n_frames, dtype=np.float64) / FPS
    idx = nearest_log_index(tracks, want_t)
    dt = np.abs(tracks["vid_t"][idx] - want_t)
    u, v = project_px832(tracks, tracks["x"][idx], tracks["y"][idx], tracks["z"][idx])
    off = np.stack([u - u[0], v - v[0]], 1)
    dvt = np.diff(tracks["vt"][idx]) if n_frames > 1 else np.zeros(1)

    reason = ""
    if not np.isfinite(off).all():
        reason = "nan"
    elif dt.max() > max_sample_dt:
        reason = f"log_gap dt={dt.max():.3f}s"
    elif n_frames > 1 and (dvt.min() < 0 or dvt.max() > max(MAX_LOG_JUMP_S, 1.0 / FPS + 2.0 * max_sample_dt + 0.1)):
        reason = f"log_jump dvt=[{dvt.min():.3f},{dvt.max():.3f}]s"
    else:
        step = np.hypot(*np.diff(off, axis=0).T)
        if step.size and step.max() > MAX_STEP_PX:
            reason = f"teleport step={step.max():.1f}px"
    valid = reason == ""
    if return_valid:
        return off, valid, reason
    return off if valid else np.full_like(off, np.nan)


def transition_frame_offsets(off_a: np.ndarray, off_b: np.ndarray, n_pre: int) -> np.ndarray:
    """转场 clip 的逐帧偏移：帧 < n_pre 抄 src_a，之后接 src_b 的增量，累计位移保持连续。

        i <  n_pre : off[i] = offA[i]
        i >= n_pre : off[i] = offA[n_pre−1] + (offB[i] − offB[n_pre−1])
    """
    off_a = np.asarray(off_a, np.float64); off_b = np.asarray(off_b, np.float64)
    if off_a.shape != off_b.shape or off_a.shape[1] != 2:
        raise ValueError(f"src_a/src_b 偏移形状不一致: {off_a.shape} vs {off_b.shape}")
    if not 1 <= n_pre <= off_a.shape[0]:
        raise ValueError(f"n_pre 越界: {n_pre}")
    out = off_a.copy()
    out[n_pre:] = off_a[n_pre - 1] + (off_b[n_pre:] - off_b[n_pre - 1])
    return out


def _known_mask(offset_px: np.ndarray, h: int, w: int, px_per_unit: float) -> np.ndarray:
    """cell k 中格 (i_h, j_w) 的世界坐标 = (i_h + dy/px, j_w + dx/px)；落在首帧足迹
    [−0.5, h−0.5] × [−0.5, w−0.5] 内 ⇒ known。边界取闭区间，偏移恰为半格时仍算 known。"""
    off = np.asarray(offset_px, np.float64)
    if off.ndim != 2 or off.shape[1] != 2:
        raise ValueError(f"offset_px 形状应为 (F, 2)，得到 {off.shape}")
    ih = np.arange(h, dtype=np.float64)[None, :, None] + off[:, 1, None, None] / px_per_unit
    jw = np.arange(w, dtype=np.float64)[None, None, :] + off[:, 0, None, None] / px_per_unit
    ok_h = (ih >= -0.5) & (ih <= h - 0.5)
    ok_w = (jw >= -0.5) & (jw <= w - 0.5)
    return ok_h & ok_w


def known_mask_tokens(offset_px: np.ndarray) -> np.ndarray:
    """token 分辨率 known 掩码 [F, 15, 26]（token = 32 px）。"""
    return _known_mask(offset_px, TOK_H, TOK_W, PX_PER_TOKEN)


def known_mask_latent(offset_px: np.ndarray) -> np.ndarray:
    """latent 分辨率 known 掩码 [F, 30, 52]（latent 像素 = 16 px）。"""
    return _known_mask(offset_px, LAT_H, LAT_W, PX_PER_LATENT)
