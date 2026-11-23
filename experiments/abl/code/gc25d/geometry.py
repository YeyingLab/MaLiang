"""gc25d 几何：源 1280×704 → 训练分辨率（默认 832×480，env GC_RES="<W>x<H>" 可选，如 1280x704），各组件共用，
纯 numpy（resample_rgb / 画 marker 用 cv2）。分辨率的唯一出处：其它模块的 H / W / latent / 池化格形状一律从这里取。

分辨率（GC_RES，import 时解析一次；默认 "832x480" = 原行为，逐位相同）
------------------------------------------------------------------
    W、H 是 32 的倍数（latent 16× ⇒ token 网格 H/32 × W/32 为整数），384 ≤ H ≤ 704、W ≤ 1280，宽高比不超过源（只做水平居中
    裁剪）。H ≥ 384 ⇔ 缩小倍数 < 2：中心对齐最近邻的 _IDX 表才盖得到首末行 / 列（更小的 H import 时的断言会炸，parse_res 先拒）。
    CROP_W = min(1280, 2·round(352·W/H))（= 704·W/H 取最近的偶数 ⇒ 裁剪恰好居中），CROP_X0 = (1280 − CROP_W)//2：
        832×480  ⇒ 裁 x ∈ [30, 1250)（1220 px）再缩放；
        1280×704 ⇒ 不裁不缩（= 原始帧 / 原始 ids.png，_IDX_X / _IDX_Y 是恒等映射，resample_* 走恒等快路径 IDENTITY）。
    RES_TAG = "<W>x<H>"：写进 index/prep_args.json、index 行、ckpt metadata 的 "res"；缺失 ⇒ LEGACY_RES = "832x480"。
    LAT_H, LAT_W = H/16, W/16（latent 网格）；POOL_H, POOL_W = H/4, W/4（gc_sem_cond / UNKNOWN 遮挡的 4×4 池化格）。
    子进程（accelerate / DataLoader worker / spawn）继承环境变量 ⇒ 同一次运行里各进程分辨率一致。

坐标约定（与 逐字一致；括号里是 832×480 的数）：
    像素中心在整数坐标上，左上原点，x 向右、y 向下。
    裁剪源列 x ∈ [CROP_X0, CROP_X0 + CROP_W)（[30, 1250)，1220 px）、全部 704 行，再缩放到 W×H：
        x_dst = (x_src − CROP_X0 + 0.5)·W/CROP_W − 0.5      （(x_src − 30 + 0.5)·832/1220 − 0.5）
        y_dst = (y_src + 0.5)·H/704 − 0.5                    （(y_src + 0.5)·480/704 − 0.5）
    玩家恒在源 (639.5, 351.5) ⇒ dst = 画幅几何中心（832×480：(415.5, 239.5)；1280×704：(639.5, 351.5)）。
    sx = W/CROP_W，sy = H/704（0.681967 / 0.681818，0.02 % 的长宽比差异忽略）。

投影（common.py::proj 同一实现，本数据上验证到 0.0002 px）：world_to_clip 是行主序、**行向量**右乘的 4×4，
    c = [X, Y, Z, 1] @ M，ndc = c[:3]/c[3]，x_src = (ndc_x+1)/2·1280 − 0.5，y_src = (1−ndc_y)/2·704 − 0.5。

ids 用「中心对齐最近邻」（不是 cv2.INTER_NEAREST，后者有半像素偏）：
    src_x = CROP_X0 + floor((x_dst + 0.5)·CROP_W/W)，src_y = floor((y_dst + 0.5)·704/H)
这正是 dst 像素中心反投到源后四舍五入的那个源像素；用整数算术求 floor，避免浮点在 .5 边界上抖。
RGB 用 cv2.INTER_AREA：非整数倍缩小时按面积覆盖加权，同样是中心对齐的（QA 实测 RGB 边缘与 id 边界峰值偏移 0）。

pose-free marker：实心椭圆，中心 = actor position_cm 的投影（dst px），
    a = R·ppc，b = (R·|sin(pitch)| + Hh·cos(pitch))·ppc，ppc = (1280/ortho_width_cm)·sx
即世界里半径 R、半高 Hh 的竖直圆柱在正交俯视下的屏幕外接椭圆。用 cv2.ellipse(LINE_8) 的定点（shift）坐标画，
中心可以落在半像素上（玩家恰在画幅中心），画出来的像素集关于该中心对称。
「椭圆与画幅相交」在 cond 里落实为 marker_drawn：几何相交且 draw_marker 真的画出 ≥ 1 个像素（只擦到画幅外缘半个像素、
一个像素中心都没盖住的椭圆算不可见；修复前全量有 150 帧 boss 这样被标成可见却没有 11 号像素）。
"""
from __future__ import annotations

import json
import math
import os
import struct

import cv2
import numpy as np

SRC_W, SRC_H = 1280, 704
RES_ENV = "GC_RES"
DEFAULT_RES = LEGACY_RES = "832x480"     # 默认分辨率；也是没写 "res" 的旧数据根 / 旧 ckpt 的分辨率
MIN_H = SRC_H // 2 + 32                  # 384：H 的下限（缩小倍数 < 2，见 parse_res）


def crop_w_for(w: int, h: int) -> int:
    """W×H 的水平裁剪宽度（源 px）= 704·W/H 取最近的偶数（≤ 1280）：偶数 ⇒ 裁剪关于源中心 x = 639.5 对称（1152×640：1268 而非 1267）。"""
    return min(SRC_W, 2 * round(SRC_H * w / (2 * h)))


def parse_res(s: str) -> tuple[int, int]:
    """"<W>x<H>" → (W, H)。W、H 为 32 的正倍数，384 ≤ H ≤ 704、W ≤ 1280，宽高比 ≤ 源的 1280:704（只做水平裁剪）。"""
    try:
        w, h = (int(v) for v in str(s).strip().lower().split("x"))
    except ValueError:
        raise ValueError(f"{RES_ENV}={s!r}：应形如 '832x480' / '1280x704'") from None
    if w <= 0 or h <= 0 or w % 32 or h % 32 or h > SRC_H or w > SRC_W:
        raise ValueError(f"{RES_ENV}={s!r}：W、H 须为 32 的正倍数且 H ≤ {SRC_H}、W ≤ {SRC_W}")
    if w * SRC_H > h * SRC_W:
        raise ValueError(f"{RES_ENV}={s!r}：宽高比 {w / h:.4f} 超过源 {SRC_W}:{SRC_H}（需要竖直裁剪，未实现）")
    if 2 * h <= SRC_H or 2 * w <= crop_w_for(w, h):     # 缩小倍数 ≥ 2 ⇒ _IDX 取不到首末行 / 列
        raise ValueError(f"{RES_ENV}={s!r}：H 须 ≥ {MIN_H}（缩小倍数 < 2，中心对齐最近邻才盖得到源的首末行 / 列）")
    return w, h


DST_W, DST_H = parse_res(os.environ.get(RES_ENV) or DEFAULT_RES)
RES_TAG = f"{DST_W}x{DST_H}"
CROP_W = crop_w_for(DST_W, DST_H)                      # 832×480 ⇒ 1220，1280×704 ⇒ 1280
CROP_X0 = (SRC_W - CROP_W) // 2                        # 832×480 ⇒ 30，1280×704 ⇒ 0（CROP_W 偶数 ⇒ 恰好居中）
SX = DST_W / CROP_W          # 832×480：0.681967…
SY = DST_H / SRC_H           # 832×480：0.681818…
SCALE = np.array([SX, SY], np.float64)
IDENTITY = CROP_X0 == 0 and CROP_W == SRC_W and (DST_W, DST_H) == (SRC_W, SRC_H)   # 1280×704：不裁不缩
LAT_H, LAT_W = DST_H // 16, DST_W // 16                # latent 网格（832×480 ⇒ 30×52，1280×704 ⇒ 44×80）
POOL_H, POOL_W = DST_H // 4, DST_W // 4                # 4×4 池化格（832×480 ⇒ 120×208，1280×704 ⇒ 176×320）
TOK_PER_CELL = (LAT_H // 2) * (LAT_W // 2)             # DiT patch (1,2,2) 之后每个 latent cell 的 token 数（390 / 880）
AREA_VS_480 = (DST_W * DST_H) / (832 * 480)            # 按像素数定的阈值随分辨率缩放（832×480 ⇒ 1.0）
PLAYER_SRC = (639.5, 351.5)
# = src_to_dst(PLAYER_SRC)，先乘后除（整数比例下精确：832×480 ⇒ (415.5, 239.5)，与旧常数逐位相同）
PLAYER_DST = ((PLAYER_SRC[0] - CROP_X0 + 0.5) * DST_W / CROP_W - 0.5, (PLAYER_SRC[1] + 0.5) * DST_H / SRC_H - 0.5)

PLAYER_ID, BOSS_ID, UNKNOWN_ID = 10, 11, 255
STATIC_IDS = (0, 1, 2, 3, 4, 5, 6)
RAW_IDS = STATIC_IDS + (PLAYER_ID, BOSS_ID)

# 中心对齐最近邻的源索引表（整数算术：floor((2i+1)·CROP_W / (2·W))；1280×704 ⇒ 恒等）
_IDX_X = CROP_X0 + ((2 * np.arange(DST_W, dtype=np.int64) + 1) * CROP_W) // (2 * DST_W)
_IDX_Y = ((2 * np.arange(DST_H, dtype=np.int64) + 1) * SRC_H) // (2 * DST_H)
assert _IDX_X.min() == CROP_X0 and _IDX_X.max() == CROP_X0 + CROP_W - 1, (_IDX_X.min(), _IDX_X.max())
assert _IDX_Y.min() == 0 and _IDX_Y.max() == SRC_H - 1, (_IDX_Y.min(), _IDX_Y.max())

# marker 定点坐标的小数位数（cv2 上限 16；8 位 = 1/256 px）
MARKER_SHIFT = 8


# ------------------------------------------------------------------ 分辨率一致性闸 ----
def root_res(root: str) -> str:
    """数据根的分辨率 = <root>/index/prep_args.json 的 "res"；没有这个文件 / 键（旧数据根）⇒ LEGACY_RES。"""
    try:
        with open(os.path.join(root, "index", "prep_args.json"), encoding="utf-8") as fh:
            return str(json.load(fh).get("res") or LEGACY_RES)
    except FileNotFoundError:
        return LEGACY_RES


def check_root_res(root: str, what: str = "data_root") -> str:
    """数据根的 res ≠ 当前 RES_TAG ⇒ RuntimeError（点名 GC_RES）。返回 res。"""
    r = root_res(root)
    if r != RES_TAG:
        raise RuntimeError(f"{what} {root} 是 {r} 的数据（index/prep_args.json 的 res，缺失 = {LEGACY_RES}），"
                           f"本进程 {RES_ENV}={RES_TAG}：设 {RES_ENV}={r}，或换成 {RES_TAG} 的数据根")
    return r


def check_run_dir_res(run_dir: str, what: str = "--output") -> str | None:
    """蒸馏 run 目录里已有的 config.json 的 "res"（缺失 = LEGACY_RES）≠ RES_TAG ⇒ RuntimeError；没有 config.json（新目录）⇒ None。
    runner 在打开 train.log 之前调：别往另一个分辨率的 run 里追加日志 / 产物。"""
    p = os.path.join(run_dir, "config.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        r = str(json.load(fh).get("res") or LEGACY_RES)
    if r != RES_TAG:
        raise RuntimeError(f"{what} {run_dir} 是 {r} 的 run 目录（config.json 的 res，缺失 = {LEGACY_RES}），"
                           f"本进程 {RES_ENV}={RES_TAG}：换一个 {RES_TAG} 的输出目录，或设 {RES_ENV}={r}")
    return r


def ckpt_res(path: str) -> str:
    """只读 safetensors 头的 metadata "res"（不碰数据）；缺失（旧 ckpt）⇒ LEGACY_RES。"""
    with open(path, "rb") as fh:
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    return str((hdr.get("__metadata__") or {}).get("res") or LEGACY_RES)


def check_ckpt_res(path: str, what: str = "ckpt") -> str:
    """ckpt 记录的 res ≠ 当前 RES_TAG ⇒ RuntimeError（点名 GC_RES）。返回 res。"""
    r = ckpt_res(path)
    if r != RES_TAG:
        raise RuntimeError(f"{what} {path} 是在 {r} 上训练 / 导出的（metadata res，缺失 = {LEGACY_RES}），"
                           f"本进程 {RES_ENV}={RES_TAG}：设 {RES_ENV}={r}，或换成 {RES_TAG} 的 ckpt")
    return r


# ------------------------------------------------------------------ 坐标映射 ----
def src_to_dst(xy) -> np.ndarray:
    """源像素坐标 [..., 2] → dst 像素坐标 [..., 2]（float64），像素中心约定。"""
    xy = np.asarray(xy, np.float64)
    if xy.shape[-1] != 2:
        raise ValueError(f"src_to_dst 期望最后一维为 2，得到 {xy.shape}")
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - CROP_X0 + 0.5) * SX - 0.5
    out[..., 1] = (xy[..., 1] + 0.5) * SY - 0.5
    return out


def dst_to_src(xy) -> np.ndarray:
    """dst 像素坐标 [..., 2] → 源像素坐标 [..., 2]（src_to_dst 的精确逆）。"""
    xy = np.asarray(xy, np.float64)
    if xy.shape[-1] != 2:
        raise ValueError(f"dst_to_src 期望最后一维为 2，得到 {xy.shape}")
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] + 0.5) / SX - 0.5 + CROP_X0
    out[..., 1] = (xy[..., 1] + 0.5) / SY - 0.5
    return out


def proj(M, P) -> np.ndarray:
    """世界点 → 源像素坐标。

    M: [4,4] 或 [N,4,4]（world_to_clip，行主序、行向量）；P: [3] 或 [N,3]（cm）。
    返回 [2]（M、P 都不带 N 时）或 [N,2]。与 analysis/canvas_consistency/scripts/common.py::proj 同一公式。
    """
    M = np.asarray(M, np.float64)
    P = np.asarray(P, np.float64)
    single = M.ndim == 2 and P.ndim == 1
    if M.ndim == 2:
        M = M[None]
    if M.shape[1:] != (4, 4):
        raise ValueError(f"proj: M 形状应为 [N,4,4]，得到 {M.shape}")
    if P.ndim == 1:
        if P.shape != (3,):
            raise ValueError(f"proj: P 形状应为 [3] 或 [N,3]，得到 {P.shape}")
        P = np.broadcast_to(P, (M.shape[0], 3))
    if M.shape[0] == 1 and P.shape[0] > 1:
        M = np.broadcast_to(M, (P.shape[0], 4, 4))
    if P.shape != (M.shape[0], 3):
        raise ValueError(f"proj: M {M.shape} 与 P {P.shape} 的 N 不一致")
    h = np.concatenate([P, np.ones((P.shape[0], 1))], 1)
    c = np.einsum("ni,nij->nj", h, M)
    ndc = c[:, :3] / c[:, 3:4]
    out = np.stack([(ndc[:, 0] + 1) * 0.5 * SRC_W - 0.5, (1 - ndc[:, 1]) * 0.5 * SRC_H - 0.5], 1)
    return out[0] if single else out


def static_origin_src(M) -> np.ndarray:
    """org[t] = proj(M[t], (0,0,0))：世界原点的屏幕位置（源 px）。静态内容在帧 t 的 p 处 ⇒ 在帧 u 的 p + org[u] − org[t]。"""
    return proj(M, np.zeros(3))


def frame_offset_dst(org_src: np.ndarray) -> np.ndarray:
    """org [T,2]（源 px）→ frame_offset_px [T,2]（dst px）= (org[t] − org[0])·(sx, sy)，静态内容相对帧 0 的位移。"""
    org_src = np.asarray(org_src, np.float64)
    if org_src.ndim != 2 or org_src.shape[1] != 2:
        raise ValueError(f"frame_offset_dst: org 形状应为 [T,2]，得到 {org_src.shape}")
    return ((org_src - org_src[0]) * SCALE).astype(np.float32)


# ------------------------------------------------------------------ 重采样 ----
def resample_ids(ids_src: np.ndarray) -> np.ndarray:
    """uint8 [..., 704, 1280] → uint8 [..., H, W]，中心对齐最近邻（numpy fancy indexing）。
    IDENTITY（1280×704）⇒ 直接拷贝一份（与查表结果逐位相同，tests 核过）。"""
    ids_src = np.asarray(ids_src)
    if ids_src.dtype != np.uint8 or ids_src.shape[-2:] != (SRC_H, SRC_W):
        raise ValueError(f"resample_ids 期望 uint8 [..., {SRC_H}, {SRC_W}]，得到 {ids_src.dtype} {ids_src.shape}")
    if IDENTITY:
        return np.array(ids_src, order="C", copy=True)
    return np.ascontiguousarray(ids_src[..., _IDX_Y[:, None], _IDX_X[None, :]])


def resample_rgb(rgb_src: np.ndarray) -> np.ndarray:
    """uint8 [704, 1280, 3] → uint8 [H, W, 3]：裁 x∈[CROP_X0, CROP_X0+CROP_W)（832×480：[30,1250)）后 cv2.INTER_AREA 缩放。
    IDENTITY（1280×704）⇒ 直接拷贝一份（= 同尺寸 INTER_AREA，tests 核过）。"""
    rgb_src = np.asarray(rgb_src)
    if rgb_src.dtype != np.uint8 or rgb_src.shape != (SRC_H, SRC_W, 3):
        raise ValueError(f"resample_rgb 期望 uint8 [{SRC_H}, {SRC_W}, 3]，得到 {rgb_src.dtype} {rgb_src.shape}")
    if IDENTITY:
        return np.array(rgb_src, order="C", copy=True)
    crop = np.ascontiguousarray(rgb_src[:, CROP_X0:CROP_X0 + CROP_W])
    out = cv2.resize(crop, (DST_W, DST_H), interpolation=cv2.INTER_AREA)
    assert out.shape == (DST_H, DST_W, 3) and out.dtype == np.uint8, out.shape
    return out


# ------------------------------------------------------------------ 可视化 ----
PALETTE = np.full((256, 3), 255, np.uint8)        # 不在词表里的值 = 白（显眼）
PALETTE[0] = (135, 190, 235)    # sky / bg
PALETTE[1] = (140, 110, 70)     # terrain
PALETTE[2] = (200, 200, 190)    # paving
PALETTE[3] = (200, 90, 60)      # architecture
PALETTE[4] = (110, 110, 120)    # rock
PALETTE[5] = (60, 160, 60)      # vegetation
PALETTE[6] = (230, 190, 40)     # prop
PALETTE[PLAYER_ID] = (255, 0, 255)
PALETTE[BOSS_ID] = (255, 0, 0)
PALETTE[UNKNOWN_ID] = (0, 0, 0)  # 255 另外叠灰色斜纹，见 colorize_ids
HATCH_GREY = np.array([96, 96, 96], np.uint8)


def colorize_ids(ids: np.ndarray) -> np.ndarray:
    """uint8 [..., H, W] → uint8 [..., H, W, 3]（固定调色板；255 = 黑底灰色斜纹，周期 8 px）。"""
    ids = np.asarray(ids)
    if ids.dtype != np.uint8 or ids.ndim < 2:
        raise ValueError(f"colorize_ids 期望 uint8 [..., H, W]，得到 {ids.dtype} {ids.shape}")
    out = PALETTE[ids]
    unk = ids == UNKNOWN_ID
    if unk.any():
        h, w = ids.shape[-2:]
        yy, xx = np.mgrid[0:h, 0:w]
        stripe = ((xx + yy) // 4) % 2 == 1
        out[unk & np.broadcast_to(stripe, ids.shape)] = HATCH_GREY
    return out


# ------------------------------------------------------------------ marker ----
def px_per_cm_dst(ortho_width_cm: float) -> float:
    """dst 像素 / 世界 cm：(1280/ortho_width_cm)·sx（832×480：sx = 832/1220）。"""
    if not ortho_width_cm > 0:
        raise ValueError(f"ortho_width_cm 必须 > 0，得到 {ortho_width_cm}")
    return (SRC_W / float(ortho_width_cm)) * SX


def marker_axes(R_cm: float, Hh_cm: float, pitch_deg: float, ortho_width_cm: float) -> tuple[float, float]:
    """(a, b) dst px：a = R·ppc，b = (R·|sin(pitch)| + Hh·cos(pitch))·ppc（pitch 取绝对值，度）。"""
    if not (R_cm > 0 and Hh_cm > 0):
        raise ValueError(f"marker 尺寸必须为正：R={R_cm}, Hh={Hh_cm}")
    p = math.radians(abs(float(pitch_deg)))
    ppc = px_per_cm_dst(ortho_width_cm)
    return R_cm * ppc, (R_cm * abs(math.sin(p)) + Hh_cm * math.cos(p)) * ppc


def ellipse_intersects_frame(cx: float, cy: float, a: float, b: float) -> bool:
    """轴对齐椭圆与 W×H 画幅（像素区域 [−0.5, W−0.5]×[−0.5, H−0.5]）是否相交。

    精确判定：取画幅矩形里离椭圆中心最近的点，看它是否落在椭圆内（矩形凸、椭圆凸，这样判是精确的）。
    """
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return False
    qx = min(max(cx, -0.5), DST_W - 0.5)
    qy = min(max(cy, -0.5), DST_H - 0.5)
    return ((qx - cx) / a) ** 2 + ((qy - cy) / b) ** 2 <= 1.0


def draw_marker(ids_dst: np.ndarray, cx: float, cy: float, a: float, b: float, cls: int) -> np.ndarray:
    """在 uint8 [H, W] 上原地画实心轴对齐椭圆（cv2.ellipse，LINE_8，定点 shift=8），返回同一数组。"""
    if ids_dst.dtype != np.uint8 or ids_dst.shape != (DST_H, DST_W) or not ids_dst.flags.c_contiguous:
        raise ValueError(f"draw_marker 期望 C 连续 uint8 [{DST_H}, {DST_W}]，得到 {ids_dst.dtype} {ids_dst.shape}")
    s = 1 << MARKER_SHIFT
    centre = (int(round(cx * s)), int(round(cy * s)))
    axes = (max(1, int(round(a * s))), max(1, int(round(b * s))))
    cv2.ellipse(ids_dst, centre, axes, 0, 0, 360, int(cls), thickness=-1, lineType=cv2.LINE_8, shift=MARKER_SHIFT)
    return ids_dst


def marker_mask(cx: float, cy: float, a: float, b: float) -> np.ndarray:
    """单个 marker 的 bool [H, W] 掩码（与 draw_marker 像素级一致，QA / 增广复用）。"""
    m = np.zeros((DST_H, DST_W), np.uint8)
    draw_marker(m, cx, cy, a, b, 1)
    return m.astype(bool)


def marker_drawn(cx: float, cy: float, a: float, b: float) -> bool:
    """draw_marker 是否会在 W×H 上画出 ≥ 1 个像素（= cond 里 visible 标记的定义）。

    ellipse_intersects_frame 判的是与像素**区域** [−0.5, W−0.5]×[−0.5, H−0.5] 相交；只擦到最外那半个像素的椭圆
    （例如底边在 y = −0.34）一个像素中心都盖不住，cv2.ellipse 什么也不画。所以先做几何相交（快速排除），再用
    与 draw_marker 同一套定点光栅化的掩码确认真的有像素——visible 与 cond 里的像素逐帧一致。
    """
    if not ellipse_intersects_frame(cx, cy, a, b):
        return False
    return bool(marker_mask(cx, cy, a, b).any())
