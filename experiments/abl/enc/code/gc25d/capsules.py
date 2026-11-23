"""v10 胶囊白模角色：逐像素射线求交（数值代理里的角色部位）+ 相机射线工具。

prep（训练条件）与代码世界（推理条件）共用这一份：角色部位 raw id 只由这里的函数决定。

几何约定（与 geometry.py / 数据集 capture_result.json 一致）
------------------------------------------------------------------
* world_to_clip / world_to_view 是行主序 4×4、**行向量**：clip = [X,Y,Z,1] @ M。
* 源分辨率 1280×704，像素中心在整数坐标；ndc_x = ((x+0.5)/1280 − 0.5)·2，ndc_y = (0.5 − (y+0.5)/704)·2。
* 目标像素（训练分辨率 W×H = geometry.DST_W × DST_H，env GC_RES，默认 832×480）(x, y) 一律用 geometry._IDX_X[x] /
  _IDX_Y[y] 选中的**源像素中心**发射线 ⇒ 在 W×H 上画与 resample_ids(在 1280×704 上画) 逐位相同
  （1280×704 时 _IDX_* 是恒等映射：每个源像素一条射线）。
* 射线：在 ndc z=0.9 与 0.1 两处反投影得到两点，方向朝视线前方，起点退后 50 m（cond_v9.char_shading 原样）；
  正交（v9_stylized）与透视（v10_real，reversed-Z 无限远）都成立。

部位（model.PART_NAMES 的顺序）
--------------------------------
core = pelvis_spine / spine_neck / head；l_arm = l_upperarm / l_forearm；r_arm = r_*；l_leg = l_thigh / l_calf；
r_leg = r_*；weapon。左右按骨骼名，不按屏幕。raw id 见 model.PLAYER_PART_IDS / BOSS_PART_IDS。

label_parts(ids, J)：训练条件 = UE 的 ids（含场景遮挡）里 10 / 11 的像素，按最近的胶囊命中换成部位 raw id；
命中角色与像素角色不同（两人深度并列的边缘像素）或没命中（≈0.02 %）⇒ 该角色的 core。
"""
from __future__ import annotations

import numpy as np

import geometry as GEO
import model as GM

SRC_W, SRC_H = GEO.SRC_W, GEO.SRC_H
CAPSULE_PART = {
    "pelvis_spine": 0, "spine_neck": 0, "head": 0,
    "l_upperarm": 1, "l_forearm": 1, "r_upperarm": 2, "r_forearm": 2,
    "l_thigh": 3, "l_calf": 3, "r_thigh": 4, "r_calf": 4,
    "weapon": 5,
}
CAPSULE_SLOTS = ("pelvis_spine", "spine_neck", "head", "l_upperarm", "l_forearm", "r_upperarm", "r_forearm",
                 "l_thigh", "l_calf", "r_thigh", "r_calf", "weapon")
ROLE_CLASS = {"player": GEO.PLAYER_ID, "boss": GEO.BOSS_ID}
ROLE_PART_IDS = {GEO.PLAYER_ID: GM.PLAYER_PART_IDS, GEO.BOSS_ID: GM.BOSS_PART_IDS}


def part_raw_id(class_id: int, part: int) -> int:
    return ROLE_PART_IDS[int(class_id)][int(part)]


# ----------------------------------------------------------------------------
# 射线
# ----------------------------------------------------------------------------

def cap_hit(ro, rd, a, b, r):
    """ray / capsule intersection (vectorised over rays); returns t (inf = miss). 逐字照抄 cond_v9.py。"""
    ba = b - a; oa = ro - a; baba = ba @ ba; bard = rd @ ba; baoa = oa @ ba; rdoa = np.einsum('ij,ij->i', rd, oa); oaoa = np.einsum('ij,ij->i', oa, oa)
    A = baba - bard * bard; B = baba * rdoa - baoa * bard; C = baba * oaoa - baoa * baoa - r * r * baba
    h = B * B - A * C; t = np.full(len(ro), np.inf)
    ok = (h >= 0) & (np.abs(A) > 1e-9)
    t1 = np.where(ok, (-B - np.sqrt(np.maximum(h, 0))) / np.where(np.abs(A) > 1e-9, A, 1), np.inf); y = baoa + t1 * bard
    body = ok & (y > 0) & (y < baba); t = np.where(body, t1, t)
    for end in (a, b):   # sphere caps
        oc = ro - end; bb = np.einsum('ij,ij->i', rd, oc); cc = np.einsum('ij,ij->i', oc, oc) - r * r; hh = bb * bb - cc
        ts = np.where(hh > 0, -bb - np.sqrt(np.maximum(hh, 0)), np.inf); t = np.minimum(t, np.where(ts > 0, ts, np.inf))
    return t


def src_pixel_rays(world_to_clip, world_to_view, sx, sy):
    """源像素坐标（整数像素中心，可为 float 数组）→ 世界系射线 (ro [n,3], rd [n,3])，rd 单位长、朝视线前方。"""
    M = np.asarray(world_to_clip, np.float64).reshape(4, 4)
    V = np.asarray(world_to_view, np.float64).reshape(4, 4)
    Mi = np.linalg.inv(M)
    sx = np.asarray(sx, np.float64).ravel(); sy = np.asarray(sy, np.float64).ravel()
    nx = ((sx + 0.5) / SRC_W - 0.5) * 2
    ny = (0.5 - (sy + 0.5) / SRC_H) * 2

    def unproj(z):
        p = np.stack([nx, ny, np.full_like(nx, z), np.ones_like(nx)], 1) @ Mi
        return p[:, :3] / p[:, 3:4]

    p0, p1 = unproj(0.9), unproj(0.1)
    rd = p1 - p0
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    fwd = V[:3, 2] / np.linalg.norm(V[:3, 2])        # 视线前方的世界方向（行向量约定）
    if len(rd) and (rd @ fwd).mean() < 0:
        rd = -rd
    ro = p0 - rd * 5000.0
    return ro, rd


def dst_pixel_rays(world_to_clip, world_to_view, xs, ys):
    """目标像素 (xs, ys)（整数，W×H 上）→ 射线：经 geometry._IDX_X / _IDX_Y 选中的源像素中心。"""
    return src_pixel_rays(world_to_clip, world_to_view, GEO._IDX_X[np.asarray(xs)], GEO._IDX_Y[np.asarray(ys)])


# ----------------------------------------------------------------------------
# 胶囊
# ----------------------------------------------------------------------------

def capsules_from_json(J: dict) -> list[tuple[np.ndarray, np.ndarray, float, int, int]]:
    """per-frame json → [(a, b, r, class_id, part)]（世界 cm）。用 json['v9']['characters']（含武器）。"""
    out = []
    for ch in ((J.get("v9") or {}).get("characters") or []):
        cls = int(ch.get("class_id", ROLE_CLASS.get(ch.get("role"), 0)))
        if cls not in ROLE_PART_IDS:
            continue
        for c in ch["capsules"]:
            part = CAPSULE_PART.get(c["name"])
            if part is None:
                raise KeyError(f"未知胶囊名 {c['name']!r}")
            out.append((np.asarray(c["a"], np.float64), np.asarray(c["b"], np.float64), float(c["r"]), cls, part))
    return out


def capsules_to_array(caps) -> np.ndarray:
    """[(a,b,r,cls,part)] → float32 [2, 12, 7]（role 0 玩家 / 1 Boss；槽位 = CAPSULE_SLOTS；a xyz, b xyz, r；缺 = NaN）。
    给 meta npz 存档用（代码世界的姿态库、对拍测试都从这里读）。"""
    arr = np.full((2, len(CAPSULE_SLOTS), 7), np.nan, np.float32)
    seen = {}
    for a, b, r, cls, part in caps:
        role = 0 if cls == GEO.PLAYER_ID else 1
        names = [n for n in CAPSULE_SLOTS if CAPSULE_PART[n] == part]
        k = seen.get((role, part), 0)
        seen[(role, part)] = k + 1
        if k >= len(names):
            raise ValueError(f"role {role} 部位 {part} 的胶囊数超过槽位 {names}")
        slot = CAPSULE_SLOTS.index(names[k])
        arr[role, slot, :3], arr[role, slot, 3:6], arr[role, slot, 6] = a, b, r
    return arr


def capsules_from_array(arr) -> list[tuple[np.ndarray, np.ndarray, float, int, int]]:
    out = []
    for role in range(arr.shape[0]):
        cls = GEO.PLAYER_ID if role == 0 else GEO.BOSS_ID
        for slot, name in enumerate(CAPSULE_SLOTS):
            v = np.asarray(arr[role, slot], np.float64)
            if np.isnan(v).any():
                continue
            out.append((v[:3], v[3:6], float(v[6]), cls, CAPSULE_PART[name]))
    return out


def nearest_capsule(ro, rd, caps):
    """每条射线最近命中的胶囊 → (raw 部位 id [n] uint8，0 = 没命中；t [n] float64，inf = 没命中)。"""
    n = len(ro)
    best = np.full(n, np.inf)
    raw = np.zeros(n, np.uint8)
    for a, b, r, cls, part in caps:
        t = cap_hit(ro, rd, a, b, r)
        w = t < best
        if w.any():
            best[w] = t[w]
            raw[w] = part_raw_id(cls, part)
    return raw, best


def label_parts(ids_dst: np.ndarray, world_to_clip, world_to_view, caps) -> np.ndarray:
    """W×H 的 ids（UE 标签重采样，角色 = 10 / 11）→ parts 词表的 raw id 图（同形状 uint8 新数组）。

    只改 10 / 11 的像素：最近命中的胶囊属于同一角色 ⇒ 该胶囊部位；否则（深度并列 / 没命中）⇒ 该角色 core。"""
    out = ids_dst.copy()
    ys, xs = np.nonzero((ids_dst == GEO.PLAYER_ID) | (ids_dst == GEO.BOSS_ID))
    if len(ys) == 0 or not caps:
        return out
    ro, rd = dst_pixel_rays(world_to_clip, world_to_view, xs, ys)
    raw, _ = nearest_capsule(ro, rd, caps)
    pix_cls = ids_dst[ys, xs]
    hit_cls = GM.collapse_parts(raw)
    same = (raw != 0) & (hit_cls == pix_cls)
    out[ys[same], xs[same]] = raw[same]
    return out


_WORLD_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64)


def render_capsules(static_ids: np.ndarray, world_to_clip, world_to_view, caps, static_depth=None,
                    pad_px: float = 2.0, cam_axes: bool | None = None) -> np.ndarray:
    """代码世界的画法：在 W×H 的静态语义图上画胶囊角色（parts raw id），各胶囊只在投影包围盒里发射线。

    static_depth：可选 [H,W] 的静态场景沿射线距离（与 cap_hit 的 t 同一参数化：起点 = 视平面后 50 m）；给了 ⇒
    只在胶囊比场景近的像素上画（场景遮挡）。不给 ⇒ 角色永远画在最上层（v10_real 的遮挡只占 ≈1 %）。

    包围盒 = 端点 ± r 沿世界轴的投影 + pad_px（dst px）。斜视下世界轴会少算球的屏幕半径（≈0.42 r），盒外的角色像素被裁掉。
    cam_axes=True 再加端点 ± r 沿相机 right / up / forward（world_to_view 的列）⇒ 实测不再漏（29 条 has_ext clip 每 3 帧，
    相对 pad 64：832×480 0.080 % → 0，1280×704 0.149 % → 0）。默认 None = 832×480（LEGACY_RES）关、其它分辨率开：
    832×480 与旧实现逐位相同（data/gc25d_v10 的 cond_ext_canvas、verify 报告都是这样画的）。"""
    H, W = static_ids.shape
    out = static_ids.copy()
    if not caps:
        return out
    M = np.asarray(world_to_clip, np.float64).reshape(4, 4)
    if cam_axes is None:
        cam_axes = GEO.RES_TAG != GEO.LEGACY_RES
    dirs = _WORLD_DIRS
    if cam_axes:
        R = np.asarray(world_to_view, np.float64).reshape(4, 4)[:3, :3]     # 列 = 相机轴的世界方向（行向量约定）
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        dirs = np.concatenate([_WORLD_DIRS, R.T, -R.T], 0)
    # 所有胶囊端点 ± r 的投影包围盒（源像素）→ 目标像素
    pts = []
    for a, b, r, _, _ in caps:
        for p in (a, b):
            for d in dirs:
                pts.append(p + d * r)
    pts = np.asarray(pts)
    c = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ M
    ndc = c[:, :2] / c[:, 3:4]
    sx = (ndc[:, 0] + 1) / 2 * SRC_W - 0.5
    sy = (1 - ndc[:, 1]) / 2 * SRC_H - 0.5
    d = GEO.src_to_dst(np.stack([sx, sy], 1))
    x0 = int(max(0, np.floor(d[:, 0].min() - pad_px))); x1 = int(min(W - 1, np.ceil(d[:, 0].max() + pad_px)))
    y0 = int(max(0, np.floor(d[:, 1].min() - pad_px))); y1 = int(min(H - 1, np.ceil(d[:, 1].max() + pad_px)))
    if x1 < x0 or y1 < y0:
        return out
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    xs, ys = xx.ravel(), yy.ravel()
    ro, rd = dst_pixel_rays(world_to_clip, world_to_view, xs, ys)
    raw, t = nearest_capsule(ro, rd, caps)
    hit = raw != 0
    if static_depth is not None:
        hit &= t < static_depth[ys, xs]
    out[ys[hit], xs[hit]] = raw[hit]
    return out


# ----------------------------------------------------------------------------
# 地面（UNKNOWN 遮挡、代码世界的地面单应）
# ----------------------------------------------------------------------------

def ground_points(world_to_clip, world_to_view, xs, ys, z_ground: float) -> np.ndarray:
    """目标像素射线与水平面 Z = z_ground 的交点（世界 XY，[n,2]）；与平面平行 / 交在身后的射线 ⇒ NaN。"""
    ro, rd = dst_pixel_rays(world_to_clip, world_to_view, xs, ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_ground - ro[:, 2]) / rd[:, 2]
    p = ro[:, :2] + rd[:, :2] * t[:, None]
    p[~np.isfinite(t) | (t <= 0)] = np.nan
    return p
