#!/usr/bin/env python3
"""L3 泛化测试的「没见过的地图」：在 val clip 的世界坐标里程序化搭一个新的静态世界，用该 clip 的真实相机 + 真实胶囊角色
逐帧渲染 parts 词表的数值语义 id（= 代码世界推理时会画的东西）。

世界（源 clip 的世界坐标，cm）
------------------------------
* 地面 = 水平面 Z = median(meta.ground_z[:81])，带 2D 语义纹理：terrain(1) 打底，paving(2) 道路（折线 + 宽度）/ 广场
  （凸多边形 / 圆）按画家顺序覆盖（plaza 里的种植池再盖回 terrain）。
* 3D 图元：凸多面体（半空间交，n·x ≤ d：房身 / 山墙屋顶 / 墙 / 箱子 / 锥形树冠 / 崖壁 / 帐篷）、椭球（树冠 / 灌木 / 石头）、
  竖直圆柱（树干 / 柱子 / 桶 / 仙人掌）。类：architecture 3、rock 4、vegetation 5（树干也算）、prop 6。
* 主题（seed 决定、可复现）：village（道路 + 房子 + 树）/ forest（密林 + 石头 + 灌木）/ plaza（大片铺装 + 围墙 / 柱廊 + 道具）/
  canyon（荒漠 + 崖壁 + 巨石）。
* 放置（GRID_CM 网格）：角色足迹 = 全片 81 帧玩家 + Boss 全部胶囊的 XY 投影（含半径），距离 ≤ CLEAR_CM 的格子禁放；只在
  任一帧的地面视野膨胀 REACH_CM 内放（高物体从视野外投进来）；物体 occ 足迹之间留间隙不重叠；房子 / 树不压 paving。
* 遮挡控制：帧 0 不许任何图元挡玩家；其余帧玩家 / Boss 被挡 > MAX_OCC 的帧删挡得最多的物体（整组删），迭代到稳定。

渲染（与代码世界同一约定）
--------------------------
每个目标像素一条 capsules.dst_pixel_rays 射线（t 参数化 = render_capsules 的 t）：图元最近交点 t_p（torch fp64，GPU）、
地面交点 t_g；静态 = t_p < t_g ? 图元类 : (t_g > 0 ? 地面纹理类 : sky 0)。角色 = capsules.render_capsules(static,
static_depth=t_p)：胶囊只被图元遮挡，永远压在地面上（脚可能略低于中位地面，见 report 的 feet_vs_plane_cm）。
值域 ⊆ model.valid_raw_ids("parts")，不出现 255（整片世界都是"见过的"）。

输出 <out>/<case>/
-----------------
ids.npz（uint8 [81,H,W]，键 ids；H×W = geometry / GC_RES，默认 832×480）· prompts.json（源 clip 的 21 条 cell prompt）· actions.npy（uint8 [20,5]，
infer --actions 直接可用）· source.json（源 clip / 主题 / seed / 世界参数）· report.json（自检）· map0.png（帧 0：
colorize_ids(collapse) + cond_v9 式角色着色；FLUX 的布局输入）· map0_shaded.png（静态按法线打光 + 轮廓，看体积）·
depth0.png（16 bit，近亮）· contact.jpg（帧 0/20/40/60/80）· preview.mp4（81 帧）· plan.png（俯视总图：净空边界白、
玩家轨迹品红、Boss 红、帧 0 / 80 视野黄 / 青）。

usage: procmap.py --default4 | --clip <clip_id> --theme village --seed 0 [--case NAME]  （CUDA_VISIBLE_DEVICES 选卡）
"""
from __future__ import annotations


import os as _mos


def _maliang_repo_root(start):
    """Locate the repository root by walking up until code/gc25d exists."""
    p = _mos.path.abspath(start)
    while True:
        if _mos.path.isdir(_mos.path.join(p, "code", "gc25d")):
            return p
        parent = _mos.path.dirname(p)
        if parent == p:
            return _mos.getcwd()
        p = parent


MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _maliang_repo_root(__file__)
import argparse
import collections
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

import capsules as CAP
import geometry as GEO
import model as GM

H, W, F = GEO.DST_H, GEO.DST_W, 81
SKY, TERRAIN, PAVING, ARCH, ROCK, VEG, PROP = range(7)
GRID_CM = 10.0       # 放置网格
CLEAR_CM = 150.0     # 角色足迹外的净空
REACH_CM = 1100.0    # 视野外多远的物体还可能投进画面（≈ 11 m 高 / tan 45°）
MAX_OCC = 0.25       # 帧 1..80 角色被图元挡住的比例上限（帧 0 玩家 = 0）
ROOT = MALIANG_ROOT
DATA = f"{ROOT}/data/gc25d_v10_fxcond"
# 非默认分辨率的输出另起目录（不覆盖 832×480 的基准 case）
OUT = f"{ROOT}/outputs/genbench/procmap" + ("" if GEO.RES_TAG == GEO.DEFAULT_RES else f"_{GEO.RES_TAG}")
THEMES = ("village", "forest", "plaza", "canyon")
CASES = [  # (源 clip, 主题, seed)：2 个 v10_real（透视长焦）+ 2 个 v9_stylized（正交；P9 = framed 漂移相机）
    ("01_Sunlit_Mediterranean_Courtyard__R10_s1512__000192", "village", 1),
    ("05_Japanese_Spring_Sanctuary__R10_s5512__000096", "forest", 2),
    ("06_Overgrown_Tropical_Water_Temple__P9C_golden_s6946__000480", "plaza", 3),
    ("04_Snowbound_Mountain_Outpost__P9_autumn_s4908__000000", "canyon", 4),
]
TINT = {10: np.array([0.62, 0.80, 1.0]), 11: np.array([1.0, 0.66, 0.60])}   # cond_v9.TINT
THEME_DESC = {   # 给首帧生成（FLUX）的场景描述，与 map0 的布局对应
    "village": "a rustic village: dirt ground, stone-paved roads, houses with gable roofs, small sheds, wooden fences, "
               "a well, trees and bushes, barrels, crates and carts",
    "forest": "a dense forest: earthy forest floor, an optional narrow stone path, broadleaf and conifer trees, "
              "young trees, bushes, mossy rocks, fallen logs and stumps",
    "plaza": "a town plaza: large stone-paved square enclosed by walls or colonnades, a fountain, statues, broken "
             "ruin walls, raised planters with trees, market stalls, benches, lamp posts, crates and barrels, "
             "buildings beyond the walls",
    "canyon": "a desert canyon: sandy terrain, tall rock cliffs and mesas, large boulders, scattered stones, cacti, "
              "dry bushes, an optional old paved road and a small camp with tents and crates",
}
CLASS_NAMES = ("sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop")


# ----------------------------------------------------------------------------
# 2D / 图元构造（世界 cm）
# ----------------------------------------------------------------------------
def _dir(a):
    return np.array([math.cos(a), math.sin(a)])


def _perp(u):
    return np.array([-u[1], u[0]])


def _rot(u, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def _rect(c, w, d, yaw):
    u, v = _dir(yaw) * w / 2, _perp(_dir(yaw)) * d / 2
    c = np.asarray(c, float)
    return np.stack([c - u - v, c + u - v, c + u + v, c - u + v])


def _ngon(c, r, n=16, phase=0.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.asarray(c, float) + r / math.cos(np.pi / n) * np.stack([np.cos(a), np.sin(a)], 1)   # 外接


def _planes(rows):
    rows = np.asarray(rows, float)
    return rows / np.linalg.norm(rows[:, :3], axis=1, keepdims=True)


def prism(poly, z0, z1, cls, taper=0.0):
    """凸多边形竖直拉伸，侧面每升高 1 cm 内收 taper cm（taper = R/h 的正多边形 ⇒ 棱锥）。"""
    poly = np.asarray(poly, float)
    if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
        poly = poly[::-1]                                          # → CCW
    rows = []
    for p, q in zip(poly, np.roll(poly, -1, 0)):
        e = q - p
        n2 = np.array([e[1], -e[0]]) / np.linalg.norm(e)          # CCW 的外法向
        rows.append([n2[0], n2[1], taper, n2 @ p + taper * z0])
    rows += [[0, 0, 1, z1], [0, 0, -1, -z0]]
    return dict(kind="poly", cls=cls, planes=_planes(rows))


def box(c, w, d, h, yaw, z0, cls):
    return prism(_rect(c, w, d, yaw), z0, z0 + h, cls)


def gable(c, w, d, yaw, z, rise, oh, cls):
    """山墙屋顶（三棱柱），屋脊沿局部 x（长 w），檐口高 z，脊高 z + rise，四周出檐 oh。"""
    u = np.r_[_dir(yaw), 0.0]; v = np.r_[_perp(_dir(yaw)), 0.0]; C = np.array([c[0], c[1], 0.0]); ez = np.array([0, 0, 1.0])
    L, D = w / 2 + oh, d / 2 + oh
    k, zt = rise / D, z + rise
    rows = [np.r_[u, u @ C + L], np.r_[-u, -u @ C + L], [0, 0, -1, -z],
            np.r_[k * v + ez, k * (v @ C) + zt], np.r_[-k * v + ez, -k * (v @ C) + zt]]
    return dict(kind="poly", cls=cls, planes=_planes(rows))


def ell(c, r, yaw, cls):
    return dict(kind="ell", cls=cls, c=np.asarray(c, float), r=np.asarray(r, float), yaw=float(yaw))


def cyl(c, r, z0, z1, cls):
    return dict(kind="cyl", cls=cls, c=np.asarray(c, float)[:2], r=float(r), z0=float(z0), z1=float(z1))


def sdf_lower(p, P):
    """图元的有符号距离下界 [n]（外正内负；多面体 = max 平面距离，椭球 = (|p/r|−1)·min r，圆柱精确）。"""
    if p["kind"] == "poly":
        return (P @ p["planes"][:, :3].T - p["planes"][:, 3]).max(1)
    if p["kind"] == "ell":
        cs, sn = math.cos(p["yaw"]), math.sin(p["yaw"])
        q = P - p["c"]
        loc = np.stack([cs * q[:, 0] + sn * q[:, 1], -sn * q[:, 0] + cs * q[:, 1], q[:, 2]], 1) / p["r"]
        return (np.linalg.norm(loc, axis=1) - 1) * p["r"].min()
    dxy = np.linalg.norm(P[:, :2] - p["c"], axis=1) - p["r"]
    dz = np.maximum(p["z0"] - P[:, 2], P[:, 2] - p["z1"])
    return np.hypot(np.maximum(dxy, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(dxy, dz), 0)


# ----------------------------------------------------------------------------
# GPU 求交（torch fp64；射线来自 capsules.dst_pixel_rays，t 参数化与 cap_hit 相同）
# ----------------------------------------------------------------------------
class Caster:
    def __init__(self, prims, dev):
        self.dev = dev
        kinds = {k: [i for i, p in enumerate(prims) if p["kind"] == k] for k in ("poly", "ell", "cyl")}
        self.order = np.array(kinds["poly"] + kinds["ell"] + kinds["cyl"], np.int64)       # 拼接序 → prims 下标
        self.cls = torch.tensor([prims[i]["cls"] for i in self.order], dtype=torch.uint8, device=dev)
        T = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=dev)
        self.P = None
        if kinds["poly"]:
            K = max(len(prims[i]["planes"]) for i in kinds["poly"])
            P = np.tile(np.array([0, 0, 0, 1.0]), (len(kinds["poly"]), K, 1))     # 补位平面恒成立
            for j, i in enumerate(kinds["poly"]):
                P[j, :len(prims[i]["planes"])] = prims[i]["planes"]
            self.P = T(P)
        self.E = None
        if kinds["ell"]:
            e = [prims[i] for i in kinds["ell"]]
            self.E = (T([p["c"] for p in e]), T([1 / p["r"] for p in e]), T([math.cos(p["yaw"]) for p in e]),
                      T([math.sin(p["yaw"]) for p in e]))
        self.C = T([[*prims[i]["c"], prims[i]["r"], prims[i]["z0"], prims[i]["z1"]] for i in kinds["cyl"]]) \
            if kinds["cyl"] else None

    def _poly(self, o, d):
        P = self.P
        te = o.new_full((len(o), len(P)), -math.inf); tx = o.new_full((len(o), len(P)), math.inf)
        bad = torch.zeros(te.shape, dtype=torch.bool, device=o.device)
        for k in range(P.shape[1]):
            n, dd = P[:, k, :3], P[:, k, 3]
            den = d @ n.T
            num = dd[None] - o @ n.T
            tt = num / den
            te = torch.where(den < 0, torch.maximum(te, tt), te)
            tx = torch.where(den > 0, torch.minimum(tx, tt), tx)
            bad |= (den == 0) & (num < 0)
        return torch.where(~bad & (te <= tx) & (te > 0), te, math.inf)

    def _ell(self, o, d):
        c, ir, cs, sn = self.E
        ox, oy, oz = o[:, :1] - c[:, 0], o[:, 1:2] - c[:, 1], o[:, 2:3] - c[:, 2]
        lx, ly, lz = (cs * ox + sn * oy) * ir[:, 0], (-sn * ox + cs * oy) * ir[:, 1], oz * ir[:, 2]
        dx, dy, dz = (cs * d[:, :1] + sn * d[:, 1:2]) * ir[:, 0], (-sn * d[:, :1] + cs * d[:, 1:2]) * ir[:, 1], d[:, 2:3] * ir[:, 2]
        a = dx * dx + dy * dy + dz * dz
        b = lx * dx + ly * dy + lz * dz
        disc = b * b - a * (lx * lx + ly * ly + lz * lz - 1)
        t = (-b - disc.clamp_min(0).sqrt()) / a
        return torch.where((disc >= 0) & (t > 0), t, math.inf)

    def _cyl(self, o, d):
        C = self.C
        ox, oy = o[:, :1] - C[:, 0], o[:, 1:2] - C[:, 1]
        a = d[:, :1] ** 2 + d[:, 1:2] ** 2
        b = ox * d[:, :1] + oy * d[:, 1:2]
        disc = b * b - a * (ox * ox + oy * oy - C[:, 2] ** 2)
        sq = disc.clamp_min(0).sqrt()
        s0, s1 = (C[:, 3] - o[:, 2:3]) / d[:, 2:3], (C[:, 4] - o[:, 2:3]) / d[:, 2:3]
        te = torch.maximum((-b - sq) / a, torch.minimum(s0, s1))
        tx = torch.minimum((-b + sq) / a, torch.maximum(s0, s1))
        return torch.where((disc >= 0) & (a > 1e-12) & (te <= tx) & (te > 0), te, math.inf)

    @torch.no_grad()
    def cast(self, ro, rd, chunk=1 << 15):
        """ro, rd: numpy / torch [n,3] → (t [n] float64 numpy，inf = 没命中；k [n] int64 = prims 下标，−1 = 没命中)。"""
        ro, rd = torch.as_tensor(ro, device=self.dev), torch.as_tensor(rd, device=self.dev)
        n = len(ro)
        t = torch.full((n,), math.inf, dtype=torch.float64, device=self.dev)
        k = torch.full((n,), -1, dtype=torch.int64, device=self.dev)
        if len(self.order) == 0:
            return t.cpu().numpy(), k.cpu().numpy()
        order = torch.as_tensor(self.order, device=self.dev)
        for s in range(0, n, chunk):
            o, d = ro[s:s + chunk], rd[s:s + chunk]
            T = torch.cat([f(o, d) for f, x in ((self._poly, self.P), (self._ell, self.E), (self._cyl, self.C))
                           if x is not None], 1)
            tt, j = T.min(1)
            t[s:s + chunk] = tt
            k[s:s + chunk] = torch.where(torch.isfinite(tt), order[j], -1)
        return t.cpu().numpy(), k.cpu().numpy()


class Ground:
    """地面 2D 语义纹理：terrain 打底，图层按添加顺序覆盖。图层 = ('seg', cls, segs [S,4], 半宽 [S]) | ('poly', cls, 凸多边形)
    | ('disk', cls, (cx, cy, r))。"""

    def __init__(self):
        self.layers = []

    @torch.no_grad()
    def classify(self, xy, dev, chunk=1 << 16) -> np.ndarray:
        xy = torch.as_tensor(np.asarray(xy, np.float64), device=dev)
        out = torch.full((len(xy),), TERRAIN, dtype=torch.uint8, device=dev)
        L = [(kind, cls, [torch.as_tensor(np.asarray(a, np.float64), device=dev) for a in data])
             for kind, cls, *data in self.layers]
        for s in range(0, len(xy), chunk):
            p, o = xy[s:s + chunk], out[s:s + chunk]
            for kind, cls, data in L:
                if kind == "seg":
                    sg, hw = data
                    a, ab = sg[:, :2], sg[:, 2:] - sg[:, :2]
                    q = p[:, None] - a[None]
                    u = ((q * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0, 1)
                    m = ((q - u[..., None] * ab) ** 2).sum(-1).lt(hw ** 2).any(1)
                elif kind == "poly":
                    v, = data
                    e = torch.roll(v, -1, 0) - v
                    q = p[:, None] - v[None]
                    m = (e[None, :, 0] * q[..., 1] - e[None, :, 1] * q[..., 0]).ge(0).all(1)
                else:
                    c, = data
                    m = ((p - c[:2]) ** 2).sum(-1).lt(c[2] ** 2)
                o[m] = cls
        return out.cpu().numpy()


# ----------------------------------------------------------------------------
# 源 clip / 世界布局
# ----------------------------------------------------------------------------
def load_source(clip_id: str, root: str = DATA) -> dict:
    m = np.load(f"{root}/meta/{clip_id}.npz")
    meta = {k: np.asarray(m[k][:F]) for k in ("world_to_clip", "world_to_view", "capsules", "actor_pos", "ground_z")}
    meta["ortho"] = bool(m["orthographic"])
    lat = torch.load(f"{root}/latents/{clip_id}.pt", map_location="cpu", weights_only=True)
    row = None
    for line in open(f"{root}/index/clips.jsonl"):
        r = json.loads(line)
        if r["clip_id"] == clip_id:
            row = r
            break
    assert row is not None, clip_id
    return dict(clip_id=clip_id, row=row, meta=meta, prompts=list(lat["prompts"]),
                blocks=np.asarray(lat["actions_block"], np.uint8))


def frame_ground_poly(M, V, z, n=24) -> np.ndarray:
    """帧的地面视野多边形（画幅边界像素射线 ∩ 地面，按边界顺序；交不到地面的点丢掉）。"""
    t = np.linspace(0, 1, n, endpoint=False)
    xs = np.r_[t * (W - 1), np.full(n, W - 1), (1 - t) * (W - 1), np.zeros(n)].round().astype(int)
    ys = np.r_[np.zeros(n), t * (H - 1), np.full(n, H - 1), (1 - t) * (H - 1)].round().astype(int)
    g = CAP.ground_points(M, V, xs, ys, z)
    return g[np.isfinite(g).all(1)]


class World:
    def __init__(self, src: dict, seed: int):
        self.src, self.rng = src, np.random.default_rng(seed)
        meta = src["meta"]
        self.z0 = float(np.median(meta["ground_z"]))
        caps = meta["capsules"].reshape(-1, 7)
        caps = caps[np.isfinite(caps).all(1)]
        self.caps = caps
        self.foot = [frame_ground_poly(meta["world_to_clip"][f], meta["world_to_view"][f], self.z0)
                     for f in range(0, F, 2)]
        pts = np.concatenate(self.foot + [caps[:, :2], caps[:, 3:5]])
        lo, hi = pts.min(0) - REACH_CM - 300, pts.max(0) + REACH_CM + 300
        self.x0, self.y0 = lo
        self.nx, self.ny = (np.ceil((hi - lo) / GRID_CM)).astype(int)
        # 角色足迹 → 净空禁区
        ch = np.zeros((self.ny, self.nx), np.uint8)
        for a in caps:
            pa, pb, r = self.px(a[:2]), self.px(a[3:5]), max(1, int(round(a[6] / GRID_CM)))
            cv2.line(ch, tuple(pa.round().astype(int)), tuple(pb.round().astype(int)), 1, 2 * r + 1)
            for p in (pa, pb):
                cv2.circle(ch, tuple(p.round().astype(int)), r, 1, -1)
        self.dist = cv2.distanceTransform((ch == 0).astype(np.uint8), cv2.DIST_L2, 5) * GRID_CM
        self.blocked = self.dist <= CLEAR_CM
        reg = np.zeros_like(ch)
        for g in self.foot:
            if len(g) >= 3:
                cv2.fillPoly(reg, [self.px(g).round().astype(np.int32)], 1)
        self.view = reg.astype(bool)
        self.vdist = cv2.distanceTransform((reg == 0).astype(np.uint8), cv2.DIST_L2, 5) * GRID_CM   # 到视野的距离
        self.region = self.vdist <= REACH_CM
        self._cells = {}
        self.gmap = np.full((self.ny, self.nx), TERRAIN, np.uint8)        # 地面纹理的栅格（放置用）
        self.occ = np.zeros((self.ny, self.nx), bool)
        self.ground, self.objs = Ground(), []
        P = meta["actor_pos"][:, 0, :2]
        self.path = P[np.isfinite(P).all(1)]
        B = meta["actor_pos"][:, 1, :2]
        self.boss_path = B[np.isfinite(B).all(1)]
        self.span = float(np.hypot(*(hi - lo)))

    # --- 网格 ---
    def px(self, xy):
        """世界 XY → 网格像素坐标（col, row；像素中心在整数上）。"""
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.x0) / GRID_CM - 0.5, (xy[..., 1] - self.y0) / GRID_CM - 0.5], -1)

    def at(self, mask, xy) -> bool:
        c, r = self.px(xy).round().astype(int)
        return 0 <= r < self.ny and 0 <= c < self.nx and bool(mask[r, c])

    def sample(self, reach=REACH_CM) -> np.ndarray:
        """视野膨胀 reach cm 内均匀采一个点（小物体 reach 小 ⇒ 密度集中在看得见的地方）。"""
        if reach not in self._cells:
            self._cells[reach] = np.flatnonzero(self.vdist.ravel() <= reach)
        cells = self._cells[reach]
        i = cells[self.rng.integers(len(cells))]
        r, c = divmod(int(i), self.nx)
        return np.array([self.x0 + (c + self.rng.random()) * GRID_CM, self.y0 + (r + self.rng.random()) * GRID_CM])

    def _raster(self, polys, pad=0.0):
        """凸多边形们 → (行切片, 列切片, 局部 bool 掩码)，pad cm 膨胀；完全在网格外 ⇒ None。"""
        g = [self.px(p) for p in polys]
        allp = np.concatenate(g)
        e = int(math.ceil(pad / GRID_CM)) + 2
        c0, r0 = np.floor(allp.min(0)).astype(int) - e
        c1, r1 = np.ceil(allp.max(0)).astype(int) + e
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, self.nx), min(r1, self.ny)
        if c1 <= c0 or r1 <= r0:
            return None
        m = np.zeros((r1 - r0, c1 - c0), np.uint8)
        for p in g:
            cv2.fillConvexPoly(m, ((p - [c0, r0]) * 16).round().astype(np.int32), 1, cv2.LINE_8, 4)
        if pad > 0:
            k = 2 * int(round(pad / GRID_CM)) + 1
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return slice(r0, r1), slice(c0, c1), m.astype(bool)

    # --- 物体 ---
    def add(self, prims, fp, occ=None, gap=60.0, on_paving=False, name="") -> bool:
        """fp：完整 XY 足迹（净空 / paving 检查），occ：占地足迹（物体间不重叠，检查时膨胀 gap）。"""
        a = self._raster(fp)
        if a is None:
            return False
        rs, cs, m = a
        if (self.blocked[rs, cs] & m).any() or (not on_paving and ((self.gmap[rs, cs] == PAVING) & m).any()):
            return False
        b = self._raster(occ if occ is not None else fp, gap)
        if b is None or (self.occ[b[0], b[1]] & b[2]).any():
            return False
        b = self._raster(occ if occ is not None else fp)
        self.occ[b[0], b[1]] |= b[2]
        self.objs.append(dict(name=name, prims=prims, alive=True))
        return True

    def scatter(self, make, n_max, spacing, tries=None, reach=REACH_CM, **kw) -> int:
        acc = []
        for _ in range(tries or 20 * n_max):
            if len(acc) >= n_max:
                break
            p = self.sample(reach)
            if acc and np.min(np.hypot(*(np.asarray(acc) - p).T)) < spacing:
                continue
            o = make(p)
            if o is not None and self.add(*o, **kw):
                acc.append(p)
        return len(acc)

    def prims(self):
        out = []
        for gi, o in enumerate(self.objs):
            if o["alive"]:
                for p in o["prims"]:
                    out.append(dict(p, obj=gi))
        return out

    # --- 地面纹理 ---
    def road(self, pts, width, cls=PAVING):
        pts = np.asarray(pts, float)
        self.ground.layers.append(("seg", cls, np.concatenate([pts[:-1], pts[1:]], 1), np.full(len(pts) - 1, width / 2)))
        cv2.polylines(self.gmap, [(self.px(pts) * 16).round().astype(np.int32)], False, int(cls),
                      max(1, int(round(width / GRID_CM))), cv2.LINE_8, 4)

    def area(self, poly, cls):
        poly = np.asarray(poly, float)
        if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
            poly = poly[::-1]
        self.ground.layers.append(("poly", cls, poly))
        cv2.fillConvexPoly(self.gmap, (self.px(poly) * 16).round().astype(np.int32), int(cls), cv2.LINE_8, 4)

    def disk(self, c, r, cls):
        self.ground.layers.append(("disk", cls, np.array([c[0], c[1], r], float)))
        cv2.circle(self.gmap, tuple((self.px(c) * 16).round().astype(int)), int(round(r / GRID_CM * 16)), int(cls), -1,
                   cv2.LINE_8, 4)

    # --- 轨迹相关 ---
    def axis(self):
        """玩家轨迹的主方向（PCA）：(中心, 单位方向, 沿主轴的半长)。"""
        P = self.path
        c = P.mean(0)
        ev, evec = np.linalg.eigh(np.cov((P - c).T) + 1e-6 * np.eye(2))
        u = evec[:, -1] if ev[-1] > 100.0 ** 2 else _dir(self.rng.uniform(0, 2 * np.pi))
        return c, u / np.linalg.norm(u), float(np.sqrt(max(ev[-1], 0)) * 2)

    def path_line(self, sigma=12.0, step=50.0):
        """平滑后的玩家轨迹（高斯 sigma 帧）按弧长重采样，两端沿切向延长到世界边界；轨迹太短 ⇒ PCA 直线。"""
        P = self.path
        if len(P) < 2 or np.linalg.norm(np.diff(P, axis=0), axis=1).sum() < 400:
            c, u, _ = self.axis()
            return self.line(c, u)
        x = np.arange(-3 * int(sigma), 3 * int(sigma) + 1)
        k = np.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
        Pp = np.pad(P, ((len(x) // 2,) * 2, (0, 0)), mode="edge")
        S = np.stack([np.convolve(Pp[:, i], k, mode="valid") for i in range(2)], 1)
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
        S = np.stack([np.interp(np.arange(0, L[-1], step), L, S[:, i]) for i in range(2)], 1)
        if len(S) < 4:
            c, u, _ = self.axis()
            return self.line(c, u)
        j = min(len(S) - 1, 6)
        t0, t1 = S[0] - S[j], S[-1] - S[-1 - j]
        t0, t1 = t0 / np.linalg.norm(t0), t1 / np.linalg.norm(t1)
        ext = np.linspace(step, self.span, 20)[:, None]
        return np.concatenate([(S[0] + ext * t0)[::-1], S, S[-1] + ext * t1])

    def line(self, p, u, wiggle=0.0, half=False, n=41):
        """过 p 沿 u 的整条（half ⇒ 只向 +u 的半条）折线，横向 wiggle cm 的正弦弯。"""
        s = np.linspace(0 if half else -self.span, self.span, n)
        lam, ph = self.rng.uniform(2500, 5000), self.rng.uniform(0, 2 * np.pi)
        lat = wiggle * (np.sin(2 * np.pi * s / lam + ph) - (math.sin(ph) if half else 0))
        return p + s[:, None] * u + lat[:, None] * _perp(u)


def walk(pts, s):
    """折线上弧长 s 处的 (点, 单位切向)；越界 ⇒ None。"""
    seg = np.diff(pts, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(L)]
    if s < 0 or s >= cum[-1]:
        return None
    i = int(np.searchsorted(cum, s, side="right") - 1)
    return pts[i] + seg[i] * (s - cum[i]) / L[i], seg[i] / L[i]


# ----------------------------------------------------------------------------
# 物体模板（返回 (prims, fp[, occ]) 给 World.add）
# ----------------------------------------------------------------------------
def tree(w, p, kind=None, scale=1.0):
    rng, z = w.rng, w.z0
    kind = kind or ("conifer" if rng.random() < 0.35 else "broad")
    if kind == "conifer":
        tr, th = rng.uniform(14, 24) * scale, rng.uniform(90, 170) * scale
        R0, n = rng.uniform(170, 250) * scale, int(rng.integers(2, 4))
        pr = [cyl(p, tr, z - 10, z + th + 40, VEG)]
        for i in range(n):
            R, h = R0 * (1 - 0.22 * i), rng.uniform(260, 340) * (1 - 0.1 * i) * scale
            zb = z + th + i * h * 0.5
            pr.append(prism(_ngon(p, R, 8, rng.uniform(0, 1)), zb, zb + h, VEG, taper=R / h))
        return pr, [_ngon(p, R0 + 10)], [_ngon(p, tr + 30, 8)]
    tr, th = rng.uniform(16, 32) * scale, rng.uniform(160, 320) * scale
    rc = rng.uniform(140, 250) * scale
    r = np.array([rc, rc * rng.uniform(0.8, 1.15), rc * rng.uniform(0.7, 1.0)])
    c = np.r_[p + rng.normal(0, 15, 2), z + th + 0.55 * r[2]]
    pr = [cyl(p, tr, z - 10, z + th + 0.6 * r[2], VEG), ell(c, r, rng.uniform(0, np.pi), VEG)]
    return pr, [_ngon(c[:2], r[:2].max() + 20)], [_ngon(p, tr + 30, 8)]


def log(w, p):
    """倒木：水平八棱柱（vegetation）。"""
    rng = w.rng
    L, r, yaw = rng.uniform(250, 500), rng.uniform(22, 38), rng.uniform(0, np.pi)
    a, e1, e2 = np.r_[_dir(yaw), 0.0], np.r_[_perp(_dir(yaw)), 0.0], np.array([0, 0, 1.0])
    C = np.r_[p, w.z0 + 0.8 * r]
    rows = [np.r_[n, n @ C + r] for n in (math.cos(th) * e1 + math.sin(th) * e2 for th in np.arange(8) * np.pi / 4)]
    rows += [np.r_[a, a @ C + L / 2], np.r_[-a, -a @ C + L / 2]]
    return [dict(kind="poly", cls=VEG, planes=_planes(rows))], [_rect(p, L, 2 * r + 10, yaw)]


def bush(w, p, big=1.0):
    r = np.array([w.rng.uniform(55, 130), 0, w.rng.uniform(40, 85)]) * big
    r[1] = r[0] * w.rng.uniform(0.7, 1.0)
    return [ell(np.r_[p, w.z0 + 0.35 * r[2]], r, w.rng.uniform(0, np.pi), VEG)], [_ngon(p, r[0] + 10)]


def rock(w, p, lo=60, hi=220):
    rx = w.rng.uniform(lo, hi)
    r = np.array([rx, rx * w.rng.uniform(0.6, 1.0), rx * w.rng.uniform(0.45, 0.8)])
    return [ell(np.r_[p, w.z0 + 0.1 * r[2]], r, w.rng.uniform(0, np.pi), ROCK)], [_ngon(p, rx + 10)]


def barrel_or_crate(w, p):
    if w.rng.random() < 0.5:
        r = w.rng.uniform(28, 40)
        return [cyl(p, r, w.z0 - 5, w.z0 + w.rng.uniform(80, 105), PROP)], [_ngon(p, r + 5)]
    s, yaw = w.rng.uniform(60, 100), w.rng.uniform(0, np.pi)
    return [box(p, s, s, s, yaw, w.z0 - 5, PROP)], [_rect(p, s + 10, s + 10, yaw)]


def prop_cluster(w, p, n=None):
    pr, fp = [], []
    for _ in range(n or int(w.rng.integers(2, 6))):
        q = p + w.rng.normal(0, 70, 2)
        if fp and min(np.linalg.norm(f.mean(0) - q) for f in fp) < 85:
            continue
        a, b = barrel_or_crate(w, q)
        pr += a; fp += b
    return pr, fp


def house(w, c, wid, dep, yaw, rng):
    z = w.z0
    h = rng.uniform(520, 680) if rng.random() < 0.25 else rng.uniform(270, 420)
    pr = [box(c, wid, dep, h, yaw, z - 20, ARCH)]
    if rng.random() < 0.8:                              # 山墙屋顶：屋脊沿长边
        ry = yaw if wid >= dep else yaw + np.pi / 2
        a, b = max(wid, dep), min(wid, dep)
        pr.append(gable(c, a, b, ry, z + h, rng.uniform(0.35, 0.6) * b, 40, ARCH))
    else:                                               # 平顶 + 女儿墙小体块
        pr.append(box(c + rng.normal(0, 60, 2), wid * 0.4, dep * 0.4, 120, yaw, z + h - 5, ARCH))
    return pr


def cliff(w, p, R, h):
    rng = w.rng
    a = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(7, 11))))
    pts = p + (R * rng.uniform(0.6, 1.0, len(a)))[:, None] * np.stack([np.cos(a), np.sin(a)], 1)
    hull = cv2.convexHull(pts.astype(np.float32))[:, 0].astype(float)
    return [prism(hull, w.z0 - 50, w.z0 + h, ROCK, taper=rng.uniform(0.08, 0.25))], [hull]


# ----------------------------------------------------------------------------
# 主题
# ----------------------------------------------------------------------------
def theme_village(w):
    rng = w.rng
    main = w.path_line()
    roads = [(main, rng.uniform(380, 520))]
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(50):
            q = walk(main, rng.uniform(0, 2 * w.span))
            if q is not None and w.at(w.vdist <= 600, q[0]):
                break
        else:
            continue
        side = _dir(math.atan2(q[1][1], q[1][0]) + rng.choice([-1, 1]) * (np.pi / 2 + rng.uniform(-0.4, 0.4)))
        roads.append((w.line(q[0], side, wiggle=rng.uniform(0, 200), half=True), rng.uniform(260, 400)))
        if rng.random() < 0.3:
            w.disk(q[0], rng.uniform(350, 550), PAVING)
    for pts, rw in roads:
        w.road(pts, rw)
    for pts, rw in roads:                               # 沿路两侧排房子，空档放矮墙
        for sd in (-1, 1):
            s = rng.uniform(0, 400)
            while (q := walk(pts, s)) is not None:
                pos, t = q
                wid, dep = rng.uniform(500, 900), rng.uniform(450, 750)
                ok = False
                if w.at(w.region, pos) and rng.random() < 0.85:
                    yaw = math.atan2(t[1], t[0])
                    n = _perp(t) * sd
                    hs = int(rng.integers(1 << 31))
                    for sb in (50, 120, 200, 300, 420, 560):          # 退线由近到远，贴着净空边界放
                        hc = pos + n * (rw / 2 + sb + dep / 2)
                        hp = house(w, hc, wid, dep, yaw, np.random.default_rng(hs))
                        ok = w.add(hp, [_rect(hc, wid + 100, dep + 100, yaw)],
                                   [_rect(hc, wid, dep, yaw)], gap=150, name="house")
                        if ok:
                            break
                    if ok:
                        for _ in range(int(rng.integers(0, 4))):     # 墙根的桶 / 箱子
                            ax = rng.choice([-1, 1]); f = rng.uniform(-0.4, 0.4)
                            loc = (np.array([f * wid, ax * (dep / 2 + 65)]) if rng.random() < 0.5
                                   else np.array([ax * (wid / 2 + 65), f * dep]))
                            pp = hc + loc[0] * t + loc[1] * _perp(t)
                            w.add(*barrel_or_crate(w, pp), gap=15, on_paving=True, name="prop")
                if not ok and w.at(w.region, pos) and rng.random() < 0.35:
                    yaw = math.atan2(t[1], t[0])
                    wc = pos + _perp(t) * sd * (rw / 2 + 70)
                    L = rng.uniform(300, 700)
                    w.add([box(wc, L, 35, rng.uniform(80, 130), yaw, w.z0 - 10, ARCH)], [_rect(wc, L, 35, yaw)],
                          gap=40, name="low_wall")
                s += wid + rng.uniform(200, 500)
    def shed(p):
        yaw, a, b, h = rng.uniform(0, np.pi), rng.uniform(220, 380), rng.uniform(200, 300), rng.uniform(200, 260)
        return ([box(p, a, b, h, yaw, w.z0 - 10, ARCH), gable(p, a, b, yaw, w.z0 + h, 0.45 * b, 25, ARCH)],
                [_rect(p, a + 50, b + 50, yaw)], [_rect(p, a, b, yaw)])
    w.scatter(shed, 4, 500, reach=300, gap=100, name="shed")
    w.scatter(lambda p: ([box(p, L := rng.uniform(300, 800), 20, rng.uniform(90, 120), yaw := rng.uniform(0, np.pi),
                              w.z0 - 10, ARCH)], [_rect(p, L, 20, yaw)]), 5, 400, reach=300, gap=50, name="fence")
    w.scatter(lambda p: ([cyl(p, r := rng.uniform(80, 100), w.z0 - 10, w.z0 + rng.uniform(80, 100), ARCH)], [_ngon(p, r + 5)]),
              1, 0, reach=300, gap=100, on_paving=True, name="well")
    w.scatter(lambda p: ([box(p, 200, 120, 100, yaw := rng.uniform(0, np.pi), w.z0, PROP)], [_rect(p, 200, 120, yaw)]),
              2, 500, reach=300, gap=60, on_paving=True, name="cart")
    w.scatter(lambda p: tree(w, p), 40, 420, reach=700, gap=60, name="tree")
    w.scatter(lambda p: bush(w, p), 30, 220, reach=300, gap=30, name="bush")
    w.scatter(lambda p: prop_cluster(w, p), 8, 500, reach=300, gap=30, on_paving=True, name="props")


def theme_forest(w):
    rng = w.rng
    if rng.random() < 0.5:                              # 林间石板小径（沿玩家轨迹）
        w.road(w.path_line(), rng.uniform(150, 220))
    w.scatter(lambda p: rock(w, p, 80, 240), 12, 500, reach=600, gap=40, name="rock")
    w.scatter(lambda p: tree(w, p), 140, rng.uniform(300, 380), reach=900, gap=60, name="tree")
    w.scatter(lambda p: tree(w, p, scale=rng.uniform(0.45, 0.7)), 40, 220, reach=300, gap=40, name="young_tree")
    w.scatter(lambda p: log(w, p), 5, 500, reach=300, gap=30, name="log")
    w.scatter(lambda p: bush(w, p, 1.2), 80, 180, reach=300, gap=20, name="bush")
    w.scatter(lambda p: rock(w, p, 30, 80), 30, 180, reach=300, gap=20, name="pebble")
    w.scatter(lambda p: ([cyl(p, r := rng.uniform(28, 45), w.z0 - 10, w.z0 + rng.uniform(35, 60), VEG)], [_ngon(p, r + 5)]),
              8, 400, reach=300, gap=30, name="stump")


def theme_plaza(w):
    rng = w.rng
    c, u, _ = w.axis()
    v = _perp(u)
    P = np.concatenate([w.path, w.boss_path]) - c
    pu, pv = P @ u, P @ v
    m = rng.uniform(150, 500, 4)
    u0, u1, v0, v1 = pu.min() - m[0], pu.max() + m[1], pv.min() - m[2], pv.max() + m[3]
    if u1 - u0 < 1800:                                  # 至少 18 m 见方
        u0, u1 = (u0 + u1) / 2 - 900, (u0 + u1) / 2 + 900
    if v1 - v0 < 1800:
        v0, v1 = (v0 + v1) / 2 - 900, (v0 + v1) / 2 + 900
    corner = lambda a, b: c + a * u + b * v
    rect = np.stack([corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)])
    w.area(rect, PAVING)
    yaw_u = math.atan2(u[1], u[0])
    edges = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]
    mids = [(a + b) / 2 for a, b in edges]
    for i in rng.permutation(4)[:int(rng.integers(2, 5))]:          # 出入口道路
        a, b = edges[i]
        out = _perp((b - a) / np.linalg.norm(b - a)) * -1          # CCW 矩形的外法向 = −perp
        w.road(w.line(mids[i], out, half=True, wiggle=rng.uniform(0, 150)), rng.uniform(400, 600))
    for (a, b) in edges:                                # 围墙（留门）或柱廊
        L = np.linalg.norm(b - a); t = (b - a) / L; out = -_perp(t); yaw = math.atan2(t[1], t[0])
        colon = rng.random() < 0.35
        s = rng.uniform(0, 300)
        hh = rng.uniform(220, 380)
        while s < L:
            if colon:
                p = a + t * s + out * 60
                r = rng.uniform(32, 45)
                w.add([cyl(p, r, w.z0 - 10, w.z0 + hh + 150, ARCH)], [_ngon(p, r + 5)], gap=40, on_paving=True,
                      name="column")
                s += rng.uniform(300, 400)
            else:
                sl = rng.uniform(400, 1000)
                p = a + t * min(s + sl / 2, L - sl / 2) + out * 50
                if rng.random() < 0.8:
                    w.add([box(p, sl, 50, hh, yaw, w.z0 - 20, ARCH)], [_rect(p, sl, 50, yaw)], gap=40, on_paving=True,
                          name="wall")
                s += sl + rng.uniform(200, 450)
    inside = lambda p: w.at(w.gmap == PAVING, p)

    def fountain(p):
        if not inside(p):
            return None
        R = rng.uniform(220, 320)
        return ([cyl(p, R, w.z0 - 10, w.z0 + 55, ARCH), cyl(p, rng.uniform(35, 50), w.z0, w.z0 + rng.uniform(180, 260), ARCH),
                 ell(np.r_[p, w.z0 + 230], [70, 70, 45], 0, ARCH)], [_ngon(p, R + 5)])
    w.scatter(fountain, 1, 0, tries=200, gap=200, on_paving=True, name="fountain")

    last = {}

    def planter(p):                                     # 种植池：四条矮池沿（architecture）+ 树 / 灌木，池里铺 terrain
        if not inside(p):
            return None
        s = rng.uniform(240, 360)
        o = tree(w, p, "broad") if rng.random() < 0.6 else bush(w, p, 1.3)
        ux, vx = _dir(yaw_u), _perp(_dir(yaw_u))
        curb = [box(p + e * (s / 2 - 12), s if i % 2 else 24, 24 if i % 2 else s, 45, yaw_u, w.z0 - 10, ARCH)
                for i, e in enumerate((ux, vx, -ux, -vx))]
        last["rect"] = _rect(p, s - 24, s - 24, yaw_u)
        return o[0] + curb, o[1] + [_rect(p, s, s, yaw_u)]
    def statue(p):                                      # 底座 + 抽象人像
        if not inside(p):
            return None
        s, hb = rng.uniform(100, 160), rng.uniform(80, 140)
        return ([box(p, s, s, hb, yaw_u, w.z0 - 10, ARCH), cyl(p, s * 0.22, w.z0 + hb, w.z0 + hb + 120, ARCH),
                 ell(np.r_[p, w.z0 + hb + 150], [s * 0.22] * 3, 0, ARCH)], [_rect(p, s, s, yaw_u)])
    w.scatter(statue, int(rng.integers(1, 4)), 600, reach=300, gap=100, on_paving=True, name="statue")
    w.scatter(lambda p: ([box(p, L := rng.uniform(250, 600), 45, rng.uniform(90, 250), yaw := yaw_u + rng.choice([0, np.pi / 2]),
                              w.z0 - 10, ARCH)], [_rect(p, L, 45, yaw)]) if inside(p) else None,
              int(rng.integers(2, 6)), 400, reach=300, gap=80, on_paving=True, name="ruin_wall")
    for _ in range(int(rng.integers(3, 7))):
        if w.scatter(planter, 1, 0, tries=100, reach=300, gap=100, on_paving=True, name="planter"):
            w.area(last["rect"], TERRAIN)

    def stall(p):
        if not inside(p):
            return None
        yaw = yaw_u + rng.choice([0, np.pi / 2]); a, b = rng.uniform(200, 300), rng.uniform(100, 140)
        d = _dir(yaw) * (a / 2 - 10)
        return ([box(p, a, b, 90, yaw, w.z0 - 5, PROP), box(p, a + 40, b + 60, 15, yaw, w.z0 + 220, PROP),
                 cyl(p + d, 7, w.z0, w.z0 + 225, PROP), cyl(p - d, 7, w.z0, w.z0 + 225, PROP)], [_rect(p, a + 40, b + 60, yaw)])
    w.scatter(stall, int(rng.integers(2, 5)), 400, reach=300, gap=80, on_paving=True, name="stall")
    w.scatter(lambda p: prop_cluster(w, p) if inside(p) else None, 14, 350, reach=300, gap=40, on_paving=True, name="props")
    w.scatter(lambda p: ([box(p, 160, 50, 45, yaw_u + rng.choice([0, np.pi / 2]), w.z0 - 5, PROP)], [_ngon(p, 85)])
              if inside(p) else None, 6, 300, reach=300, gap=60, on_paving=True, name="bench")
    w.scatter(lambda p: ([cyl(p, 10, w.z0 - 5, w.z0 + 330, PROP), box(p, 40, 40, 40, 0, w.z0 + 320, PROP)], [_ngon(p, 25)])
              if inside(p) else None, 6, 500, reach=300, gap=80, on_paving=True, name="lamp")
    w.scatter(lambda p: (([box(p, a := rng.uniform(500, 1000), b := rng.uniform(500, 900), rng.uniform(400, 800),
                                yaw_u, w.z0 - 20, ARCH)], [_rect(p, a, b, yaw_u)]) if not inside(p) else None),
              10, 900, gap=150, name="building")
    w.scatter(lambda p: tree(w, p), 15, 450, gap=60, name="tree")


def theme_canyon(w):
    rng = w.rng
    if rng.random() < 0.4:
        c, u, _ = w.axis()
        w.road(w.line(c + _perp(u) * rng.uniform(-300, 300), u, wiggle=rng.uniform(0, 300)), rng.uniform(300, 450))
    w.scatter(lambda p: cliff(w, p, rng.uniform(350, 900), rng.uniform(400, 1100)), 10, 1100, gap=100, name="cliff")
    w.scatter(lambda p: rock(w, p, 150, 420), 16, 500, reach=700, gap=60, name="boulder")
    w.scatter(lambda p: rock(w, p, 30, 90), 40, 120, reach=300, gap=15, name="pebble")

    def cactus(p):
        r, h = rng.uniform(14, 26), rng.uniform(120, 260)
        pr, fp = [cyl(p, r, w.z0 - 5, w.z0 + h, VEG)], [_ngon(p, r + 5)]
        for _ in range(int(rng.integers(0, 3))):
            d = _dir(rng.uniform(0, 2 * np.pi)); za = w.z0 + rng.uniform(0.35, 0.6) * h; q = p + d * (r + 30)
            pr += [box(p + d * (r + 15) / 2, r + 30, r * 1.2, r * 1.2, math.atan2(d[1], d[0]), za, VEG),
                   cyl(q, r * 0.7, za, za + rng.uniform(50, 90), VEG)]
            fp.append(_ngon(q, r * 0.7 + 5))
        return pr, fp
    w.scatter(cactus, 14, 300, reach=400, gap=40, name="cactus")
    w.scatter(lambda p: bush(w, p, 0.7), 20, 250, reach=300, gap=20, name="dry_bush")
    if rng.random() < 0.7:
        def camp(p):
            s, hh, yaw = rng.uniform(280, 360), rng.uniform(200, 260), rng.uniform(0, np.pi)
            pr = [prism(_rect(p, s, s, yaw), w.z0 - 5, w.z0 + hh, PROP, taper=s / 2 / hh)]
            q = p + _dir(yaw) * (s / 2 + 120)
            a, b = prop_cluster(w, q, 4)
            return pr + a, [_rect(p, s, s, yaw)] + b
        w.scatter(camp, int(rng.integers(1, 3)), 600, reach=400, gap=80, name="camp")


THEME_FN = dict(village=theme_village, forest=theme_forest, plaza=theme_plaza, canyon=theme_canyon)


# ----------------------------------------------------------------------------
# 渲染 / 自检 / 可视化
# ----------------------------------------------------------------------------
_YS, _XS = (a.ravel() for a in np.mgrid[0:H, 0:W])


def char_rays(meta, f):
    """帧 f 胶囊（不考虑场景）覆盖的像素与沿射线的胶囊 t：(ys, xs, raw, t_cap, ro, rd)。"""
    M, V = meta["world_to_clip"][f], meta["world_to_view"][f]
    caps = CAP.capsules_from_array(meta["capsules"][f])
    only = CAP.render_capsules(np.zeros((H, W), np.uint8), M, V, caps)
    ys, xs = np.nonzero(only)
    ro, rd = CAP.dst_pixel_rays(M, V, xs, ys)
    raw, t = CAP.nearest_capsule(ro, rd, caps)
    return ys, xs, raw, t, ro, rd


def control_occlusion(w: World, dev, cache) -> dict:
    """删掉帧 0 挡玩家的物体，以及帧 1..80 挡角色 > MAX_OCC 的帧里挡得最多的物体；迭代到不再删。"""
    removed = []
    for it in range(10):
        prims = w.prims()
        cs = Caster(prims, dev)
        kill = set()
        for f in range(F):
            ys, xs, raw, tcap, ro, rd = cache[f]
            if not len(ys):
                continue
            t, k = cs.cast(ro, rd)
            occ = t < tcap
            role = GM.collapse_parts(raw)
            for cid in (GEO.PLAYER_ID, GEO.BOSS_ID):
                sel = role == cid
                o = occ & sel
                if not o.any():
                    continue
                objs = np.array([prims[i]["obj"] for i in k[o]])
                if f == 0 and cid == GEO.PLAYER_ID:
                    kill |= set(objs.tolist())
                elif o.sum() > MAX_OCC * sel.sum():
                    vals, cnt = np.unique(objs, return_counts=True)
                    kill.add(int(vals[cnt.argmax()]))
        if not kill:
            break
        for gi in kill:
            w.objs[gi]["alive"] = False
            removed.append(w.objs[gi]["name"])
    return dict(iterations=it + 1, removed=removed)


def render(w: World, dev):
    """81 帧 → ids uint8 [81,H,W]，以及自检 / 可视化用的帧 0 与抽样帧的静态层（class, t_surface, prim 下标）。"""
    meta = w.src["meta"]
    prims = w.prims()
    cs = Caster(prims, dev)
    cls_of = np.array([p["cls"] for p in prims] or [SKY], np.uint8)
    ids = np.empty((F, H, W), np.uint8)
    keep = {}
    for f in range(F):
        M, V = meta["world_to_clip"][f], meta["world_to_view"][f]
        ro, rd = CAP.dst_pixel_rays(M, V, _XS, _YS)
        tp, k = cs.cast(ro, rd)
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = (w.z0 - ro[:, 2]) / rd[:, 2]
        tg = np.where(np.isfinite(tg) & (tg > 0), tg, np.inf)
        gcls = w.ground.classify(ro[:, :2] + rd[:, :2] * np.where(np.isfinite(tg), tg, 0)[:, None], dev)
        pcls = np.where(k >= 0, cls_of[np.maximum(k, 0)], SKY).astype(np.uint8)
        on_p = tp < tg
        static = np.where(on_p, pcls, np.where(np.isfinite(tg), gcls, SKY)).astype(np.uint8)
        caps = CAP.capsules_from_array(meta["capsules"][f])
        ids[f] = CAP.render_capsules(static.reshape(H, W), M, V, caps, static_depth=tp.reshape(H, W))
        if f % 20 == 0:
            keep[f] = dict(static=static.reshape(H, W), t=np.minimum(tp, tg).reshape(H, W),
                           sid=np.where(on_p, k, np.where(np.isfinite(tg), -2, -1)).reshape(H, W), ro=ro, rd=rd)
    return ids, keep


def consistency(w, ids, keep, rng, n=20000, tol=30.0) -> dict:
    """同一世界点 → 同一类：帧 f 的静态像素反投到世界（ro + t·rd），投到帧 g，两帧都是可见同一点（世界距离 < tol）时比类。"""
    meta, out = w.src["meta"], {}
    for f, g in ((0, 40), (40, 80), (0, 80), (20, 60)):
        A, B = keep[f], keep[g]
        cand = np.flatnonzero((ids[f].ravel() <= 6) & np.isfinite(A["t"].ravel()))
        sel = rng.choice(cand, min(n, len(cand)), replace=False)
        X = A["ro"][sel] + A["rd"][sel] * A["t"].ravel()[sel, None]
        d = GEO.src_to_dst(GEO.proj(meta["world_to_clip"][g].reshape(4, 4), X)).round().astype(int)
        ok = (d[:, 0] >= 0) & (d[:, 0] < W) & (d[:, 1] >= 0) & (d[:, 1] < H)
        sel, X, d = sel[ok], X[ok], d[ok]
        j = d[:, 1] * W + d[:, 0]
        Xg = B["ro"][j] + B["rd"][j] * np.where(np.isfinite(B["t"].ravel()[j]), B["t"].ravel()[j], 0)[:, None]
        vis = (np.linalg.norm(Xg - X, axis=1) < tol) & (ids[g].ravel()[j] <= 6)
        agree = A["static"].ravel()[sel[vis]] == B["static"].ravel()[j[vis]]
        out[f"{f}->{g}"] = dict(n_visible=int(vis.sum()), agree=round(float(agree.mean()), 5) if vis.any() else None)
    return out


def clearance(w) -> dict:
    """胶囊轴上采样点到图元的距离下界 − 胶囊半径（cm）；< 0 ⇒ 穿模。"""
    c = w.caps
    s = np.linspace(0, 1, 9)
    P = (c[:, None, :3] * (1 - s[:, None]) + c[:, None, 3:6] * s[:, None]).reshape(-1, 3)
    R = np.repeat(c[:, 6], len(s))
    best = np.full(len(P), np.inf)
    for p in w.prims():
        best = np.minimum(best, sdf_lower(p, P))
    cl = best - R
    return dict(min_clearance_cm=round(float(cl.min()), 1), n_penetrating_samples=int((cl < 0).sum()))


def char_color(ids, M, V, caps, img):
    """cond_v9 式角色着色（原地改 img）：最近胶囊法线 + 视空间固定光源 lambert，左肢 0.8 / 武器 0.88，按角色 TINT。"""
    m = (ids >= 10) & (ids != GEO.UNKNOWN_ID)
    if not m.any():
        return img
    ys, xs = np.nonzero(m)
    ro, rd = CAP.dst_pixel_rays(M, V, xs, ys)
    best, kk, nn = np.full(len(ro), np.inf), np.ones(len(ro)), np.zeros((len(ro), 3))
    for a, b, r, cls, part in caps:
        t = CAP.cap_hit(ro, rd, a, b, r)
        u = t < best
        if u.any():
            best[u] = t[u]; kk[u] = 0.88 if part == 5 else (0.80 if part in (1, 3) else 1.0)
            Pp = ro[u] + rd[u] * t[u, None]; ba = b - a
            hseg = np.clip(((Pp - a) @ ba) / max(ba @ ba, 1e-6), 0, 1)
            nv = Pp - a - hseg[:, None] * ba
            nn[u] = nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-6)
    V4 = np.asarray(V, float).reshape(4, 4)
    right, up, fwd = (V4[:3, i] / np.linalg.norm(V4[:3, i]) for i in range(3))
    L = -0.45 * right + 0.55 * up - 0.70 * fwd
    L /= np.linalg.norm(L)
    s = np.where(np.isfinite(best), (0.30 + 0.70 * np.clip(nn @ L, 0, 1)) * kk, 0.62)
    role = GM.collapse_parts(ids[ys, xs])
    for cid, tint in TINT.items():
        q = role == cid
        img[ys[q], xs[q]] = (np.clip(s[q, None] * 1.1, 0, 1) * tint * 255).astype(np.uint8)
    return img


def frame_rgb(w, ids, f):
    meta = w.src["meta"]
    img = GEO.colorize_ids(GM.collapse_parts(ids[f]))
    return char_color(ids[f], meta["world_to_clip"][f], meta["world_to_view"][f],
                      CAP.capsules_from_array(meta["capsules"][f]), img)


def shaded0(w, ids, k0):
    """帧 0 的体积感版本：静态色 × lambert（屏幕空间法线，同一表面的邻点差分）+ 表面边界描暗。"""
    meta = w.src["meta"]
    P = (k0["ro"] + k0["rd"] * np.where(np.isfinite(k0["t"].ravel()), k0["t"].ravel(), 0)[:, None]).reshape(H, W, 3)
    sid = k0["sid"]

    def diff(ax):
        fw, bw = np.roll(P, -1, ax) - P, P - np.roll(P, 1, ax)
        sf, sb = np.roll(sid, -1, ax) == sid, np.roll(sid, 1, ax) == sid
        return np.where(sf[..., None], fw, np.where(sb[..., None], bw, 0)), ~(sf & sb)
    dx, ex = diff(1)
    dy, ey = diff(0)
    n = np.cross(dx, dy)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n = np.where((n * k0["rd"].reshape(H, W, 3)).sum(-1, keepdims=True) > 0, -n, n)
    n[sid == -2] = (0, 0, 1)
    V4 = np.asarray(meta["world_to_view"][0], float).reshape(4, 4)
    right, up, fwd = (V4[:3, i] / np.linalg.norm(V4[:3, i]) for i in range(3))
    L = -0.45 * right + 0.55 * up - 0.70 * fwd
    L /= np.linalg.norm(L)
    sh = 0.45 + 0.55 * np.clip((n * L).sum(-1), 0, 1)
    sh[(ex | ey) & (sid != -2)] *= 0.7
    img = (GEO.PALETTE[k0["static"]] * sh[..., None]).clip(0, 255).astype(np.uint8)
    keepc = ids[0] >= 10
    img[keepc] = frame_rgb(w, ids, 0)[keepc]
    return img


def depth0(w, ids, k0) -> np.ndarray:
    meta = w.src["meta"]
    t = k0["t"].copy()
    ys, xs, raw, tc, _, _ = char_rays(meta, 0)
    drawn = ids[0][ys, xs] >= 10
    t[ys[drawn], xs[drawn]] = tc[drawn]
    fin = np.isfinite(t)
    lo, hi = np.percentile(t[fin], [0.5, 99.5]) if fin.any() else (0, 1)
    d = np.where(fin, 1 - (t - lo) / max(hi - lo, 1e-6), 0).clip(0, 1)
    return (d * 65535).round().astype(np.uint16)


def plan(w, dev, max_px=1400) -> np.ndarray:
    """俯视总图（+X 向右，+Y 向下）：图元按高度提亮；白 = 净空禁区边界，品红 / 红 = 玩家 / Boss 轨迹，黄 / 青 = 帧 0 / 80 视野。"""
    res = max(w.nx, w.ny) * GRID_CM / max_px
    nx, ny = int(w.nx * GRID_CM / res), int(w.ny * GRID_CM / res)
    yy, xx = np.mgrid[0:ny, 0:nx]
    xy = np.stack([w.x0 + (xx.ravel() + 0.5) * res, w.y0 + (yy.ravel() + 0.5) * res], 1)
    ro = np.c_[xy, np.full(len(xy), w.z0 + 5000.0)]
    rd = np.tile(np.array([3e-4, 2e-4, -1.0]) / np.linalg.norm([3e-4, 2e-4, -1.0]), (len(xy), 1))
    prims = w.prims()
    t, k = Caster(prims, dev).cast(ro, rd)
    cls = w.ground.classify(xy, dev)
    hit = k >= 0
    cls[hit] = [prims[i]["cls"] for i in k[hit]]
    hgt = np.where(hit, 5000.0 - t, 0)
    img = GEO.PALETTE[cls].astype(np.float32) * (0.75 + 0.25 * np.clip(hgt / 800, 0, 1))[:, None]
    img = img.reshape(ny, nx, 3).clip(0, 255).astype(np.uint8)[..., ::-1].copy()      # → BGR 画线
    sc = GRID_CM / res
    q = lambda P: ((w.px(P) + 0.5) * sc * 16).round().astype(np.int32)
    cs, _ = cv2.findContours(cv2.resize(w.blocked.astype(np.uint8), (nx, ny), interpolation=cv2.INTER_NEAREST),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (255, 255, 255), 1)
    for P, col in ((w.boss_path, (0, 0, 255)), (w.path, (255, 0, 255))):
        if len(P) > 1:
            cv2.polylines(img, [q(P)], False, col, 2, cv2.LINE_AA, 4)
    for g, col in ((w.foot[0], (0, 255, 255)), (w.foot[-1], (255, 255, 0))):
        cv2.polylines(img, [q(g)], True, col, 2, cv2.LINE_AA, 4)
    return img


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------
def build_case(clip_id: str, theme: str, seed: int, case: str | None = None, out_root: str = OUT, dev="cuda",
               video: bool = True) -> dict:
    import viz
    t_all = time.time()
    case = case or f"{theme}__{clip_id.split('__', 1)[1]}"
    od = os.path.join(out_root, case)
    os.makedirs(od, exist_ok=True)
    src = load_source(clip_id)
    meta = src["meta"]
    w = World(src, seed)
    THEME_FN[theme](w)
    n_built = sum(o["alive"] for o in w.objs)
    t0 = time.time()
    cache = [char_rays(meta, f) for f in range(F)]
    occ_ctl = control_occlusion(w, dev, cache)
    t_occ = time.time() - t0
    t0 = time.time()
    ids, keep = render(w, dev)
    t_render = time.time() - t0
    assert set(np.unique(ids).tolist()) <= GM.valid_raw_ids("parts"), np.unique(ids)
    # --- 自检 ---
    prims = w.prims()
    occ_stats = {}
    for cid, nm in ((GEO.PLAYER_ID, "player"), (GEO.BOSS_ID, "boss")):
        fr = []
        for f in range(F):
            ys, xs, raw, *_ = cache[f]
            sel = GM.collapse_parts(raw) == cid
            if sel.any():
                fr.append(1 - (GM.collapse_parts(ids[f][ys[sel], xs[sel]]) == cid).mean())
        occ_stats[nm] = dict(frames_visible=len(fr), occ_mean=round(float(np.mean(fr)), 4) if fr else None,
                             occ_max=round(float(np.max(fr)), 4) if fr else None)
    coll = GM.collapse_parts(ids)
    player_px0 = int((coll[0] == GEO.PLAYER_ID).sum())
    hist = np.bincount(coll.ravel(), minlength=12)[:12] / coll.size
    calf = meta["capsules"][:, :, [8, 10]]                                  # l_calf / r_calf
    feet = (np.minimum(calf[..., 2], calf[..., 5]) - calf[..., 6]).min(-1) - w.z0  # [F, 2] 脚底相对地面（cm）
    rep = dict(
        case=case, clip_id=clip_id, theme=theme, seed=seed, subset=src["row"]["subset"], ortho=meta["ortho"],
        z_ground=round(w.z0, 2), n_objects=n_built, n_objects_kept=sum(o["alive"] for o in w.objs),
        n_prims=len(prims), prims_by_class={str(int(k)): int(v) for k, v in
                                            zip(*np.unique([p["cls"] for p in prims], return_counts=True))} if prims else {},
        occlusion_control=occ_ctl, player_px_f0=player_px0, boss_px_f0=int((coll[0] == GEO.BOSS_ID).sum()),
        occlusion=occ_stats, **clearance(w), consistency=consistency(w, ids, keep, np.random.default_rng(0)),
        class_frac={n: round(float(hist[i]), 4) for i, n in enumerate(CLASS_NAMES) if hist[i] > 0}
        | {"player": round(float(hist[10]), 4), "boss": round(float(hist[11]), 4)},
        feet_vs_plane_cm={nm: ([round(float(v), 1) for v in np.nanpercentile(feet[:, i], [5, 50, 95])]
                               if np.isfinite(feet[:, i]).any() else None) for i, nm in enumerate(("player", "boss"))},
        world_extent_m=[round(w.nx * GRID_CM / 100, 1), round(w.ny * GRID_CM / 100, 1)],
        objects_by_name=dict(sorted(collections.Counter(o["name"] for o in w.objs if o["alive"]).items())),
        sec_occlusion_control=round(t_occ, 2), sec_render_81=round(t_render, 2),
        gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None)
    rep["checks_ok"] = bool(player_px0 >= 500 * GEO.AREA_VS_480 and rep["n_penetrating_samples"] == 0 and
                            all(v["agree"] is None or v["agree"] > 0.98 for v in rep["consistency"].values()))
    # --- 输出 ---
    np.savez_compressed(os.path.join(od, "ids.npz"), ids=ids)
    json.dump(src["prompts"], open(os.path.join(od, "prompts.json"), "w"), indent=1)
    np.save(os.path.join(od, "actions.npy"), src["blocks"][1:])
    row = src["row"]
    json.dump(dict(clip_id=clip_id, subset=row["subset"], scene=row["scene"], run=row["run"], start=row["start"],
                   theme=theme, theme_desc=THEME_DESC[theme], seed=seed, ortho=meta["ortho"], z_ground_cm=w.z0,
                   legend={n: GEO.PALETTE[i].tolist() for i, n in enumerate(CLASS_NAMES)} | {
                       "player (map0: cond_v9 tint)": (TINT[10] * 255).round().astype(int).tolist(),
                       "boss (map0: cond_v9 tint)": (TINT[11] * 255).round().astype(int).tolist()},
                   actions_block=src["blocks"].tolist(),
                   world_origin_cm=[float(w.x0), float(w.y0)], grid_cm=GRID_CM, clear_cm=CLEAR_CM,
                   reach_cm=REACH_CM, max_occ=MAX_OCC, generator="code/gc25d/procmap.py"),
              open(os.path.join(od, "source.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(od, "report.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(od, "map0.png"), frame_rgb(w, ids, 0)[..., ::-1])
    cv2.imwrite(os.path.join(od, "map0_shaded.png"), shaded0(w, ids, keep[0])[..., ::-1])
    cv2.imwrite(os.path.join(od, "depth0.png"), depth0(w, ids, keep[0]))
    cv2.imwrite(os.path.join(od, "plan.png"), plan(w, dev))
    tiles = [label(cv2.resize(frame_rgb(w, ids, f), (W // 2, H // 2), interpolation=cv2.INTER_AREA), f"f{f}")
             for f in (0, 20, 40, 60, 80)]
    cv2.imwrite(os.path.join(od, "contact.jpg"), np.concatenate(tiles, 1)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if video:
        viz.write_mp4((label(frame_rgb(w, ids, f), f"{case}  f{f}") for f in range(F)), os.path.join(od, "preview.mp4"))
    rep["sec_total"] = round(time.time() - t_all, 2)
    json.dump(rep, open(os.path.join(od, "report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("case", "n_prims", "player_px_f0", "min_clearance_cm", "consistency",
                                          "occlusion", "sec_render_81", "sec_total", "checks_ok")}), flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--default4", action="store_true", help="4 个基准 case（CASES）")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--theme", choices=THEMES, default="village")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args(argv)
    todo = CASES if a.default4 else [(a.clip, a.theme, a.seed)]
    assert all(c for c, _, _ in todo), "需要 --clip 或 --default4"
    reps = [build_case(c, th, s, None if a.default4 else a.case, a.out, a.device, not a.no_video) for c, th, s in todo]
    if a.default4:
        json.dump(reps, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
