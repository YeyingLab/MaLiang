#!/usr/bin/env python3
"""gc25d 泛化测试的首帧服务：FLUX.2 klein（默认 9B）/ FLUX.3 klein 生成 W×H RGB 首帧（TI2V 的 cell 0 锚点）。
输出 W×H = 训练分辨率（env GC_RES，默认 832×480；规则与 geometry.parse_res 相同，这里抄一份 —— geometry 要 cv2）。

两种模式
--------
* restyle   ：真实帧 0 + 风格 ⇒ 同布局 / 同角色 / 同相机，换外观（L2：见过的地图 + 没见过的首帧风格）。
              klein 多参考编辑，image 1 = 原帧，纯噪声起步（strength 1）；短风格名经 STYLE_PRESETS 展开成具体描述。
* map2frame ：语义图（+ 可选风格参考帧）⇒ 照图布局的 2.5D 等距 ARPG 截图（L3：没见过的地图 + 没见过的首帧）。
              输入先还原成 parts raw id（ids_from_map：认 geometry.colorize_ids / sample.colorize_parts / 本模块色，
              也收单通道 raw id png / cond npz），再按 FLUX_PALETTES 重画给 FLUX 看：静态类 = geometry.PALETTE 的自然色，
              玩家身体蓝、Boss 身体红、武器白，铺地叠 16 px 暗网格（colorize_ids 的玩家是品红、部位是白，FLUX 看不懂），
              提示词里只写图中出现的类与角色（map_legend）。best_of 个种子按 layout_score（只用 ids）留最好的。

默认配方（sweep：outputs/genbench/flux_examples/sweep/，6 条 val clip 帧 0，对照真实帧 0 + layout_metrics + 目测）：
klein-9B、4 步、map 提示词 v4、best_of 4、无风格参考；restyle 提示词 v1 + STYLE_PRESETS。
* klein-4B：布局同样好、细节差一点。FLUX.3 klein（16 步、不跑 PUP）：map2frame 画面更写实但发糊，
  大片铺地常画成泥地；restyle 只做了调色（raw 提示词太弱，PCE 的笔记里也是这个结论）⇒ 不用。
* strength < 1 从色图起步（--init map）会照抄平涂色；8 步不比 4 步好；风格参考帧（--ref）对布局无系统增益。
实测耗时与显存因硬件而异，此处不列。

生成在 W_GEN×H_GEN 上做（gen_grid：长宽比精确 = W:H、16 的倍数、面积最接近 0.9 MP 且不小于输出；832×480 ⇒ 1248×720
= 精确 1.5 倍，1280×704 ⇒ 1280×704 本身），参考图也放到同一网格（语义图最近邻放大），再用 INTER_AREA 缩回 W×H
（area_resize：numpy 实现，与 cv2.INTER_AREA 逐像素相同；PCE 的 env 里没有 cv2；同尺寸 ⇒ 原样）。

运行环境：PCE 的 env（只读复用，不改 PCE）pce_f3（diffusers main，同时有 Flux2KleinPipeline 与 FLUX.3 modular
pipeline）；权重 = PCE/weights/hf_hub 的本地 snapshot，离线加载。maliang venv 没有 diffusers ⇒ 别的组件用子进程调 CLI。

    PY=python; F=code/gc25d/firstframe_flux.py   # needs a diffusers/FLUX environment
    CUDA_VISIBLE_DEVICES=3 $PY $F restyle --image f0.png --style "snowy winter night" --out a.png
    CUDA_VISIBLE_DEVICES=3 $PY $F restyle --clip <clip_id> --style "watercolor painting" --out b.png
    CUDA_VISIBLE_DEVICES=3 $PY $F map2frame --map ids_or_map.png [--ref style.png] [--prompt "..."] --out c.png
    CUDA_VISIBLE_DEVICES=3 $PY $F batch --jobs jobs.json                 # 模型只加载一次
jobs.json = [{"mode": "restyle"|"map2frame", "out": ..., 其余键同 CLI 参数名（image/clip/frame/style/map/ref/prompt/
seed/steps/strength/version/best_of/save_map）}, ...]。每张图旁边写 <out>.json（提示词、设置、秒数、显存峰值、
best_of 候选分；有 ids 时附 layout_metrics，restyle 附 struct_metrics）。

Python API：FirstFrameFlux(model).restyle(rgb, style) / .map2frame(ids, ref=None, prompt="") → uint8 [H, W, 3]。
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
import io
import json
import math
import os
import sys
import tarfile
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_ROOT = os.path.join(ROOT, "data", "gc25d_v10")
RAW_ROOT = os.path.join(ROOT, "data", "recordings", "data")
HUB = _mos.environ.get("HF_HUB_CACHE", _mos.path.expanduser("~/.cache/huggingface/hub"))
MODELS = {  # name → (backend, repo, revision)；klein-base-9B 在 hub 里没有 snapshot（未下载完），不列
    "klein9b": ("flux2", "black-forest-labs/FLUX.2-klein-9B", "92196c8e11f7b6cf2b7493e037d8c5345c559216"),
    "klein4b": ("flux2", "black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294"),
    "flux3": ("flux3", "black-forest-labs/flux-3-image-klein", "337337621b8526e120f2275d250242f4a6bedd75"),
}
DEFAULT_STEPS = {"flux2": 4, "flux3": 16}
# ------------------------------------------------------------------ 分辨率 ----
# 与 geometry.parse_res / CROP_W / CROP_X0 同一套规则（tests/test_res.py 在 maliang venv 里逐项核对）
SRC_W, SRC_H = 1280, 704
GEN_AREA = 0.9e6                                  # FLUX 生成网格的目标面积（klein 的甜区 ≈ 1 MP）


def parse_res(s: str) -> tuple[int, int]:
    """"<W>x<H>" → (W, H)（= geometry.parse_res）。"""
    try:
        w, h = (int(v) for v in str(s).strip().lower().split("x"))
    except ValueError:
        raise ValueError(f"GC_RES={s!r}：应形如 '832x480' / '1280x704'") from None
    if w <= 0 or h <= 0 or w % 32 or h % 32 or h > SRC_H or w > SRC_W:
        raise ValueError(f"GC_RES={s!r}：W、H 须为 32 的正倍数且 H ≤ {SRC_H}、W ≤ {SRC_W}")
    if w * SRC_H > h * SRC_W:
        raise ValueError(f"GC_RES={s!r}：宽高比 {w / h:.4f} 超过源 {SRC_W}:{SRC_H}（需要竖直裁剪，未实现）")
    if 2 * h <= SRC_H or 2 * w <= crop_w_for(w, h):     # 缩小倍数 ≥ 2（H < 384）：geometry 的 _IDX 表盖不到首末行 / 列
        raise ValueError(f"GC_RES={s!r}：H 须 ≥ {SRC_H // 2 + 32}（缩小倍数 < 2）")
    return w, h


def crop_w_for(w: int, h: int) -> int:
    """= geometry.crop_w_for：704·W/H 取最近的偶数（≤ 1280），裁剪关于源中心对称。"""
    return min(SRC_W, 2 * round(SRC_H * w / (2 * h)))


def gen_grid(w: int, h: int, area: float = GEN_AREA) -> tuple[int, int]:
    """输出 w×h → FLUX 生成网格 (16·a·t, 16·b·t)，a:b = w:h 的最简比 ⇒ 长宽比精确、16 的倍数；t 取面积最接近 area
    （对数距离）且网格不小于输出的那个（area_resize 只做缩小）。832×480 ⇒ 1248×720，1280×704 ⇒ 1280×704。"""
    g = math.gcd(w, h)
    a, b = w // g, h // g
    ts = [t for t in range(1, 64) if 16 * a * t >= w and 16 * b * t >= h]
    t = min(ts, key=lambda t: abs(math.log(256 * a * b * t * t / area)))
    return 16 * a * t, 16 * b * t


W_OUT, H_OUT = parse_res(os.environ.get("GC_RES") or "832x480")
RES_TAG = f"{W_OUT}x{H_OUT}"
CROP_W = crop_w_for(W_OUT, H_OUT)                  # = geometry.CROP_W（832×480 ⇒ 1220）
CROP_X0 = (SRC_W - CROP_W) // 2                   # = geometry.CROP_X0（832×480 ⇒ 30）
W_GEN, H_GEN = gen_grid(W_OUT, H_OUT)

# ------------------------------------------------------------------ 调色板 ----
# geometry.PALETTE 的静态类 / 角色 core 与 sample.PART_COLORS 的部位色（这两个模块要 cv2 / torch+DiffSynth，PCE env
# 里 import 不了，所以抄一份；tests/test_firstframe_flux.py 在 maliang venv 里核对与原表逐字一致）。
GEO_COLORS = {0: (135, 190, 235), 1: (140, 110, 70), 2: (200, 200, 190), 3: (200, 90, 60), 4: (110, 110, 120),
              5: (60, 160, 60), 6: (230, 190, 40), 10: (255, 0, 255), 11: (255, 0, 0)}
PART_COLORS = {12: (255, 150, 255), 13: (150, 60, 255), 14: (255, 70, 160), 15: (110, 40, 190), 16: (0, 255, 255),
               17: (255, 150, 110), 18: (255, 120, 0), 19: (200, 0, 70), 20: (130, 20, 20), 21: (0, 90, 255)}
UNKNOWN_ID, HATCH_GREY = 255, (96, 96, 96)
PLAYER_PARTS, BOSS_PARTS = (10, 12, 13, 14, 15, 16), (11, 17, 18, 19, 20, 21)
VALID_IDS = frozenset(range(7)) | set(PLAYER_PARTS) | set(BOSS_PARTS) | {UNKNOWN_ID}

# FLUX 看的语义图：静态类 = geometry 的自然色；玩家身体蓝、Boss 身体红；武器 v1/v2 = 青 / 橙，v3 起 = 白（彩色武器会被画成
# 发光光剑）；UNKNOWN 纯黑（不画斜纹）
def _flux_palette(weapon_player, weapon_boss) -> np.ndarray:
    pal = np.zeros((256, 3), np.uint8)
    for k, c in GEO_COLORS.items():
        pal[k] = c
    pal[list(PLAYER_PARTS[:-1])], pal[16] = (30, 80, 255), weapon_player
    pal[list(BOSS_PARTS[:-1])], pal[21] = (255, 0, 0), weapon_boss
    return pal


FLUX_PALETTES = {"color": _flux_palette((0, 230, 255), (255, 140, 0)),
                 "white": _flux_palette((245, 245, 245), (245, 245, 245))}
LEGEND_V1 = ("Color legend of the layout map: brown = natural ground (dirt, soil, sand, grass or snow); "
             "light grey = paved floor (stone tiles, road, plaza, wooden deck); orange-red = architecture (building "
             "walls, roofs, stairs, pillars, bridges, towers); dark grey = rocks and boulders; green = vegetation "
             "(trees, bushes, grass tufts, flowers); yellow = props and objects (crates, barrels, carts, tables, "
             "fences, lamps, statues); sky blue = open sky or void; black = unknown, fill it with scenery that "
             "continues the "
             "surroundings. The blue humanoid figure is the player character, a small armored hero, holding a sword "
             "(cyan stroke); the red humanoid figure is the boss, a monster, holding a weapon (orange stroke).")
# 动态色表：只写图里出现的类（v1 把没有的 Boss 也写进去 ⇒ klein 会把蓝色玩家画成红色怪物 / 另加小人）
CLASS_TEXT = {
    0: "sky blue = open sky or empty void",
    1: "brown = natural open ground (bare soil, dirt, sand, grass or snow)",
    2: "light grey = paved floor (stone tiles, cobblestone, road, plaza or wooden deck)",
    3: "orange-red = architecture (building walls, roofs, stairs, pillars, bridges, towers)",
    4: "dark grey = rocks and boulders",
    5: "green = vegetation (trees, bushes, grass tufts, flowers)",
    6: "yellow = props and objects (crates, barrels, carts, tables, fences, lamps, statues)",
    UNKNOWN_ID: "black = unknown area, fill it with scenery that continues the surroundings",
}
BROWN_STRICT = "; keep it open ground, no paths or floors"     # v2：压住 B 里凭空多出的铺地（副作用：A 的广场变泥地）
MIN_PX = round(40 * W_OUT * H_OUT / (832 * 480))   # 少于这么多像素的类 / 角色不写进色表（832×480 上 40，按面积缩放）


def map_legend(ids: np.ndarray, version: str = "v4") -> str:
    """图里实际出现的类 / 角色的色表句子（versions 见 MAP_PROMPTS 上方的注释）。"""
    n = np.bincount(np.asarray(ids).reshape(-1), minlength=256)
    ct = dict(CLASS_TEXT)
    if version == "v2":
        ct[1] += BROWN_STRICT
    if version == "v4":
        ct[2] = "light grey with a dark tile grid = paved floor (stone tiles, flagstones, cobblestone, plaza or road)"
    if version == "v3h":                                       # 色名后附 hex（没看出差别）
        ct = {k: t.replace(" = ", f" (#{bytes(GEO_COLORS[k]).hex().upper()}) = ", 1) if k in GEO_COLORS else t
              for k, t in ct.items()}
    txt = "Color legend of the layout map: " + "; ".join(t for k, t in ct.items() if n[k] >= MIN_PX) + "."
    has_p, has_b = n[list(PLAYER_PARTS)].sum() >= MIN_PX, n[list(BOSS_PARTS)].sum() >= MIN_PX
    wp, wb = ("cyan", "orange") if version == "v2" else ("white", "white")
    if has_p:
        txt += (" The blue humanoid figure is the player character, an armored human warrior"
                + (f" holding a sword (the {wp} stroke)" if n[16] else "")
                + "; draw him exactly where the blue figure is, at the same size and in the same pose.")
    if has_b:
        txt += (" The red humanoid figure is the boss, a hulking monster"
                + (f" wielding a weapon (the {wb} stroke)" if n[21] else "")
                + "; draw it exactly where the red figure is, at the same size and in the same pose.")
    if has_p or has_b:
        txt += (" These are the only characters; do not add any other people or creatures."
                if has_p and has_b else " This is the only character; do not add any other people or creatures.")
        if version != "v2":
            txt += (" The flat blue and red colors only mark who is who: give the characters natural, realistic "
                    "colors and materials, and render weapons as ordinary steel or wood, not glowing.")
    else:
        txt += " There are no characters in this scene."
    return txt


# 提示词版本（sweep 见 outputs/genbench/flux_examples/sweep，klein-9B 4 步，6 条 val clip 帧 0）：
#   v1  静态色表（无 Boss 也写 Boss、"small hero"）：蓝色玩家常被画成红色怪物，棕色地面上凭空铺石板
#   v2  动态色表 + "nothing is added" + 棕地 BROWN_STRICT：角色对了，但大片浅灰广场变泥地
#   v3  v2 去掉 BROWN_STRICT，武器白色 + "颜色只是标记"：角色自然、不再是光剑；广场仍常变泥地
#   v3h v3 + 色名 hex：无可见差别
#   v4  v3 + 铺地画 16 px 暗网格（默认）：广场 / 地砖稳定画成铺地，其余同 v3
MAP_PROMPTS = {
    "v1": ("Turn image 1, a flat color-coded layout map, into a finished screenshot of a 3D action RPG video game "
           "seen from a high top-down isometric camera. Follow the layout of image 1 exactly: every region, object "
           "and character keeps its exact position, shape and size. {legend} Replace every flat color with detailed "
           "textured materials, natural lighting and soft shadows; no flat colors, no outlines, no text, no UI."),
    "v2": ("Turn image 1, a flat color-coded layout map, into a finished screenshot of a 3D action RPG video game "
           "seen from a high top-down isometric camera. Follow the layout of image 1 exactly: every region, object "
           "and character keeps its exact position, shape and size, and nothing is added that is not in the map. "
           "{legend} Replace every flat color with detailed textured materials, natural lighting and soft shadows; "
           "no flat colors, no outlines, no text, no UI."),
}
MAP_PROMPTS["v3"] = MAP_PROMPTS["v3h"] = MAP_PROMPTS["v4"] = MAP_PROMPTS["v2"]
MAP_STYLE_REFS = {  # v3 / v3h / v4 用 v2 的句子
    "v1": (" Match the art style, rendering, color palette and lighting of image 2, but take nothing of its "
           "layout or content."),
    "v2": (" Image 2 is another screenshot of the same game: render image 1 in exactly the same art style, "
           "rendering technique, textures, color palette and lighting as image 2, but copy nothing of its layout, "
           "objects or characters."),
}
MAP_STYLE_REFS["v3"] = MAP_STYLE_REFS["v3h"] = MAP_STYLE_REFS["v4"] = MAP_STYLE_REFS["v2"]
MAP_PALETTE = {"v1": "color", "v2": "color", "v3": "white", "v3h": "white", "v4": "white"}
DEFAULT_MAP_PROMPT = "v4"
DEFAULT_BEST_OF = 4
RESTYLE_PROMPTS = {  # v2（"Completely restyle"）与 v1 几乎一样；风格强度靠 STYLE_PRESETS 的具体描述
    "v1": ("Restyle this top-down isometric action RPG game screenshot as {style}. Keep the exact same composition, "
           "camera angle and layout: the ground, paths, rocks, plants, buildings and objects stay where they are with "
           "the same shapes, and the characters keep exactly the same positions, sizes, poses and weapons. Change only "
           "the look: materials, colors, lighting, weather and rendering style."),
    "v2": ("Completely restyle this top-down isometric action RPG game screenshot: the whole image must clearly look "
           "like {style}. Keep the exact same composition, camera angle and layout: the ground, paths, rocks, plants, "
           "buildings and objects stay where they are with the same shapes, and the characters keep exactly the same "
           "positions, sizes, poses and weapons. Transform everything else: materials, colors, lighting, weather, "
           "atmosphere and rendering style."),
}
DEFAULT_RESTYLE_PROMPT = "v1"
# 短风格名 → 具体描述（klein 的编辑偏保守：只写 "pixel-art retro game" 时几乎不像素化）；不在表里的 --style 原样用
STYLE_PRESETS = {
    "snowy winter night": "a snowy winter night: thick fresh snow covering the ground, roofs and plants, cold blue "
                          "moonlight, warm lantern glow and falling snowflakes",
    "watercolor painting": "a hand-painted watercolor illustration with soft translucent washes, visible paper "
                           "texture, loose brush strokes and paint bleeding at the edges",
    "pixel-art retro game": "a 16-bit pixel-art retro video game with large chunky clearly visible square pixels, a "
                            "small limited color palette, dithering and no smooth gradients",
    "neon cyberpunk night": "a neon cyberpunk night: dark wet reflective surfaces lit by glowing magenta and cyan neon "
                            "signs and light strips, haze and rain",
    "lush tropical jungle": "a lush tropical jungle: dense bright green foliage, ferns, moss and hanging vines growing "
                            "over everything, humid haze and dappled sunlight",
}


def _palette_table():
    """(colors [K,3] int, ids [K]) of every colour a colorized map can contain (geometry + parts + flux + 255)."""
    cols, ids = [], []
    for k, c in list(GEO_COLORS.items()) + list(PART_COLORS.items()):
        cols.append(c), ids.append(k)
    for pal in FLUX_PALETTES.values():
        for k in sorted(set(PLAYER_PARTS) | set(BOSS_PARTS)):
            cols.append(tuple(pal[k])), ids.append(k)
    cols += [(0, 0, 0), HATCH_GREY]
    ids += [UNKNOWN_ID, UNKNOWN_ID]
    return np.asarray(cols, np.int32), np.asarray(ids, np.uint8)


def ids_from_map(img: np.ndarray) -> np.ndarray:
    """colorized map（RGB uint8 [H,W,3]：geometry.colorize_ids / sample.colorize_parts / FLUX_PALETTES）或单通道 raw id
    → parts raw id uint8 [H,W]。逐像素最近调色板色。两种色只认得出角色、认不出是谁：colorize_ids 的部位 12..21 是纯白，
    FLUX "white" 色的两把武器都是 245 灰白 ⇒ 按连通块（块里只有一种 core 就归它，否则按到 core 的距离）归玩家 / Boss，
    纯白 → 该角色 core（geometry 色本来就分不出部位），245 → 该角色的武器 16 / 21。"""
    img = np.asarray(img)
    if img.ndim == 2:
        bad = sorted(set(np.unique(img).tolist()) - VALID_IDS)
        if bad:
            raise ValueError(f"单通道 id 图含 parts 词表外的值 {bad[:8]}")
        return img.astype(np.uint8)
    cols, ids = _palette_table()
    flat = img[..., :3].reshape(-1, 3).astype(np.int32)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    d = ((uniq[:, None, :] - cols[None]) ** 2).sum(-1)
    out = ids[d.argmin(1)][inv.reshape(-1)].reshape(img.shape[:2])
    geo_w = (uniq == 255).all(1)[inv.reshape(-1)].reshape(img.shape[:2])
    flux_w = (uniq == FLUX_PALETTES["white"][16]).all(1)[inv.reshape(-1)].reshape(img.shape[:2])
    wm = geo_w | flux_w
    if wm.any():
        from scipy import ndimage
        dist = [ndimage.distance_transform_edt(out != c) if (out == c).any() else np.full(out.shape, np.inf)
                for c in (10, 11)]
        fill = np.where(dist[1] < dist[0], 11, 10).astype(np.uint8)
        comp, n = ndimage.label(wm | (out == 10) | (out == 11))
        for k in range(1, n + 1):
            cores = set(np.unique(out[(comp == k) & ~wm]).tolist()) & {10, 11}
            if len(cores) == 1:
                fill[comp == k] = cores.pop()
        out[geo_w] = fill[geo_w]
        out[flux_w] = np.where(fill[flux_w] == 11, 21, 16)
    return out


TILE = 16                                          # v4：铺地画一层 16 px 的暗网格（不加时 klein / flux3 常把大片浅灰铺地画成泥地）


def flux_map(ids: np.ndarray, palette: str = "white", tiles: bool = False) -> np.ndarray:
    ids = np.asarray(ids, np.uint8)
    out = FLUX_PALETTES[palette][ids]
    if tiles:
        yy, xx = np.mgrid[0:ids.shape[0], 0:ids.shape[1]]
        m = (ids == 2) & ((yy % TILE == 0) | (xx % TILE == 0))
        out[m] = (out[m] * 0.72).astype(np.uint8)
    return out


# ------------------------------------------------------------------ 图像工具 ----
def _area_w(n_src: int, n_dst: int) -> np.ndarray:
    s = n_src / n_dst
    e = np.arange(n_dst + 1) * s
    j = np.arange(n_src)
    return (np.clip(np.minimum(e[1:, None], j + 1) - np.maximum(e[:-1, None], j), 0, None) / s).astype(np.float32)


def area_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """uint8 [H,W,3] 缩小到 [h,w,3]，面积加权（与 cv2.INTER_AREA 逐像素相同，已在 maliang venv 里核对）。"""
    x = np.asarray(img, np.float32)
    H, W, C = x.shape
    if (H, W) == (h, w):
        return np.asarray(img, np.uint8)
    assert H >= h and W >= w, f"area_resize 只做缩小：{(H, W)} → {(h, w)}"
    y = (_area_w(H, h) @ x.reshape(H, -1)).reshape(h, W, C)
    y = np.einsum("xw,ywc->yxc", _area_w(W, w), y, optimize=True)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def to_gen(img: np.ndarray, nearest: bool = False) -> Image.Image:
    """任意 RGB → W_GEN×H_GEN PIL（长宽比不对时居中裁成 W:H；语义图用最近邻，照片用 Lanczos）。"""
    im = Image.fromarray(np.asarray(img, np.uint8)[..., :3])
    w, h = im.size
    if abs(w / h - W_GEN / H_GEN) > 1e-3:
        if w / h > W_GEN / H_GEN:
            nw = round(h * W_GEN / H_GEN)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = round(w * H_GEN / W_GEN)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize((W_GEN, H_GEN), Image.NEAREST if nearest else Image.LANCZOS)


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_ids(path: str, frame: int = 0) -> np.ndarray:
    """npz（键 ids，[F,H,W] 取 frame）/ 单通道 png（raw id）/ RGB png（colorized，经 ids_from_map）。"""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            a = z["ids"]
        return ids_from_map(a[frame] if a.ndim == 3 else a)
    im = Image.open(path)
    return ids_from_map(np.asarray(im if im.mode == "L" else im.convert("RGB")))


# ------------------------------------------------------------------ 数据集帧 0 ----
def clip_row(cid: str, data_root: str = DATA_ROOT) -> dict:
    with open(os.path.join(data_root, "index", "clips.jsonl")) as f:
        for line in f:
            if f'"{cid}"' in line:
                r = json.loads(line)
                if r["clip_id"] == cid:
                    return r
    raise KeyError(f"clips.jsonl 里没有 {cid}")


def clip_frame(cid: str, frame: int = 0, data_root: str = DATA_ROOT, raw_root: str = RAW_ROOT) -> np.ndarray:
    """clip 的第 frame 帧原始 RGB（tar 里的 jpg，1280×704）→ 裁 x∈[CROP_X0, CROP_X0+CROP_W) → INTER_AREA W×H
    （= geometry.resample_rgb：832×480 裁 [30,1250)，1280×704 原样；jpg 用 PIL 解码，与 cv2 可能差 ±1）。"""
    r = clip_row(cid, data_root)
    fi = int(r["start"]) + frame
    shard = fi // 1600 * 1600                                    # infer.SHARD：原始 tar 每片 1,600 帧
    path = os.path.join(raw_root, r["subset"], r["scene"], r["run"], f"frames-{shard:06d}.tar")
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            key, _, ext = m.name.rpartition("/")[2].partition(".")
            if m.isfile() and ext == "jpg" and key.isdigit() and int(key) == fi:
                rgb = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
                assert rgb.shape == (SRC_H, SRC_W, 3), rgb.shape
                return area_resize(rgb[:, CROP_X0:CROP_X0 + CROP_W], W_OUT, H_OUT)
    raise FileNotFoundError(f"{path} 里没有帧 {fi}")


def clip_ids(cid: str, frame: int = 0, data_root: str = DATA_ROOT) -> np.ndarray:
    with np.load(os.path.join(data_root, "cond", cid + ".npz"), allow_pickle=False) as z:
        ids = np.ascontiguousarray(z["ids"][frame])
    if ids.shape != (H_OUT, W_OUT):
        raise ValueError(f"{data_root} 的 cond 是 {ids.shape[1]}x{ids.shape[0]}，本进程 GC_RES={RES_TAG}：设 GC_RES 或换数据根")
    return ids


# ------------------------------------------------------------------ 布局指标 ----
def _lab(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(np.maximum(xyz, 1e-8)), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def layout_metrics(rgb: np.ndarray, ids: np.ndarray, cell: int = 8, tol: int = 3) -> dict:
    """生成帧对语义图的布局贴合度（无参考，同一套数在真实帧 0 上算出上限）。

    * sep_miou：8×8 格的特征（Lab 均值 + L 标准差），按 GT 类在**本图上**拟合每类对角高斯，再逐格判最近类 ⇒ mIoU
      （只算覆盖 ≥ 1 % 且纯度 ≥ 75 % 的格的类；角色 parts 合成 player / boss）。布局对 ⇒ 各类区域内外观一致、类间可分。
    * edge_recall：GT 类边界像素里，tol px 内有图像强边缘（Sobel |∇L| 的前 15 %）的比例。
    """
    from scipy import ndimage
    raw = np.asarray(ids, np.uint8)
    ids = raw.copy()
    ids[np.isin(ids, PLAYER_PARTS)] = 10
    ids[np.isin(ids, BOSS_PARTS)] = 11
    L = _lab(rgb)
    H, W = ids.shape
    h, w = H // cell, W // cell
    blk = L[:h * cell, :w * cell].reshape(h, cell, w, cell, 3)
    feat = np.concatenate([blk.mean((1, 3)), blk[..., 0].std((1, 3))[..., None]], -1).reshape(-1, 4)
    lab_blk = ids[:h * cell, :w * cell].reshape(h, cell, w, cell).transpose(0, 2, 1, 3).reshape(h * w, -1)
    maj = np.array([np.bincount(r, minlength=256).argmax() for r in lab_blk])
    pure = (lab_blk == maj[:, None]).mean(1) >= 0.75
    cls = [c for c in np.unique(maj[pure]) if c != UNKNOWN_ID and (maj[pure] == c).mean() >= 0.01]
    out = {"classes": [int(c) for c in cls]}
    if len(cls) >= 2:
        sel = pure & np.isin(maj, cls)
        X, y = feat[sel], maj[sel]
        mu = np.stack([X[y == c].mean(0) for c in cls])
        var = np.stack([X[y == c].var(0) for c in cls]) + 4.0
        ll = -0.5 * (((X[:, None] - mu[None]) ** 2 / var[None]).sum(-1) + np.log(var).sum(-1)[None])
        pred = np.asarray(cls)[ll.argmax(1)]
        ious = [((pred == c) & (y == c)).sum() / max(((pred == c) | (y == c)).sum(), 1) for c in cls]
        out.update(sep_miou=float(np.mean(ious)), sep_acc=float((pred == y).mean()),
                   sep_iou={int(c): round(float(v), 3) for c, v in zip(cls, ious)})
    # flat_frac：8×8 格里没有纹理（L 标准差 < 2）且均值离 FLUX 色图 ΔE < 10 的比例 = 照抄平涂色（上面两项会虚高）
    mblk = _lab(flux_map(raw))[:h * cell, :w * cell].reshape(h, cell, w, cell, 3).mean((1, 3)).reshape(-1, 3)
    out["flat_frac"] = float(((feat[:, 3] < 2) & (np.linalg.norm(feat[:, :3] - mblk, axis=-1) < 10)).mean())
    edges = _edges(rgb)
    bnd = np.zeros_like(ids, bool)
    bnd[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    bnd[1:, :] |= ids[1:, :] != ids[:-1, :]
    bnd &= ids != UNKNOWN_ID
    if bnd.any():
        near = ndimage.distance_transform_edt(~edges) <= tol
        out["edge_recall"] = float(near[bnd].mean())
    return out


def layout_score(m: dict) -> float:
    """best_of 的打分：sep_miou + edge_recall − 2·max(0, flat_frac − 0.3)（只有一类时 sep_miou 记 0.5）。
    flat_frac 只罚明显照抄平涂色的图（strength < 1 从色图起步 ≈ 0.8–0.95）；正常生成的大片平滑泥地也有 0.1–0.3。"""
    return m.get("sep_miou", 0.5) + m.get("edge_recall", 0.0) - 2.0 * max(0.0, m.get("flat_frac", 0.0) - 0.3)


def _edges(rgb: np.ndarray, q: float = 0.85) -> np.ndarray:
    from scipy import ndimage
    g = ndimage.gaussian_filter(_lab(rgb)[..., 0], 1.0)
    mag = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return mag >= np.quantile(mag, q)


def struct_metrics(src: np.ndarray, out: np.ndarray, tol: int = 2) -> dict:
    """restyle 的结构保持：源帧与输出的强边缘（各自 |∇L| 前 15 %）在 tol px 内互相命中的 recall / precision / F1。"""
    from scipy import ndimage
    a, b = _edges(src), _edges(out)
    r = float((ndimage.distance_transform_edt(~b) <= tol)[a].mean())
    p = float((ndimage.distance_transform_edt(~a) <= tol)[b].mean())
    return {"edge_recall": r, "edge_precision": p, "edge_f1": 2 * r * p / max(r + p, 1e-9)}


# ------------------------------------------------------------------ 模型 ----
def snapshot(repo: str, rev: str) -> str:
    p = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"本地没有 {repo}@{rev}（{p}）；本工具离线运行，不下载")
    return p


class _Flux2:
    """FLUX.2 klein（步数蒸馏，无 CFG）：Flux2KleinPipeline 的组件 + 自己的采样循环（多参考 + 可选 strength < 1 起步）。"""

    def __init__(self, path: str, device: str):
        import torch
        from diffusers import Flux2KleinPipeline
        self.torch, self.device = torch, device
        self.pipe = Flux2KleinPipeline.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def _encode(self, img: Image.Image):
        p = self.pipe
        x = p.image_processor.preprocess(img, height=img.height, width=img.width, resize_mode="crop")
        return p._pack_latents(p._encode_vae_image(x.to(self.device, p.vae.dtype), generator=None))

    def _decode(self, tok, h: int, w: int) -> np.ndarray:
        p = self.pipe
        lat = tok.permute(0, 2, 1).reshape(1, -1, h, w)
        mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        std = self.torch.sqrt(p.vae.bn.running_var.view(1, -1, 1, 1) + p.vae.config.batch_norm_eps)
        std = std.to(lat.device, lat.dtype)
        img = p.vae.decode(p._unpatchify_latents(lat * std + mean).to(p.vae.dtype), return_dict=False)[0]
        return np.asarray(p.image_processor.postprocess(img, output_type="pil")[0])

    def __call__(self, prompt: str, refs: list[Image.Image], steps: int, seed: int,
                 init: Image.Image | None = None, strength: float = 1.0) -> np.ndarray:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        torch, p, dev = self.torch, self.pipe, self.device
        h, w = H_GEN // 16, W_GEN // 16
        emb, txt_ids = p.encode_prompt(prompt=prompt, device=dev)
        g = torch.Generator(dev).manual_seed(seed)
        img_lat = img_ids = None
        if refs:
            conds = [p.image_processor.preprocess(c, height=c.height, width=c.width, resize_mode="crop") for c in refs]
            img_lat, img_ids = p.prepare_image_latents(conds, 1, g, dev, p.vae.dtype)
        noise, lat_ids = p.prepare_latents(1, p.transformer.config.in_channels // 4, H_GEN, W_GEN, emb.dtype, dev, g)
        mu = compute_empirical_mu(image_seq_len=noise.shape[1], num_steps=steps)
        ts, _ = retrieve_timesteps(p.scheduler, steps, dev, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
        sig = p.scheduler.sigmas.float()
        start = 0
        if init is not None and strength < 1.0:          # SDEdit 起步：最接近 strength 的 sigma
            start = int(np.argmin(np.abs(sig[:-1].cpu().numpy() - strength)))
        lat = noise.clone()
        if start > 0:
            lat = ((1 - sig[start]) * self._encode(init).float() + sig[start] * noise.float()).to(noise.dtype)
        p.scheduler.set_begin_index(start)
        for i in range(start, len(ts)):
            inp, ids = lat, lat_ids
            if img_lat is not None:
                inp, ids = torch.cat([lat, img_lat], 1), torch.cat([lat_ids, img_ids], 1)
            v = p.transformer(hidden_states=inp.to(p.transformer.dtype), timestep=ts[i].expand(1).to(lat.dtype) / 1000,
                              guidance=None, encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=ids,
                              return_dict=False)[0][:, :lat.shape[1]]
            lat = p.scheduler.step(v, ts[i], lat, return_dict=False)[0]
        return self._decode(lat, h, w)


class _Flux3:
    """FLUX.3 klein（flow-map 蒸馏，16 步，shift 16，无 CFG；照 PCE recolor/flux3_recolor.py 的加载与采样），提示词原样
    （不跑 prompt upsampler：我们的提示词已是完整描述）。"""

    def __init__(self, path: str, device: str):
        import torch
        from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel, ModularPipeline
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch, self.device = torch, device
        self.pipe = ModularPipeline.from_pretrained(path)
        bf16 = torch.bfloat16
        self.pipe.update_components(
            transformer=Flux3Transformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=bf16).eval(),
            vae=AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32).eval(),
            text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(path, "text_encoder"),
                                                                         dtype=bf16).eval(),
            tokenizer=AutoProcessor.from_pretrained(os.path.join(path, "text_encoder")))
        self.pipe.to(device)

    def _encode(self, img: Image.Image):
        torch, vae = self.torch, self.pipe.vae
        x = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1.0
        lat = vae.encode(x[None].to(self.device, vae.dtype)).latent_dist.mode()
        b, c, h, w = lat.shape
        lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat)
        return ((lat - mean) / std).flatten(2).transpose(1, 2).float()

    def _decode(self, tok, h: int, w: int) -> np.ndarray:
        torch, vae = self.torch, self.pipe.vae
        lat = tok.float().transpose(1, 2).reshape(1, -1, h, w).to(vae.dtype)
        lat = lat * torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat) \
            + vae.bn.running_mean.view(1, -1, 1, 1).to(lat)
        b, c, hh, ww = lat.shape
        lat = lat.reshape(b, c // 4, 2, 2, hh, ww).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, hh * 2, ww * 2)
        img = vae.decode(lat, return_dict=False)[0][0].float()
        return ((img.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def __call__(self, prompt: str, refs: list[Image.Image], steps: int, seed: int,
                 init: Image.Image | None = None, strength: float = 1.0, shift: float = 16.0) -> np.ndarray:
        from diffusers.modular_pipelines.flux3.before_denoise import Flux3TextRoPEInputsStep, _prepare_image_ids
        from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
        torch, dev, tr = self.torch, self.device, self.pipe.transformer
        h, w = H_GEN // 16, W_GEN // 16
        emb = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(self.pipe.text_encoder, self.pipe.tokenizer, prompt,
                                                                device=dev)
        txt_ids = Flux3TextRoPEInputsStep._prepare_text_ids(emb).to(dev)
        kw = {}
        if refs:
            kw["image_cond_hidden_states"] = torch.cat([self._encode(c) for c in refs], 1).to(tr.dtype)
            kw["image_cond_ids"] = torch.cat([_prepare_image_ids(c.height // 16, c.width // 16, 0.5, l_coord=-(1 + k))
                                              for k, c in enumerate(refs)], 0).to(dev)
        img_ids = _prepare_image_ids(h, w, 0.5).to(dev)
        noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(seed))
        noise = noise.to(torch.bfloat16).float()[None].to(dev)                 # BFL：CPU generator，bf16 值
        t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
        sched = (shift * t / (1.0 + (shift - 1.0) * t)).float().to(dev)
        start = int(np.argmin(np.abs(sched[:-1].cpu().numpy() - strength))) if init is not None and strength < 1 else 0
        x = noise.clone() if start == 0 else (1 - sched[start]) * self._encode(init) + sched[start] * noise
        for i in range(start, steps):
            u = tr(hidden_states=x.to(tr.dtype), encoder_hidden_states=emb, timestep=sched[i].reshape(1).to(tr.dtype),
                   target_timestep=sched[i + 1].reshape(1).to(tr.dtype), img_ids=img_ids, txt_ids=txt_ids,
                   return_dict=False, **kw)[0]
            x = x + (sched[i + 1] - sched[i]) * u.float()
        return self._decode(x, h, w)


class FirstFrameFlux:
    def __init__(self, model: str = "klein9b", device: str = "cuda"):
        backend, repo, rev = MODELS[model]
        t0 = time.time()
        self.model, self.backend = model, backend
        self.net = (_Flux2 if backend == "flux2" else _Flux3)(snapshot(repo, rev), device)
        self.load_sec = time.time() - t0
        self.last: dict = {}

    def generate(self, prompt: str, refs: list[Image.Image], steps: int | None = None, seed: int = 0,
                 init: Image.Image | None = None, strength: float = 1.0) -> np.ndarray:
        import torch
        steps = steps or DEFAULT_STEPS[self.backend]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            big = self.net(prompt, refs, steps, seed, init=init, strength=strength)
        torch.cuda.synchronize()
        self.last = {"model": self.model, "prompt": prompt, "steps": steps, "seed": seed, "strength": strength,
                     "n_refs": len(refs), "gen_size": [W_GEN, H_GEN], "sec": round(time.time() - t0, 3),
                     "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        return area_resize(big, W_OUT, H_OUT)

    def restyle(self, rgb: np.ndarray, style: str, version: str = DEFAULT_RESTYLE_PROMPT, **kw) -> np.ndarray:
        src = to_gen(rgb)
        text = RESTYLE_PROMPTS[version].format(style=STYLE_PRESETS.get(style.strip().lower(), style))
        return self.generate(text, [src], init=src, **kw)

    def map2frame(self, ids: np.ndarray, ref: np.ndarray | None = None, prompt: str = "",
                  version: str = DEFAULT_MAP_PROMPT, init: str = "none", best_of: int = DEFAULT_BEST_OF, seed: int = 0,
                  **kw) -> np.ndarray:
        """ids: parts raw id [H,W]（W:H）；ref: 风格参考帧 RGB；prompt: 场景描述（追加在模板后）；
        version: MAP_PROMPTS 的键；init: none | map（strength < 1 时从 FLUX 色图的带噪 latent 起步，会照抄平涂色，不推荐）；
        best_of: 种子 seed..seed+N−1 各出一张，按 layout_score（只要 ids，不要 GT 帧，L3 也能用）留最好的。"""
        m = to_gen(flux_map(ids, MAP_PALETTE[version], tiles=version == "v4"), nearest=True)
        refs = [m] + ([to_gen(ref)] if ref is not None else [])
        legend = LEGEND_V1 if version == "v1" else map_legend(ids, version)
        text = MAP_PROMPTS[version].format(legend=legend) + (MAP_STYLE_REFS[version] if ref is not None else "")
        if prompt:
            text += " Scene: " + prompt.strip()
        best, cands, sec, peak, res = None, [], 0.0, 0.0, 0.0
        for s in range(seed, seed + max(1, best_of)):
            out = self.generate(text, refs, init=m if init == "map" else None, seed=s, **kw)
            sec, peak = sec + self.last["sec"], max(peak, self.last["peak_mem_gib"])
            res = max(res, self.last["peak_reserved_gib"])
            if best_of > 1:
                cands.append({"seed": s, "score": round(layout_score(layout_metrics(out, ids)), 4)})
                if best is None or cands[-1]["score"] > max(c["score"] for c in cands[:-1]):
                    best = (out, s)
            else:
                best = (out, s)
        self.last.update(seed=best[1], sec=round(sec, 3), peak_mem_gib=peak, peak_reserved_gib=res, version=version,
                         best_of=best_of, candidates=cands)
        return best[0]


# ------------------------------------------------------------------ CLI ----
def run_job(ff: FirstFrameFlux, job: dict) -> dict:
    kw = {k: job[k] for k in ("steps", "seed", "strength") if job.get(k) is not None}
    gt_ids = src = None
    if job["mode"] == "restyle":
        rgb = load_rgb(job["image"]) if job.get("image") else clip_frame(job["clip"], job.get("frame", 0))
        if job.get("clip"):
            gt_ids = clip_ids(job["clip"], job.get("frame", 0))
        src = rgb
        out = ff.restyle(rgb, job["style"], version=job.get("version") or DEFAULT_RESTYLE_PROMPT, **kw)
    elif job["mode"] == "map2frame":
        if job.get("map"):
            gt_ids = load_ids(job["map"], job.get("frame", 0))
        else:
            gt_ids = clip_ids(job["clip"], job.get("frame", 0))
        ref = load_rgb(job["ref"]) if job.get("ref") else None
        out = ff.map2frame(gt_ids, ref=ref, prompt=job.get("prompt", ""), init=job.get("init", "none"),
                           best_of=int(job.get("best_of") or DEFAULT_BEST_OF),
                           version=job.get("version") or DEFAULT_MAP_PROMPT, **kw)
    else:
        raise ValueError(job["mode"])
    os.makedirs(os.path.dirname(os.path.abspath(job["out"])), exist_ok=True)
    Image.fromarray(out).save(job["out"])
    if job.get("save_map") and gt_ids is not None:                  # FLUX 实际看到的色图（W×H）
        v = job.get("version") or (DEFAULT_MAP_PROMPT if job["mode"] == "map2frame" else "v4")
        Image.fromarray(flux_map(gt_ids, MAP_PALETTE.get(v, "white"), tiles=v == "v4")).save(
            os.path.splitext(job["out"])[0] + "_fluxmap.png")
    info = {"job": job, **ff.last, "load_sec": round(ff.load_sec, 1)}
    if gt_ids is not None and gt_ids.shape == out.shape[:2]:
        info["layout"] = layout_metrics(out, gt_ids)
    if src is not None and src.shape == out.shape:
        info["struct"] = struct_metrics(src, out)
    with open(os.path.splitext(job["out"])[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="klein9b", choices=sorted(MODELS))
    common.add_argument("--steps", type=int, default=None, help="默认 klein 4 / flux3 16")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--strength", type=float, default=None)
    common.add_argument("--device", default="cuda")
    common.add_argument("--version", default=None, help="提示词版本（MAP_PROMPTS / RESTYLE_PROMPTS 的键）")
    common.add_argument("--best_of", type=int, default=None,
                        help=f"map2frame：出 N 个种子按 layout_score 留最好的（默认 {DEFAULT_BEST_OF}；1 = 最快）")
    r = sub.add_parser("restyle", parents=[common])
    r.add_argument("--image")
    r.add_argument("--clip", help="data/gc25d_v10 的 clip_id（取原始帧 --frame）")
    r.add_argument("--frame", type=int, default=0)
    r.add_argument("--style", required=True)
    r.add_argument("--out", required=True)
    m = sub.add_parser("map2frame", parents=[common])
    m.add_argument("--map", help="colorized 语义图 png / 单通道 raw id png / cond npz")
    m.add_argument("--clip", help="用该 clip 的 cond 帧 --frame 作语义图")
    m.add_argument("--frame", type=int, default=0)
    m.add_argument("--ref", help="风格参考帧（可选）")
    m.add_argument("--prompt", default="", help="场景描述，追加在提示词模板后")
    m.add_argument("--init", default="none", choices=["none", "map"])
    m.add_argument("--save_map", action="store_true", help="另存 FLUX 看到的色图 <out>_fluxmap.png")
    m.add_argument("--out", required=True)
    b = sub.add_parser("batch", parents=[common])
    b.add_argument("--jobs", required=True)
    a = ap.parse_args(argv)
    ff = FirstFrameFlux(a.model, a.device)
    if a.mode == "batch":
        jobs = json.load(open(a.jobs))
        for j in jobs:
            j = {"steps": a.steps, "seed": a.seed, "strength": a.strength, "version": a.version, "best_of": a.best_of,
                 **j}
            info = run_job(ff, j)
            print(json.dumps({"out": j["out"], "sec": info["sec"], "peak_mem_gib": info["peak_mem_gib"],
                              "peak_reserved_gib": info["peak_reserved_gib"],
                              "layout": info.get("layout"), "struct": info.get("struct")}), flush=True)
        return
    job = {k: v for k, v in vars(a).items() if k not in ("model", "device")}
    info = run_job(ff, job)
    print(json.dumps({k: info[k] for k in ("sec", "peak_mem_gib", "peak_reserved_gib", "load_sec")}
                     | {"layout": info.get("layout"), "struct": info.get("struct")}))


if __name__ == "__main__":
    sys.exit(main())
