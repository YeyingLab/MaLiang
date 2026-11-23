"""可视化：并排 mp4（16 fps）`GT | generated | ids 色图 | (可选) overlay`，动作文字烧进画面。

布局
----
* 每个面板 = 训练分辨率 W×H（geometry.DST_W × DST_H，env GC_RES；默认 832×480），面板上方一条 48 px 的标题栏
  （总高 H+48：832×480 ⇒ 528 = 16×33，1280×704 ⇒ 752 = 16×47；宽 = W×面板数；W、H 是 32 的倍数 ⇒ 都是 16 的倍数，
  libx264 / yuv420p 不需要再补边）：
    第 1 行：各面板名（GT / generated / ids / overlay）；
    第 2 行：`frame i/80 · cell k · keys W+D atk · moving up-right, attacking`（当前帧所属 cell 的动作块与 prompt），
             右侧是 title（clip_id 等）。
* 帧 → cell：帧 0 → cell 0；帧 i ≥ 1 → cell ⌈i/4⌉ = (i+3)//4。
* 没有 GT（用户自带首帧 + ids 的推理）⇒ 不画 GT 面板（不拿首帧冒充 GT）。
* ids 色图用 geometry.colorize_ids。
  ids 面板画的必须是**模型实际收到的条件**（infer 在 --sem_mode unknown / none 时传全 255，而不是用户给的 ids），
  面板名（gen_name / ids_name / overlay_name）由调用方写明 sem_mode、cfg 等，见 infer.run_item。
* overlay = 生成帧 × (1−α) + ids 色图 × α（α 默认 0.45），看生成内容是否贴着语义图。
* 面板之间的 1 px 分隔线直接画在后一面板的第 0 列上（不改变总宽 W×n，仍是 16 的倍数）；只影响并排视频，
  指标与 _gen.mp4 不受影响。

写 mp4 走 imageio-ffmpeg（libx264，yuv420p，quality 8），逐帧流式写，不在内存里拼整段 [81, H+48, W×n, 3]。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

import geometry as GEO

FPS = 16
PANEL_H, PANEL_W = GEO.DST_H, GEO.DST_W        # 832×480 ⇒ 480, 832
HEADER_H = 48                                  # 固定 48 px（16 的倍数 ⇒ 总高 PANEL_H + 48 仍是 16 的倍数）
assert (PANEL_H + HEADER_H) % 16 == 0 and PANEL_W % 16 == 0, (PANEL_H, PANEL_W)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 24, 24)
FG = (235, 235, 235)
ACCENT = (255, 210, 90)


def frame_cell(i: int) -> int:
    """帧号（clip 内 0..80）→ latent cell（0..20），。"""
    assert 0 <= i <= 80, i
    return 0 if i == 0 else (i + 3) // 4


def keys_label(block) -> str:
    """动作块 (w, a, s, d, attack) → 'W+D atk' / 'idle' / 'idle atk'。"""
    b = [int(v) for v in block]
    assert len(b) == 5, f"动作块应为 5 维 (w,a,s,d,attack)，得到 {b}"
    keys = "+".join(k for k, v in zip("WASD", b[:4]) if v)
    return (keys or "idle") + (" atk" if b[4] else "")


def motion_text(prompt: str) -> str:
    """'In a 2.5D isometric view, the player is moving right, attacking.' → 'moving right, attacking'。
    非模板串（例如 NULL prompt）原样返回。"""
    key = "the player is "
    if key in prompt:
        return prompt.split(key, 1)[1].rstrip(".")
    return prompt


def colorize(ids: np.ndarray) -> np.ndarray:
    """uint8 id [..., H, W] → uint8 RGB [..., H, W, 3]（geometry.colorize_ids）。"""
    assert ids.dtype == np.uint8, ids.dtype
    out = GEO.colorize_ids(ids)
    assert out.shape == ids.shape + (3,) and out.dtype == np.uint8, (out.shape, out.dtype)
    return out


def blend(rgb: np.ndarray, ids_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    assert rgb.shape == ids_rgb.shape and rgb.dtype == ids_rgb.dtype == np.uint8, (rgb.shape, ids_rgb.shape)
    return cv2.addWeighted(rgb, 1.0 - alpha, ids_rgb, alpha, 0.0)


def _put(img: np.ndarray, text: str, x: int, y: int, color=FG, scale: float = 0.55) -> None:
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _fit(text: str, max_w: int, scale: float = 0.55) -> str:
    """截断到 max_w 像素宽（cv2.putText 不会自动换行）。"""
    if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= max_w:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, 1)[0][0] > max_w:
        text = text[:-1]
    return text + "..."


class SideBySide:
    """逐帧拼 `GT | generated | ids | overlay`。gen / gt：uint8 [F,H,W,3]；ids：uint8 [F,H,W]；
    prompts：21 条 cell prompt；blocks：uint8 [21,5] cell 顺序的动作块（cell 0 = cell 1）。"""

    def __init__(self, gen: np.ndarray, ids: np.ndarray, gt: Optional[np.ndarray] = None, overlay: bool = False,
                 prompts: Optional[Sequence[str]] = None, blocks: Optional[np.ndarray] = None, title: str = "",
                 alpha: float = 0.45, gen_name: str = "generated", ids_name: str = "ids", overlay_name: str = "overlay"):
        F = gen.shape[0]
        assert gen.dtype == np.uint8 and gen.shape == (F, PANEL_H, PANEL_W, 3), \
            f"gen 应为 uint8 [F,{PANEL_H},{PANEL_W},3]，得到 {gen.dtype} {gen.shape}"
        assert ids.dtype == np.uint8 and ids.shape == (F, PANEL_H, PANEL_W), \
            f"ids 应为 uint8 [F,{PANEL_H},{PANEL_W}]，得到 {ids.dtype} {ids.shape}"
        if gt is not None:
            assert gt.shape == gen.shape and gt.dtype == np.uint8, f"gt 形状 {gt.shape} 与 gen {gen.shape} 不符"
        if prompts is not None:
            assert len(prompts) == 21, f"应有 21 条 cell prompt，得到 {len(prompts)}"
        if blocks is not None:
            blocks = np.asarray(blocks)
            assert blocks.shape == (21, 5), f"blocks 应为 [21,5]（cell 顺序），得到 {blocks.shape}"
        self.gen, self.ids, self.gt, self.overlay = gen, ids, gt, overlay
        self.prompts, self.blocks, self.title, self.alpha = prompts, blocks, title, alpha
        self.names = (["GT"] if gt is not None else []) + [gen_name, ids_name] + ([overlay_name] if overlay else [])
        self.gen_index = 1 if gt is not None else 0
        self.n_frames = F
        self.width = PANEL_W * len(self.names)
        self.height = PANEL_H + HEADER_H

    def caption(self, i: int) -> str:
        k = frame_cell(i)
        parts = [f"frame {i}/{self.n_frames - 1}", f"cell {k}"]
        if self.blocks is not None:
            parts.append("keys " + keys_label(self.blocks[k]))
        if self.prompts is not None:
            parts.append(motion_text(self.prompts[k]))
        return "  |  ".join(parts)

    def frame(self, i: int) -> np.ndarray:
        ids_rgb = colorize(self.ids[i])
        panels = ([self.gt[i]] if self.gt is not None else []) + [self.gen[i], ids_rgb]
        if self.overlay:
            panels.append(blend(self.gen[i], ids_rgb, self.alpha))
        out = np.empty((self.height, self.width, 3), np.uint8)
        out[:HEADER_H] = BG
        out[HEADER_H:] = np.concatenate(panels, axis=1)
        for j, name in enumerate(self.names):
            x0 = j * PANEL_W
            _put(out, _fit(name, PANEL_W - 16), x0 + 8, 18, ACCENT if j == self.gen_index else FG)
            if j:
                out[:, x0:x0 + 1] = BG                     # 面板分隔线
        cap = self.caption(i)
        title = _fit(self.title, self.width // 2, 0.5) if self.title else ""      # 标题至多占一半宽，不挤掉动作说明
        title_w = cv2.getTextSize(title, FONT, 0.5, 1)[0][0] if title else 0
        _put(out, _fit(cap, self.width - title_w - 32), 8, 40, FG)
        if title:
            _put(out, title, self.width - title_w - 8, 40, (170, 170, 170), 0.5)
        return out


def write_mp4(frames_iter, path: str, fps: int = FPS, quality: int = 8) -> None:
    """逐帧写 mp4（libx264 / yuv420p）。frames_iter 产出 uint8 [H,W,3]，H、W 须是 16 的倍数（不让 imageio 偷偷补边）。"""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, quality=quality, codec="libx264", pixelformat="yuv420p",
                                macro_block_size=16)
    n = 0
    try:
        for fr in frames_iter:
            assert fr.dtype == np.uint8 and fr.ndim == 3 and fr.shape[0] % 16 == 0 and fr.shape[1] % 16 == 0, \
                f"帧应为 uint8 [H,W,3] 且 H、W 为 16 的倍数，得到 {fr.dtype} {fr.shape}"
            writer.append_data(fr)
            n += 1
    finally:
        writer.close()
    assert n > 0, f"{path}：没有写入任何帧"


def write_side_by_side(path: str, gen: np.ndarray, ids: np.ndarray, gt: Optional[np.ndarray] = None,
                       overlay: bool = False, prompts=None, blocks=None, title: str = "", fps: int = FPS,
                       **names) -> SideBySide:
    """names：gen_name / ids_name / overlay_name（见 SideBySide）。"""
    sbs = SideBySide(gen, ids, gt=gt, overlay=overlay, prompts=prompts, blocks=blocks, title=title, **names)
    write_mp4((sbs.frame(i) for i in range(sbs.n_frames)), path, fps=fps)
    return sbs


def write_video(path: str, frames: np.ndarray, fps: int = FPS) -> None:
    """单独的生成视频（W×H，无标题栏），给下游评测 / 拼接用。"""
    assert frames.dtype == np.uint8 and frames.ndim == 4, (frames.dtype, frames.shape)
    write_mp4(iter(frames), path, fps=fps)


def contact_sheet(path: str, sbs: SideBySide, frames=(0, 20, 40, 60, 80), scale: float = 0.5) -> np.ndarray:
    """挑几帧的并排图竖着拼成一张 jpg（快速人工检查用）。"""
    rows = [sbs.frame(i) for i in frames if i < sbs.n_frames]
    sheet = np.concatenate(rows, axis=0)
    if scale != 1.0:
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok = cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, f"写 {path} 失败"
    return sheet


def read_mp4(path: str) -> np.ndarray:
    """mp4 → uint8 [F,H,W,3]（测试 / 复查用）。"""
    import imageio
    r = imageio.get_reader(path)
    try:
        frames = [np.asarray(f) for f in r]
    finally:
        r.close()
    return np.stack(frames, 0)


def mp4_fps(path: str) -> float:
    import imageio
    r = imageio.get_reader(path)
    try:
        return float(r.get_meta_data()["fps"])
    finally:
        r.close()
