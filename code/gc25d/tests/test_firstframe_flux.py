"""firstframe_flux.py 的 CPU 单测（不加载 FLUX）：调色板与 geometry / sample 原表一致、色图 → raw id 还原、
INTER_AREA 等价、动态色表、布局指标的上下限、数据集帧 0 读取。

    PYTHONPATH=code/gc25d:code/arope .venv/bin/python -m pytest -q code/gc25d/tests/test_firstframe_flux.py
（maliang venv：有 cv2 / torch，用来核对；firstframe_flux 本身在 PCE 的 pce_f3 env 里跑，不依赖 cv2。）
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

import firstframe_flux as FF
import geometry as GEO
import model as GM
import sample as S

CLIP = "06_Overgrown_Tropical_Water_Temple__P9C_golden_s6946__003552"
HAVE_DATA = os.path.isfile(os.path.join(FF.DATA_ROOT, "cond", CLIP + ".npz")) and GEO.root_res(FF.DATA_ROOT) == GEO.RES_TAG


def _ids(seed=0, shape=(480, 832)):
    vals = np.asarray(sorted(FF.VALID_IDS), np.uint8)
    rng = np.random.default_rng(seed)
    out = np.asarray([1, 2, 3, 5], np.uint8)[rng.integers(0, 4, (shape[0] // 16, shape[1] // 16))]
    out = np.kron(out, np.ones((16, 16), np.uint8))
    out[100:140, 200:230] = vals[(np.arange(40 * 30) % len(vals))].reshape(40, 30)   # 所有值都出现
    return out


def test_palette_matches_sources():
    for k, c in FF.GEO_COLORS.items():
        assert tuple(GEO.PALETTE[k]) == c, k
    assert FF.PART_COLORS == S.PART_COLORS
    assert FF.PLAYER_PARTS == GM.PLAYER_PART_IDS and FF.BOSS_PARTS == GM.BOSS_PART_IDS
    assert FF.VALID_IDS == GM.valid_raw_ids("parts")
    assert tuple(GEO.HATCH_GREY) == FF.HATCH_GREY


def test_ids_from_map_roundtrip():
    ids = _ids()
    assert (FF.ids_from_map(S.colorize_parts(ids)) == ids).all()          # parts 色：逐像素还原
    assert (FF.ids_from_map(ids) == ids).all()                            # 单通道 raw id 原样
    geo = FF.ids_from_map(GEO.colorize_ids(ids))                          # geometry 色：部位是白 ⇒ 只能到角色级
    assert (GM.collapse_parts(geo) == GM.collapse_parts(ids)).mean() > 0.99
    with pytest.raises(ValueError):
        FF.ids_from_map(np.full((4, 4), 7, np.uint8))


def test_flux_map_merges_parts():
    ids = _ids()
    fm = FF.ids_from_map(FF.flux_map(ids))                                # FLUX 色：身体部位合成 core，武器单独一色
    assert (fm[np.isin(ids, FF.PLAYER_PARTS[:-1])] == 10).all() and (fm[np.isin(ids, FF.BOSS_PARTS[:-1])] == 11).all()
    assert (fm[ids == 16] == 16).all() and (fm[ids == 21] == 21).all()
    static = ~np.isin(ids, FF.PLAYER_PARTS + FF.BOSS_PARTS)
    assert (fm[static] == ids[static]).all()


def test_area_resize_equals_cv2():
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 256, (720, 1248, 3)).astype(np.uint8), (0, 0), 1.5)
    assert (FF.area_resize(a, 832, 480) == cv2.resize(a, (832, 480), interpolation=cv2.INTER_AREA)).all()
    b = rng.integers(0, 256, (704, 1220, 3)).astype(np.uint8)            # resample_rgb 的裁剪后尺寸
    assert np.abs(FF.area_resize(b, 832, 480).astype(int)
                  - cv2.resize(b, (832, 480), interpolation=cv2.INTER_AREA).astype(int)).max() <= 1


def test_map_legend_only_present_classes():
    ids = np.full((480, 832), 1, np.uint8)
    t = FF.map_legend(ids)
    assert "brown" in t and "light grey" not in t and "no characters" in t
    ids[200:260, 400:430] = 10
    ids[200:260, 430:432] = 16
    t = FF.map_legend(ids)
    assert "blue humanoid" in t and "red humanoid" not in t and "only character" in t and "white stroke" in t
    ids[50:120, 100:150] = 11
    t = FF.map_legend(ids)
    assert "red humanoid" in t and "only characters" in t and "orange stroke" not in t


def test_struct_metrics_bounds():
    rng = np.random.default_rng(1)
    a = cv2.GaussianBlur(rng.integers(0, 256, (480, 832, 3)).astype(np.uint8), (0, 0), 3)
    b = cv2.GaussianBlur(rng.integers(0, 256, (480, 832, 3)).astype(np.uint8), (0, 0), 3)
    assert FF.struct_metrics(a, a)["edge_f1"] == pytest.approx(1.0)
    assert FF.struct_metrics(a, b)["edge_f1"] < 0.7


@pytest.mark.skipif(not HAVE_DATA, reason="gc25d_v10 数据不在")
def test_clip_frame_and_layout_metrics():
    import infer as INF
    row = FF.clip_row(CLIP)
    ref = GEO.resample_rgb(INF.read_raw_frames(row["scene"], row["run"], int(row["start"]), 1, "jpg",
                                               raw_root=os.path.join(FF.RAW_ROOT, row["subset"]))[0])
    rgb = FF.clip_frame(CLIP)
    assert rgb.shape == (GEO.DST_H, GEO.DST_W, 3) and np.abs(rgb.astype(int) - ref).max() <= 2     # PIL vs cv2 解 jpg
    ids = FF.clip_ids(CLIP)
    real = FF.layout_metrics(rgb, ids)
    flat = FF.layout_metrics(FF.flux_map(ids, "color"), ids)            # 平涂色图 = 上限
    other = FF.layout_metrics(np.ascontiguousarray(rgb[::-1, ::-1]), ids)  # 布局错开 = 下限
    assert flat["sep_miou"] > 0.95 and flat["edge_recall"] > 0.95
    assert real["sep_miou"] > other["sep_miou"] and real["edge_recall"] > other["edge_recall"]
