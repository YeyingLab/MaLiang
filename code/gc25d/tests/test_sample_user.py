"""sample_user.py 的 CPU 单测：首帧尺寸规则、ids 读取 / 严格校验 / 源分辨率重采样、prompts / actions、条件占比、clip 素材导出。

    PYTHONPATH=code/gc25d:code/arope .venv/bin/python -m pytest -q code/gc25d/tests/test_sample_user.py
最后一个用例导出一条真实 val clip（数据不在、或数据根的 res ≠ GC_RES 就跳过）。形状取自 geometry（GC_RES）。
"""
from __future__ import annotations

import json
import os
import types

import cv2
import numpy as np
import pytest

import geometry as GEO
import infer as INF
import model as GM
import sample as S
import sample_user as U
import text as TX

CLIP = "08_Gothic_Cathedral_Quarter__P9C_default_s8946__003552"
H, W = GEO.DST_H, GEO.DST_W
SRC_MODE = "as_is" if GEO.IDENTITY else "resample_rgb"        # 1280×704：源 = 训练分辨率 ⇒ 原样收


def _ids(shape=(81, H, W), seed=0, values=(0, 1, 3, 5, 10, 12, 16, 11, 21, 255)):
    rng = np.random.default_rng(seed)
    return np.asarray(values, np.uint8)[rng.integers(0, len(values), shape)]


def _png(path, rgb):
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return str(path)


# ---------------------------------------------------------------- 首帧
def test_first_frame_sizes(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    img, info = U.load_first_frame(_png(tmp_path / "a.png", a))
    assert info["resize"] == "as_is" and np.array_equal(img, a)
    src = rng.integers(0, 256, (704, 1280, 3), dtype=np.uint8)
    img, info = U.load_first_frame(_png(tmp_path / "s.png", src))
    assert info["resize"] == SRC_MODE and np.array_equal(img, GEO.resample_rgb(src))
    img, info = U.load_first_frame(_png(tmp_path / "o.png", rng.integers(0, 256, (600, 1000, 3), dtype=np.uint8)))
    assert info["resize"] == "center_crop_area" and img.shape == (H, W, 3) and info["src_size"] == [1000, 600]


# ---------------------------------------------------------------- ids
def test_ids_npz_npy_keys(tmp_path):
    ids = _ids()
    np.savez_compressed(tmp_path / "k.npz", ids=ids, other=np.zeros(3))
    np.savez_compressed(tmp_path / "one.npz", whatever=ids)
    np.savez_compressed(tmp_path / "two.npz", a=ids, b=ids)
    np.save(tmp_path / "x.npy", ids)
    for f in ("k.npz", "one.npz", "x.npy"):
        out, info = U.load_ids(str(tmp_path / f), "parts")
        assert np.array_equal(out, ids) and not info["resampled"]
    with pytest.raises(ValueError, match="不止一个"):
        U.load_ids(str(tmp_path / "two.npz"), "parts")


def test_ids_strict(tmp_path):
    ids = _ids()
    np.save(tmp_path / "f.npy", ids.astype(np.int32))
    with pytest.raises(ValueError, match="uint8"):
        U.load_ids(str(tmp_path / "f.npy"), "parts")
    np.save(tmp_path / "s.npy", ids[:80])
    with pytest.raises(ValueError, match=rf"\[81,{H},{W}\]"):
        U.load_ids(str(tmp_path / "s.npy"), "parts")
    for bad in (7, 9, 22, 254):
        b = ids.copy()
        b[40, 100, 200:203] = bad
        np.save(tmp_path / "b.npy", b)
        with pytest.raises(ValueError, match=rf"{bad}: 3"):
            U.load_ids(str(tmp_path / "b.npy"), "parts")


def test_ids_source_res_and_ids_vocab(tmp_path):
    src = _ids((81, 704, 1280), seed=3)
    np.savez(tmp_path / "src.npz", ids=src)
    out, info = U.load_ids(str(tmp_path / "src.npz"), "parts")
    assert info["resampled"] == (not GEO.IDENTITY) and info["src_shape"] == [81, 704, 1280]
    assert np.array_equal(out, GEO.resample_ids(src))
    # ids 词表的 ckpt：parts 先折叠（与 clip 模式 sample.model_ids 相同），结果 ⊆ ids 词表
    out_ids, _ = U.load_ids(str(tmp_path / "src.npz"), "ids")
    assert np.array_equal(out_ids, GM.collapse_parts(GEO.resample_ids(src)))
    assert set(np.unique(out_ids).tolist()) <= GM.valid_raw_ids("ids")
    assert np.array_equal(S.model_ids(out_ids, "ids", "cond"), out_ids)
    assert (S.model_ids(out, "parts", "unknown") == 255).all()


# ---------------------------------------------------------------- prompts
def _tt():
    return types.SimpleNamespace(table={p: None for p in TX.all_prompts() + [TX.NULL_PROMPT]})


def test_prompts_json_and_actions(tmp_path):
    p = TX.cell_prompts(INF.parse_actions("D:10,A:10"))
    a = types.SimpleNamespace(prompts=None, actions="D:10,A:10")
    prompts, blocks, info = U.load_prompts(a, _tt())
    assert prompts == p and blocks.shape == (21, 5) and info["source"] == "actions"
    assert all("moving right" in x for x in prompts[:11]) and all("moving left" in x for x in prompts[11:])
    (tmp_path / "p.json").write_text(json.dumps(p))
    prompts2, blocks2, info2 = U.load_prompts(types.SimpleNamespace(prompts=str(tmp_path / "p.json"), actions=None), _tt())
    assert prompts2 == p and blocks2 is None and info2["source"] == "prompts_json"
    (tmp_path / "short.json").write_text(json.dumps(p[:20]))
    with pytest.raises(ValueError, match="21"):
        U.load_prompts(types.SimpleNamespace(prompts=str(tmp_path / "short.json"), actions=None), _tt())
    (tmp_path / "bad.json").write_text(json.dumps(p[:20] + ["a cat"]))
    with pytest.raises(KeyError):
        U.load_prompts(types.SimpleNamespace(prompts=str(tmp_path / "bad.json"), actions=None), _tt())
    with pytest.raises(ValueError):
        U.load_prompts(types.SimpleNamespace(prompts=None, actions="D:10,A:9"), _tt())


def test_cond_stats():
    ids = _ids((5, 48, 64), seed=4)
    st = U.cond_stats(ids)
    assert abs(sum(st["class_frac"].values()) - 1) < 1e-9 and abs(sum(st["class_frac_frame0"].values()) - 1) < 1e-9
    assert set(st["class_frac"]) <= set(U.CLASS_NAMES.values())
    assert abs(st["unknown_frac_per_frame"]["mean"] - (ids == 255).mean()) < 1e-9


# ---------------------------------------------------------------- 真实 clip 导出
@pytest.mark.skipif(not (os.path.isfile(os.path.join(S.DATA_ROOT, "cond", CLIP + ".npz"))
                         and GEO.root_res(S.DATA_ROOT) == GEO.RES_TAG), reason="数据不在（或 res ≠ GC_RES）")
def test_export_clip_roundtrip(tmp_path):
    p = U.main(["--export_clip", CLIP, "--out", str(tmp_path)])
    c = S.load_clip(S.DATA_ROOT, CLIP)
    ids, _ = U.load_ids(p["ids.npz"], "parts")
    assert np.array_equal(ids, c["ids"])
    f0, i0 = U.load_first_frame(p["frame0.png"])
    fs, i1 = U.load_first_frame(p["frame0_src.png"])
    assert i0["resize"] == "as_is" and i1["resize"] == SRC_MODE and np.array_equal(f0, fs)
    with open(p["prompts.json"]) as fh:
        assert json.load(fh) == c["prompts"]
    assert TX.cell_prompts(INF.parse_actions(p["actions.npy"])) == c["prompts"]
