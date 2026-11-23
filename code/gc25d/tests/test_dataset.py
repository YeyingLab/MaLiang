"""dataset.GcLatentDataset / gc_collate / 世界锚定 UNKNOWN 遮挡的 CPU 测试。

合成数据根（latents / cond / meta / index / 文本表）+ 合成 UE 约定相机（行向量 4×4，正交 / 透视 reversed-Z）。
真实数据（data/gc25d_v10_mini 有 READY；GC_RES≠832x480 时用 data/gc25d_v10_<H>_mini）在就跑一遍形状与遮挡覆盖率，不在就 skip。
形状一律取自 dataset / geometry（GC_RES），合成数据根写 index/prep_args.json 的 res = 当前分辨率。

    PYTHONPATH=code/gc25d:code/arope .venv/bin/python -m pytest -q code/gc25d/tests/test_dataset.py
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import pytest
import torch

import dataset as DS
import geometry as GEO
import model as GM
import text as TXT

M_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MINI = os.path.join(M_ROOT, "data", "gc25d_v10_mini" if GEO.RES_TAG == GEO.DEFAULT_RES else f"gc25d_v10_{GEO.DST_H}_mini")
MINI_OK = os.path.isfile(os.path.join(MINI, "READY")) and GEO.root_res(MINI) == GEO.RES_TAG
T = DS.N_FRAMES
H, W = DS.IDS_SHAPE[1:]


# ----------------------------------------------------------------------------
# 合成相机（与 capsules.src_pixel_rays / geometry.proj 同一约定：clip = [X,Y,Z,1] @ M）
# ----------------------------------------------------------------------------

def view_matrix(target, pitch_deg: float, yaw_deg: float, dist: float) -> np.ndarray:
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    r = np.array([-math.sin(y), math.cos(y), 0.0])
    u = np.cross(f, r)                                   # 屏幕上方（Z 分量 > 0）
    c = np.asarray(target, np.float64) - f * dist
    V = np.eye(4)
    V[:3, 0], V[:3, 1], V[:3, 2] = r, u, f
    V[3, :3] = [-c @ r, -c @ u, -c @ f]
    return V


def ortho_proj(width_cm: float) -> np.ndarray:
    hh = width_cm * GEO.SRC_H / GEO.SRC_W
    return np.array([[2 / width_cm, 0, 0, 0], [0, 2 / hh, 0, 0], [0, 0, -2e-5, 0], [0, 0, 1, 1]], np.float64)


def persp_proj(fov_x_deg: float) -> np.ndarray:
    c = 1.0 / math.tan(math.radians(fov_x_deg) / 2)
    return np.array([[c, 0, 0, 0], [0, c * GEO.SRC_W / GEO.SRC_H, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], np.float64)


def make_meta(n: int = T, vel=(15.0, 8.0), ortho: bool = True, pitch: float = -50.0, yaw: float = 30.0,
              z_ground: float = 0.0, fov: float = 39.0) -> dict:
    """玩家在地面上匀速走、相机跟着平移（v10_real / P9C 那样居中）。"""
    w2c, w2v = [], []
    P = ortho_proj(1500.0) if ortho else persp_proj(fov)
    for t in range(n):
        tgt = (100.0 + vel[0] * t, -50.0 + vel[1] * t, z_ground)
        V = view_matrix(tgt, pitch, yaw, 3000.0 if ortho else 2200.0)
        w2v.append(V.ravel())
        w2c.append((V @ P).ravel())
    return {"world_to_clip": np.array(w2c), "world_to_view": np.array(w2v),
            "ground_z": np.full(n, z_ground, np.float32), "orthographic": np.array(ortho)}


def make_ids(seed: int = 0) -> np.ndarray:
    """静态类按 40 px 棋盘铺满，中间一个玩家（10、12..16）、右边一个 Boss（11、17..21）。"""
    yy, xx = np.mgrid[0:H, 0:W]
    base = (((xx // 40) + (yy // 40) + seed) % 7).astype(np.uint8)
    ids = np.broadcast_to(base, (T, H, W)).copy()
    for k, raw in enumerate(GM.PLAYER_PART_IDS):
        ids[:, 200 + 8 * k:208 + 8 * k, 400:430] = raw
    for k, raw in enumerate(GM.BOSS_PART_IDS):
        ids[:, 150 + 6 * k:156 + 6 * k, 600:640] = raw
    return ids


# ----------------------------------------------------------------------------
# 合成数据根
# ----------------------------------------------------------------------------

CLIPS = [  # (clip_id, split, run, drop)
    ("01_S__R_t1__000000", "train", "R_t1", False),
    ("01_S__R_t1__000096", "train", "R_t1", False),
    ("02_S__R_t2__000000", "train", "R_t2", True),
    ("02_S__R_t2__000096", "train", "R_t2", False),
    ("03_S__R_v1__000000", "val", "R_v1", False),
    ("03_S__R_v1__000096", "val", "R_v1", True),
]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("gc25d_v10_synth")
    for sub in ("index", "latents", "cond", "meta"):
        (r / sub).mkdir()
    keys = sorted(TXT.all_prompts() + [TXT.NULL_PROMPT])
    # 每个键一张可区分的 [512,4096]（值 = 键的排序下标）
    torch.save({k: torch.full(DS.CONTEXT_SHAPE, float(i), dtype=torch.bfloat16) for i, k in enumerate(keys)},
               r / DS.TEXT_TABLE_NAME)
    rows = []
    rng = np.random.default_rng(0)
    for i, (cid, split, run, drop) in enumerate(CLIPS):
        blocks20 = rng.integers(0, 2, size=(20, 5)).astype(np.uint8)
        torch.save({"latent": torch.randn(DS.LATENT_SHAPE).to(torch.bfloat16), "prompts": TXT.cell_prompts(blocks20),
                    "actions_block": torch.from_numpy(TXT.cell_blocks(blocks20)), "clip_id": cid},
                   r / "latents" / f"{cid}.pt")
        np.savez_compressed(r / "cond" / f"{cid}.npz", ids=make_ids(i))
        np.savez(r / "meta" / f"{cid}.npz", **make_meta())
        scene = cid.split("__")[0]
        rows.append({"clip_id": cid, "subset": "synth", "scene": scene, "run": run, "start": int(cid[-6:]),
                     "end": int(cid[-6:]) + 81, "split": split, "drop": drop, "drop_reason": "buried" if drop else ""})
    with open(r / "index" / "clips.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in rows) + "\n")
    with open(r / "index" / "prep_args.json", "w") as fh:
        json.dump({"res": GEO.RES_TAG}, fh)
    return str(r)


# ----------------------------------------------------------------------------
# 基本契约
# ----------------------------------------------------------------------------

def test_sample_contract(root):
    ds = DS.GcLatentDataset(root, "train", text_dropout=0.0, cond_dropout=0.0, mask_p=0.0)
    assert ds.clip_ids() == [c for c, s, _, d in CLIPS if s == "train" and not d]
    s = ds[0]
    assert s["input_latents"].dtype == torch.bfloat16 and tuple(s["input_latents"].shape) == DS.LATENT_SHAPE
    assert s["context"].dtype == torch.bfloat16 and tuple(s["context"].shape) == (21,) + DS.CONTEXT_SHAPE
    assert s["context_ids"].dtype == torch.int64 and tuple(s["context_ids"].shape) == (21,)
    assert s["sem_ids"].dtype == torch.uint8 and tuple(s["sem_ids"].shape) == DS.IDS_SHAPE
    assert s["clip_id"] == ds.items[0]
    assert s["dropped"] == {"text": False, "cond": False, "mask": 0.0}
    # key_to_id = 排序后的下标，且与 infer._KEY_TO_ID 的初值相同；context 行 = 文本表对应键
    keys = sorted(ds.text_table)
    assert ds.key_to_id == {k: i for i, k in enumerate(keys)}
    assert ds.key_to_id == {k: i for i, k in enumerate(sorted(TXT.all_prompts() + [TXT.NULL_PROMPT]))}
    prompts = torch.load(os.path.join(root, "latents", ds.items[0] + ".pt"), weights_only=False)["prompts"]
    assert s["context_ids"].tolist() == [ds.key_to_id[p] for p in prompts]
    for j in range(21):
        assert torch.equal(s["context"][j], ds.text_table[prompts[j]])
    # 没 dropout ⇒ ids 原样
    assert np.array_equal(s["sem_ids"].numpy(), make_ids(0))


def test_split_filter_limit(root):
    assert len(DS.GcLatentDataset(root, "train")) == 3
    assert len(DS.GcLatentDataset(root, "train", include_dropped=True)) == 4
    assert DS.GcLatentDataset(root, "val").clip_ids() == ["03_S__R_v1__000000"]
    a = DS.GcLatentDataset(root, "train", limit=2, seed=3).clip_ids()
    assert a == DS.GcLatentDataset(root, "train", limit=2, seed=3).clip_ids() and len(a) == 2 and a == sorted(a)
    with pytest.raises(ValueError):
        DS.GcLatentDataset(root, "test")


def test_val_never_drops(root):
    ds = DS.GcLatentDataset(root, "val", text_dropout=1.0, cond_dropout=1.0, mask_p=1.0)
    assert (ds.text_dropout, ds.cond_dropout, ds.mask_p) == (0.0, 0.0, 0.0)
    s = ds[0]
    assert s["dropped"] == {"text": False, "cond": False, "mask": 0.0}
    assert np.array_equal(s["sem_ids"].numpy(), make_ids(4))


def test_vocab_ids_collapses(root):
    s = DS.GcLatentDataset(root, "train", vocab="ids", text_dropout=0, cond_dropout=0, mask_p=0)[0]
    v = set(np.unique(s["sem_ids"].numpy()).tolist())
    assert v <= GM.valid_raw_ids("ids") and {10, 11} <= v
    assert np.array_equal(s["sem_ids"].numpy(), GM.collapse_parts(make_ids(0)))


# ----------------------------------------------------------------------------
# dropout：概率、独立性、确定性
# ----------------------------------------------------------------------------

def test_dropout_rates_and_independence(root):
    ds = DS.GcLatentDataset(root, "train")             # 默认 0.1 / 0.05 / 0.5
    n = 40000
    d = [ds.draw(ds.rng(i)) for i in range(n)]
    text = np.array([x["text"] for x in d])
    cond = np.array([x["cond"] for x in d])
    mask = np.array([x["mask"] for x in d])
    assert abs(text.mean() - 0.10) < 0.006, text.mean()
    assert abs(cond.mean() - 0.05) < 0.004, cond.mean()
    assert not (mask & cond).any()                      # 遮挡只在 cond 没 drop 时
    assert abs(mask[~cond].mean() - 0.5) < 0.01, mask[~cond].mean()
    assert abs((text & cond).mean() - 0.005) < 0.0015, (text & cond).mean()   # 独立：0.1 × 0.05


def test_dropout_deterministic_per_seed_epoch_index(root):
    ds = DS.GcLatentDataset(root, "train")
    a = [ds.draw(ds.rng(i)) for i in range(200)]
    assert a == [ds.draw(ds.rng(i)) for i in range(200)]
    ds.set_epoch(1)
    b = [ds.draw(ds.rng(i)) for i in range(200)]
    assert a != b
    ds2 = DS.GcLatentDataset(root, "train", seed=1)
    assert a != [ds2.draw(ds2.rng(i)) for i in range(200)]
    # 整个样本（含遮挡）逐位可复现
    ds3 = DS.GcLatentDataset(root, "train", mask_p=1.0, cond_dropout=0.0)
    s1, s2 = ds3[1], ds3[1]
    assert torch.equal(s1["sem_ids"], s2["sem_ids"]) and s1["dropped"] == s2["dropped"]


def test_getitem_applies_drops(root):
    s = DS.GcLatentDataset(root, "train", text_dropout=1.0, cond_dropout=0.0, mask_p=0.0)[0]
    ds = DS.GcLatentDataset(root, "train")
    assert s["dropped"]["text"] and s["context_ids"].tolist() == [ds.key_to_id[TXT.NULL_PROMPT]] * 21
    assert torch.equal(s["context"][5], ds.text_table[TXT.NULL_PROMPT])
    s = DS.GcLatentDataset(root, "train", text_dropout=0.0, cond_dropout=1.0, mask_p=1.0)[0]
    assert s["dropped"] == {"text": False, "cond": True, "mask": 0.0} and bool((s["sem_ids"] == 255).all())
    s = DS.GcLatentDataset(root, "train", text_dropout=0.0, cond_dropout=0.0, mask_p=1.0)[2]
    assert s["dropped"]["mask"] > 0 and not s["dropped"]["cond"]
    ids0, ids = make_ids(3), s["sem_ids"].numpy()
    changed = ids != ids0
    assert (ids[changed] == 255).all() and (ids0[changed] <= 6).all()
    assert abs(changed.mean() - s["dropped"]["mask"]) < 1e-9


# ----------------------------------------------------------------------------
# 世界锚定遮挡
# ----------------------------------------------------------------------------

def _pixel_of(w2c_t: np.ndarray, P: np.ndarray):
    """世界点 [n,3] → dst 像素（四舍五入，[n,2] int）与是否在画面内。"""
    src = GEO.proj(np.broadcast_to(w2c_t.reshape(4, 4), (len(P), 4, 4)), P)
    dst = GEO.src_to_dst(src)
    px = np.rint(dst).astype(np.int64)
    ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return px, ok


@pytest.mark.parametrize("ortho", [True, False])
def test_mask_world_anchored(ortho):
    meta = make_meta(ortho=ortho, pitch=-50.0 if ortho else -45.0)
    ids0 = make_ids(0)
    for seed in range(4):
        ids = ids0.copy()
        info = DS.apply_unknown_mask(ids, meta, random.Random(seed))
        assert info["planes"] and info["frac"] > 0
        masked = ids == 255
        # 世界里的地面点网格：离任一半平面边界都远（> 60 cm，比 4×4 格的地面尺寸大一个量级）的点
        gx, gy = np.meshgrid(np.linspace(-1500, 3000, 120), np.linspace(-1500, 2000, 100))
        P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
        beyond = np.zeros(len(P), bool)
        near = np.zeros(len(P), bool)
        for p in info["planes"]:
            s = P[:, :2] @ p["d"]
            beyond |= s > p["tau"]
            near |= np.abs(s - p["tau"]) < 60.0
        n_checked, seen = 0, np.zeros(len(P), np.int64)
        for t in range(0, T, 5):
            px, ok = _pixel_of(meta["world_to_clip"][t], P)
            sel = ok & ~near
            sel &= ids0[t, px[:, 1].clip(0, H - 1), px[:, 0].clip(0, W - 1)] <= 6   # 只看静态像素
            got = masked[t, px[sel, 1], px[sel, 0]]
            assert np.array_equal(got, beyond[sel]), (seed, t, int((got != beyond[sel]).sum()))
            n_checked += int(sel.sum())
            seen += sel
        assert n_checked > 5000
        # 同一个世界点在多帧里出现，遮挡状态一致（上面逐帧 = 世界判定，这里确认确有跨帧的点）
        assert (seen >= 3).sum() > 500


def test_mask_never_touches_characters():
    meta = make_meta()
    ids0 = make_ids(0)
    for seed in range(6):
        ids = ids0.copy()
        DS.apply_unknown_mask(ids, meta, random.Random(seed))
        char = ids0 >= 10
        assert np.array_equal(ids[char], ids0[char])
        changed = ids != ids0
        assert (ids[changed] == 255).all() and (ids0[changed] <= 6).all()
        for raw in GM.PLAYER_PART_IDS + GM.BOSS_PART_IDS:
            assert (ids == raw).sum() == (ids0 == raw).sum()


def _frame_cov(ids, ids0):
    """逐帧：静态像素里被改成 255 的占比 [T]。"""
    st = ids0 <= 6
    return np.array([(ids[t][st[t]] == 255).mean() for t in range(len(ids))])


def test_mask_coverage_in_range():
    meta = make_meta()
    ids0 = make_ids(0)
    n2 = 0
    for seed in range(40):
        ids = ids0.copy()
        info = DS.apply_unknown_mask(ids, meta, random.Random(seed))
        cov = _frame_cov(ids, ids0)
        qs = [p["q"] for p in info["planes"]]
        rng_q = DS.MASK_Q_RANGE if len(qs) == 1 else DS.MASK_Q_RANGE_2
        assert all(rng_q[0] <= q <= rng_q[1] for q in qs)
        cal = cov[DS.calib_frames(len(ids))]
        if len(qs) == 1:
            assert abs(cal.max() - qs[0]) < 0.02, (seed, cal.max(), qs)   # 覆盖最大的标定帧恰 ≈ q
        else:
            n2 += 1
            assert max(qs) - 0.02 <= cal.max() <= sum(qs) + 0.02, (seed, cal.max(), qs)
        assert cov.max() <= sum(qs) + 0.03, (seed, cov.max(), qs)        # 任一帧都不超过 q（非标定帧留点余量）
        assert 0.0 < info["frac"] < 0.6
    assert 4 <= n2 <= 22                                                  # p(2 个半平面) = 0.3


def test_mask_nan_ground_points_masked():
    # 透视、俯角很浅：画面上部的射线朝天，不交地面 ⇒ NaN ⇒ 遮挡开启时一律 255
    meta = make_meta(ortho=False, pitch=-8.0, vel=(0.0, 0.0))
    g = DS.grid_ground_points(meta["world_to_clip"][:1], meta["world_to_view"][:1], 0.0)
    nan_cells = np.isnan(g[0]).any(-1)
    assert nan_cells[0].all() and not nan_cells[-1].any()                  # 顶行朝天、底行看地
    ids = make_ids(0)[:1].copy()
    planes = [{"theta": 0.0, "q": 0.1, "d": np.array([1.0, 0.0]), "tau": 1e12}]   # 半平面本身什么都不遮
    m = DS.planes_mask(ids, g, planes)
    up = DS.upsample_grid(nan_cells)
    assert np.array_equal(m[0], up & (ids[0] <= 6))


def test_mask_speed_reasonable():
    meta, ids = make_meta(), make_ids(0)
    t0 = time.time()
    DS.apply_unknown_mask(ids, meta, random.Random(0))
    assert time.time() - t0 < 3.0                                           # worker 里每个样本的开销（实测 ~0.3 s）


# ----------------------------------------------------------------------------
# collate
# ----------------------------------------------------------------------------

def test_collate(root):
    ds = DS.GcLatentDataset(root, "train", text_dropout=1.0, cond_dropout=0.0, mask_p=0.0)
    b = DS.gc_collate([ds[0], ds[1]])
    assert tuple(b["input_latents"].shape) == (2,) + DS.LATENT_SHAPE
    assert tuple(b["context"].shape) == (2, 21) + DS.CONTEXT_SHAPE
    assert tuple(b["context_ids"].shape) == (2, 21) and b["context_ids"].dtype == torch.int64
    assert tuple(b["sem_ids"].shape) == (2,) + DS.IDS_SHAPE and b["sem_ids"].dtype == torch.uint8
    assert b["clip_id"] == ds.items[:2]
    assert b["dropped"]["text"].dtype == torch.bool and b["dropped"]["text"].tolist() == [True, True]
    assert b["dropped"]["mask"].dtype == torch.float32 and tuple(b["dropped"]["mask"].shape) == (2,)
    b1 = DS.gc_collate([ds[0]])
    assert tuple(b1["sem_ids"].shape) == (1,) + DS.IDS_SHAPE


# ----------------------------------------------------------------------------
# 真实数据（mini 集就绪才跑）
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not MINI_OK, reason=f"{os.path.basename(MINI)} 未就绪（或 res ≠ {GEO.RES_TAG}）")
def test_real_mini():
    ds = DS.GcLatentDataset(MINI, "train", text_dropout=0.0, cond_dropout=0.0, mask_p=0.0)
    s = ds[0]
    assert tuple(s["sem_ids"].shape) == DS.IDS_SHAPE and set(np.unique(s["sem_ids"].numpy()).tolist()) <= GM.valid_raw_ids("parts")
    assert s["context_ids"].max() < len(ds.text_table)
    cid = ds.items[0]
    with np.load(os.path.join(MINI, "meta", cid + ".npz")) as meta:
        meta = {k: meta[k] for k in ("world_to_clip", "world_to_view", "ground_z")}
    ids0 = s["sem_ids"].numpy()
    for seed in range(6):
        ids = ids0.copy()
        t0 = time.time()
        info = DS.apply_unknown_mask(ids, meta, random.Random(seed))
        dt = time.time() - t0
        cov = _frame_cov(ids, ids0)
        qs = [p["q"] for p in info["planes"]]
        assert qs and max(qs) - 0.03 <= cov.max() <= sum(qs) + 0.03, (cid, seed, cov.max(), qs)
        assert np.array_equal(ids[ids0 >= 10], ids0[ids0 >= 10])
        assert dt < 3.0
