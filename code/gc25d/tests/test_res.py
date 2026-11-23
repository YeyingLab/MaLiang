"""分辨率可配置（env GC_RES，geometry.py 是唯一出处）的门禁：默认 832x480 与改动前逐位相同；1280x704 在子进程里再跑一遍。

    PYTHONPATH=code/gc25d:code/arope .venv/bin/python -m pytest -q code/gc25d/tests/test_res.py

默认分辨率下（父进程）：
* test_default_constants_bit_identical   geometry 的旧名字 = 改动前写死的常数 / _IDX 表（逐位 + sha256），各模块派生的形状 = 旧值
* test_default_resample_equals_old_impl  resample_rgb / resample_ids = 旧实现（裁 [30,1250) + INTER_AREA / 旧索引表）
* test_capsules_rays_consistent_across_res  704p 子进程算的胶囊 / 部位 / 地面点，按 480p 的 _IDX 取样 = 480p 直接算的
* test_704_suite_subprocess              GC_RES=1280x704 子进程：dataset（合成）/ train tiny / gcdit + data_gc / stage1-3 dry run /
                                         prep / sample_user / firstframe_flux / 本文件的 704p 专项，全部要过
任何分辨率下：
* test_parse_res / test_parse_res_accepted_are_import_safe / test_listed_res_import_subprocess / test_guards /
  test_tiny_train_step_at_geometry / test_infer_sample_and_viz_shapes
GC_RES=1280x704 时（子进程里）：
* test_704_identity_paths / test_704_derived_constants
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import geometry as GEO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = GEO.RES_TAG == GEO.DEFAULT_RES
AT_704 = GEO.RES_TAG == "1280x704"

# 改动前 geometry.py 写死的值（832×480）
OLD = dict(CROP_X0=30, CROP_W=1220, SRC_W=1280, SRC_H=704, DST_W=832, DST_H=480)
OLD_IDX_X = 30 + ((2 * np.arange(832, dtype=np.int64) + 1) * 1220) // (2 * 832)
OLD_IDX_Y = ((2 * np.arange(480, dtype=np.int64) + 1) * 704) // (2 * 480)
OLD_IDX_SHA = ("e66923042dfe92847d3ac21f2692a8e8c34e6eaa147ded14b9faf379ee9aed75",
               "1b3f5f553549a9bf4647d59dca363bc3edee720ac59d861870d26652420fda0e")


def _sub_env(res: str) -> dict:
    env = dict(os.environ, GC_RES=res, DIFFSYNTH_SKIP_DOWNLOAD="True",
               PYTHONPATH=os.pathsep.join([os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")]))
    env.setdefault("TMPDIR", os.path.join(M, ".tmp"))
    return env


# ----------------------------------------------------------------------------
# 默认分辨率：逐位等于改动前
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not DEFAULT, reason="只在默认 GC_RES=832x480 下比旧常数")
def test_default_constants_bit_identical():
    for k, v in OLD.items():
        assert getattr(GEO, k) == v, k
    assert GEO.SX == 832 / 1220 and GEO.SY == 480 / 704
    assert np.array_equal(GEO.SCALE, np.array([832 / 1220, 480 / 704], np.float64))
    assert GEO.PLAYER_SRC == (639.5, 351.5) and GEO.PLAYER_DST == (415.5, 239.5)
    assert GEO.PLAYER_DST == tuple(float(v) for v in np.round(GEO.src_to_dst(GEO.PLAYER_SRC), 9))
    assert GEO._IDX_X.dtype == OLD_IDX_X.dtype and np.array_equal(GEO._IDX_X, OLD_IDX_X)
    assert np.array_equal(GEO._IDX_Y, OLD_IDX_Y)
    assert (hashlib.sha256(GEO._IDX_X.tobytes()).hexdigest(), hashlib.sha256(GEO._IDX_Y.tobytes()).hexdigest()) == OLD_IDX_SHA
    assert (GEO.RES_TAG, GEO.LAT_H, GEO.LAT_W, GEO.POOL_H, GEO.POOL_W, GEO.TOK_PER_CELL) == ("832x480", 30, 52, 120, 208, 390)
    assert not GEO.IDENTITY and GEO.AREA_VS_480 == 1.0
    # 各模块派生的形状 = 改动前写死的
    import dataset as DS
    import firstframe_flux as FF
    import infer as INF
    import prep_v10 as P
    import viz as VIZ
    from gcdistill import data_gc as DG
    from gcdistill import dmd_gc as S3
    assert DS.LATENT_SHAPE == (48, 21, 30, 52) and DS.IDS_SHAPE == (81, 480, 832) and (DS.GRID_H, DS.GRID_W) == (120, 208)
    assert (INF.LAT_C, INF.LAT_H, INF.LAT_W, INF.H, INF.W, INF.SRC_H, INF.SRC_W) == (48, 30, 52, 480, 832, 704, 1280)
    assert (VIZ.PANEL_H, VIZ.PANEL_W, VIZ.HEADER_H) == (480, 832, 48)
    assert P.LAT_SHAPE == (48, 21, 30, 52) and (P.DST_H, P.DST_W) == (480, 832)
    assert DG.LAT_HW == (30, 52)
    assert (FF.W_OUT, FF.H_OUT, FF.W_GEN, FF.H_GEN, FF.CROP_X0, FF.CROP_W, FF.MIN_PX) == (832, 480, 1248, 720, 30, 1220, 40)
    assert (S3.GIB_PER_GRAD_FRAME, S3.GIB_KV_CACHE, S3.GIB_SCORING_MAX) == (6.46, 2.72, 5.4)


@pytest.mark.skipif(not DEFAULT, reason="只在默认 GC_RES=832x480 下比旧实现")
def test_default_resample_equals_old_impl():
    rng = np.random.default_rng(0)
    rgb = cv2.GaussianBlur(rng.integers(0, 256, (704, 1280, 3)).astype(np.uint8), (0, 0), 1.0)
    old = cv2.resize(np.ascontiguousarray(rgb[:, 30:1250]), (832, 480), interpolation=cv2.INTER_AREA)
    assert np.array_equal(GEO.resample_rgb(rgb), old)
    ids = rng.integers(0, 22, (3, 704, 1280)).astype(np.uint8)
    assert np.array_equal(GEO.resample_ids(ids), ids[..., OLD_IDX_Y[:, None], OLD_IDX_X[None, :]])


# ----------------------------------------------------------------------------
# 任何分辨率
# ----------------------------------------------------------------------------

def test_parse_res():
    assert GEO.parse_res("832x480") == (832, 480) and GEO.parse_res(" 1280X704 ") == (1280, 704)
    assert GEO.parse_res("960x544") == (960, 544) and GEO.parse_res("1024x576") == (1024, 576)
    for bad in ("832", "832x", "abcx480", "816x480", "832x488", "1280x736", "1312x704", "1280x640", "0x0", "-832x480",
                "640x352", "608x320", "32x96"):                 # H < 384：缩小倍数 ≥ 2
        with pytest.raises(ValueError, match="GC_RES"):
            GEO.parse_res(bad)


def test_parse_res_accepted_are_import_safe():
    """parse_res 收下的每个 W×H：import 时的 _IDX 断言都成立（这里按同一公式模拟），CROP_W 偶数且裁剪恰好居中、宽高比误差
    ≤ 1 源 px；拒掉的都是点名 GC_RES 的 ValueError（不是 import 时的裸 AssertionError）。"""
    n_ok = 0
    for w in range(32, GEO.SRC_W + 1, 32):
        for h in range(32, GEO.SRC_H + 1, 32):
            try:
                GEO.parse_res(f"{w}x{h}")
            except ValueError as e:
                assert "GC_RES" in str(e)
                continue
            n_ok += 1
            cw = GEO.crop_w_for(w, h)
            x0 = (GEO.SRC_W - cw) // 2
            ix = x0 + ((2 * np.arange(w, dtype=np.int64) + 1) * cw) // (2 * w)
            iy = ((2 * np.arange(h, dtype=np.int64) + 1) * GEO.SRC_H) // (2 * h)
            assert (ix.min(), ix.max(), iy.min(), iy.max()) == (x0, x0 + cw - 1, 0, GEO.SRC_H - 1), (w, h)
            assert cw % 2 == 0 and 2 * x0 + cw == GEO.SRC_W and abs(cw - GEO.SRC_H * w / h) <= 1, (w, h, cw)
            assert h >= GEO.MIN_H
    assert n_ok == 335, n_ok


RES_LIST = ("832x480", "1280x704", "960x544", "1024x576", "1152x640", "672x384", "896x512")


def test_listed_res_import_subprocess():
    """列出的每个分辨率在子进程里真 import geometry + firstframe_flux（import 时的断言、两份规则一致、玩家 = 画幅几何中心）；
    非法值在 import 时报点名 GC_RES 的 ValueError。"""
    code = ("import geometry as G, firstframe_flux as FF; "
            "assert (FF.W_OUT, FF.H_OUT, FF.CROP_W, FF.CROP_X0) == (G.DST_W, G.DST_H, G.CROP_W, G.CROP_X0); "
            "assert G.PLAYER_DST == ((G.DST_W - 1) / 2, (G.DST_H - 1) / 2), G.PLAYER_DST; print(G.RES_TAG, G.CROP_W)")
    for s in RES_LIST:
        p = subprocess.run([sys.executable, "-c", code], env=_sub_env(s), capture_output=True, text=True, timeout=120, cwd=M)
        assert p.returncode == 0 and p.stdout.split()[0] == s, (s, p.stdout, p.stderr[-2000:])
    for bad in ("640x352", "1280x640"):
        p = subprocess.run([sys.executable, "-c", "import geometry"], env=_sub_env(bad), capture_output=True, text=True,
                           timeout=120, cwd=M)
        assert p.returncode != 0 and "ValueError" in p.stderr and "GC_RES" in p.stderr and "AssertionError" not in p.stderr


def test_guards(tmp_path):
    import prep_v10 as P
    from safetensors.torch import save_file
    from gcdistill import data_gc as DG
    other = "1280x704" if DEFAULT else GEO.DEFAULT_RES
    # 数据根：没有 prep_args.json / 没有 res 键 ⇒ 832x480（旧根）
    r0 = tmp_path / "r0"
    (r0 / "index").mkdir(parents=True)
    assert GEO.root_res(str(r0)) == GEO.LEGACY_RES
    (r0 / "index" / "prep_args.json").write_text(json.dumps({"argv": []}))
    assert GEO.root_res(str(r0)) == GEO.LEGACY_RES
    for res, ok in ((GEO.RES_TAG, True), (other, False)):
        (r0 / "index" / "prep_args.json").write_text(json.dumps({"res": res}))
        assert GEO.root_res(str(r0)) == res
        if ok:
            assert GEO.check_root_res(str(r0)) == res
            DG.check_res(root=str(r0))
        else:
            with pytest.raises(RuntimeError, match=f"GC_RES={GEO.RES_TAG}"):
                GEO.check_root_res(str(r0))
            with pytest.raises(SystemExit, match="GC_RES"):
                DG.check_res(root=str(r0))
            with pytest.raises(SystemExit, match="GC_RES"):
                P.check_out_res(str(r0))                         # prep 不往别的分辨率的数据根里写
    # prep 的 index 行：缺 res ⇒ 832x480
    r1 = tmp_path / "r1"
    (r1 / "index").mkdir(parents=True)
    P.check_out_res(str(tmp_path / "fresh"))                     # 新目录：放行
    (r1 / "index" / "rows_rank0.jsonl").write_text(json.dumps({"clip_id": "a", "res": other}) + "\n")
    with pytest.raises(SystemExit, match="GC_RES"):
        P.check_out_res(str(r1))
    (r1 / "index" / "rows_rank0.jsonl").write_text(json.dumps({"clip_id": "a"}) + "\n")
    if DEFAULT:
        P.check_out_res(str(r1))
    else:
        with pytest.raises(SystemExit):
            P.check_out_res(str(r1))
    # ckpt：metadata res；缺失 ⇒ 832x480
    t = {"w": torch.zeros(2)}
    save_file(t, str(tmp_path / "old.safetensors"))
    save_file(t, str(tmp_path / "mine.safetensors"), metadata={"res": GEO.RES_TAG, "format": "gc25d"})
    save_file(t, str(tmp_path / "other.safetensors"), metadata={"res": other})
    assert GEO.ckpt_res(str(tmp_path / "old.safetensors")) == GEO.LEGACY_RES
    assert GEO.check_ckpt_res(str(tmp_path / "mine.safetensors")) == GEO.RES_TAG
    with pytest.raises(RuntimeError, match=f"GC_RES={GEO.RES_TAG}"):
        GEO.check_ckpt_res(str(tmp_path / "other.safetensors"), "--ckpt")
    with pytest.raises(SystemExit, match="--teacher_ckpt"):
        DG.check_res(ckpts=[("--student_ckpt", str(tmp_path / "mine.safetensors")),
                            ("--teacher_ckpt", str(tmp_path / "other.safetensors"))])
    if DEFAULT:
        GEO.check_ckpt_res(str(tmp_path / "old.safetensors"))     # 旧 ckpt 在默认分辨率下照收
    else:
        with pytest.raises(RuntimeError):
            GEO.check_ckpt_res(str(tmp_path / "old.safetensors"))
    # save_gc 恒写 res（调用方显式给的优先）
    from gcdistill.gc_dit import save_gc
    m = DG.tiny_gcdit(causal=False, dtype=torch.float32)
    save_gc(None, m, str(tmp_path / "g.safetensors"), {"stage": "x"}, dry_run=True)
    assert GEO.ckpt_res(str(tmp_path / "g.safetensors")) == GEO.RES_TAG
    save_gc(None, m, str(tmp_path / "h.safetensors"), {"res": other}, dry_run=True)
    assert GEO.ckpt_res(str(tmp_path / "h.safetensors")) == other
    # 蒸馏 run 目录：config.json 的 res（缺失 = 832x480）；没有 config.json ⇒ 新目录，放行
    rd = tmp_path / "run"
    rd.mkdir()
    assert GEO.check_run_dir_res(str(rd)) is None
    (rd / "config.json").write_text(json.dumps({"stage": "x"}))
    if DEFAULT:
        assert GEO.check_run_dir_res(str(rd)) == GEO.LEGACY_RES
    else:
        with pytest.raises(RuntimeError, match="GC_RES"):
            GEO.check_run_dir_res(str(rd))
    (rd / "config.json").write_text(json.dumps({"res": GEO.RES_TAG}))
    DG.check_res(run_dir=str(rd))
    # 三个 runner 都在打开 train.log 之前拒掉另一个分辨率的 run 目录（chain 指错 D 时不往别人的 run 里追加日志）
    from gcdistill import ar_tf_gc as S1
    from gcdistill import cd_gc as S2
    from gcdistill import dmd_gc as S3
    for i, mod in enumerate((S1, S2, S3)):
        rd = tmp_path / f"run_other{i}"
        rd.mkdir()
        (rd / "config.json").write_text(json.dumps({"res": other}))
        with pytest.raises(SystemExit, match="GC_RES"):
            mod.main(["--dry_run", "--output", str(rd), "--num_workers", "0"])
        assert sorted(os.listdir(rd)) == ["config.json"], (mod.__name__, os.listdir(rd))


def test_tiny_train_step_at_geometry():
    """tiny DiT（dim 24）在**当前分辨率的真实 latent / 语义图网格**上走 GcTrainModule 前向 + 反向（2 个 cell = 5 帧）。"""
    import model as GM
    import train as TR
    from diffsynth.models.wan_video_dit import WanModel
    torch.manual_seed(0)
    dit = WanModel(dim=24, in_dim=48, ffn_dim=48, out_dim=48, text_dim=4096, freq_dim=256, eps=1e-6,
                   patch_size=(1, 2, 2), num_heads=2, num_layers=1, has_image_input=False).float()
    GM.install_sem_cond(dit, "parts")
    enc = getattr(dit, GM.SEM_COND_NAME)
    with torch.no_grad():
        enc.proj_out.weight.normal_(0, 0.02)                 # 打开条件通路（零初始化时 sem 项恒 0）
    m = TR.GcTrainModule(dit.train().requires_grad_(True), first_frame_cond=True, gradient_checkpointing=False)
    g = torch.Generator().manual_seed(0)
    f = 2
    raw = torch.tensor(sorted(GM.valid_raw_ids("parts")), dtype=torch.uint8)
    sem = raw[torch.randint(0, len(raw), (1, 4 * f - 3, GEO.POOL_H, GEO.POOL_W), generator=g)]
    sem = sem.repeat_interleave(4, 2).repeat_interleave(4, 3)
    assert tuple(sem.shape) == (1, 4 * f - 3, GEO.DST_H, GEO.DST_W)
    batch = {"input_latents": torch.randn(1, 48, f, GEO.LAT_H, GEO.LAT_W, generator=g),
             "context": torch.randn(1, f, 6, 4096, generator=g), "context_ids": torch.arange(f)[None], "sem_ids": sem}
    loss, st = m(batch, timestep_id=400)
    assert torch.isfinite(loss) and st["sem_rms"].item() > 0
    loss.backward()
    assert enc.proj_in.weight.grad is not None and enc.proj_in.weight.grad.abs().sum() > 0
    with pytest.raises(AssertionError):                       # 语义图分辨率与 latent 不符 ⇒ gc_forward 的形状闸
        m(dict(batch, sem_ids=sem[..., :-16, :]), timestep_id=400)


def test_infer_sample_and_viz_shapes():
    import infer as INF
    import viz as VIZ
    z0 = torch.zeros(1, 48, 1, GEO.LAT_H, GEO.LAT_W, dtype=torch.bfloat16)
    ctx = torch.zeros(1, 21, 512, 4096, dtype=torch.bfloat16)
    sem = torch.zeros(1, 81, GEO.DST_H, GEO.DST_W, dtype=torch.uint8)
    seen = []

    def fwd(dit, lat, t, c, s, **kw):
        seen.append(tuple(lat.shape))
        return torch.zeros_like(lat)
    out = INF.sample(None, z0, ctx, None, sem, steps=2, shift=5.0, cfg=1.0, sem_cfg=1.0, seed=0, forward=fwd,
                     log=lambda *a: None)
    assert tuple(out.shape) == (1, 48, 21, GEO.LAT_H, GEO.LAT_W) == seen[0] and torch.equal(out[:, :, :1], z0)
    with pytest.raises(AssertionError):
        INF.sample(None, z0[..., :-2], ctx, None, sem, steps=1, shift=5.0, cfg=1.0, sem_cfg=1.0, seed=0, forward=fwd)
    gen = np.zeros((2, GEO.DST_H, GEO.DST_W, 3), np.uint8)
    sbs = VIZ.SideBySide(gen, np.zeros((2, GEO.DST_H, GEO.DST_W), np.uint8), gt=gen, overlay=True)
    fr = sbs.frame(1)
    assert fr.shape == (GEO.DST_H + 48, 4 * GEO.DST_W, 3) and fr.shape[0] % 16 == 0 and fr.shape[1] % 16 == 0


# ----------------------------------------------------------------------------
# GC_RES=1280x704（子进程里执行）
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not AT_704, reason="GC_RES=1280x704 时才跑（test_704_suite_subprocess 会带上）")
def test_704_identity_paths():
    assert GEO.IDENTITY and (GEO.CROP_X0, GEO.CROP_W, GEO.SX, GEO.SY) == (0, 1280, 1.0, 1.0)
    assert np.array_equal(GEO._IDX_X, np.arange(1280)) and np.array_equal(GEO._IDX_Y, np.arange(704))
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (704, 1280, 3)).astype(np.uint8)
    out = GEO.resample_rgb(rgb)
    general = cv2.resize(np.ascontiguousarray(rgb[:, GEO.CROP_X0:GEO.CROP_X0 + GEO.CROP_W]), (GEO.DST_W, GEO.DST_H),
                         interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, general) and np.array_equal(out, rgb) and out.flags.c_contiguous
    out[0, 0] ^= 1
    assert not np.array_equal(out, rgb), "快路径必须返回拷贝，不是视图"
    ids = rng.integers(0, 22, (2, 704, 1280)).astype(np.uint8)
    o = GEO.resample_ids(ids)
    assert np.array_equal(o, ids[..., GEO._IDX_Y[:, None], GEO._IDX_X[None, :]]) and not np.shares_memory(o, ids)
    xy = rng.uniform(-10, 1300, (50, 2))
    assert np.array_equal(GEO.src_to_dst(xy), xy) and np.allclose(GEO.dst_to_src(xy), xy)
    assert GEO.PLAYER_DST == GEO.PLAYER_SRC == (639.5, 351.5)


@pytest.mark.skipif(not AT_704, reason="GC_RES=1280x704 时才跑")
def test_704_derived_constants():
    import dataset as DS
    import firstframe_flux as FF
    import infer as INF
    import prep_v10 as P
    import viz as VIZ
    from gcdistill import data_gc as DG
    from gcdistill import dmd_gc as S3
    assert (GEO.RES_TAG, GEO.LAT_H, GEO.LAT_W, GEO.POOL_H, GEO.POOL_W, GEO.TOK_PER_CELL) == ("1280x704", 44, 80, 176, 320, 880)
    assert DS.LATENT_SHAPE == (48, 21, 44, 80) and DS.IDS_SHAPE == (81, 704, 1280) and (DS.GRID_H, DS.GRID_W) == (176, 320)
    assert (INF.LAT_H, INF.LAT_W, INF.H, INF.W) == (44, 80, 704, 1280) and P.LAT_SHAPE == (48, 21, 44, 80)
    assert (VIZ.PANEL_H, VIZ.PANEL_W) == (704, 1280) and (VIZ.PANEL_H + VIZ.HEADER_H) % 16 == 0
    assert DG.LAT_HW == (44, 80)
    assert (FF.W_OUT, FF.H_OUT, FF.W_GEN, FF.H_GEN, FF.CROP_X0, FF.CROP_W) == (1280, 704, 1280, 704, 0, 1280)
    assert S3.TOK_SCALE == 880 / 390 and S3.GIB_KV_CACHE == 2.72 * 880 / 390 and S3.GIB_SHARDED_TOTAL == 167.7


def test_firstframe_flux_copy_matches_geometry():
    """firstframe_flux（PCE env 里跑，import 不了 geometry）抄的 GC_RES 规则与 geometry 逐项相同；生成网格 = W:H、16 的倍数。"""
    import firstframe_flux as FF
    assert (FF.W_OUT, FF.H_OUT, FF.CROP_W, FF.CROP_X0) == (GEO.DST_W, GEO.DST_H, GEO.CROP_W, GEO.CROP_X0)
    for s in RES_LIST:
        w, h = GEO.parse_res(s)
        assert FF.parse_res(s) == (w, h)
        gw, gh = FF.gen_grid(w, h)
        assert gw % 16 == 0 and gh % 16 == 0 and gw * h == gh * w and gw >= w and gh >= h
        assert 0.5e6 < gw * gh < 1.6e6, (s, gw, gh)
    for bad in ("816x480", "1280x640", "1312x704", "640x352"):
        with pytest.raises(ValueError):
            FF.parse_res(bad)
        with pytest.raises(ValueError):
            GEO.parse_res(bad)
    assert FF.gen_grid(832, 480) == (1248, 720) and FF.gen_grid(1280, 704) == (1280, 704)


# ----------------------------------------------------------------------------
# 跨分辨率：704p 子进程 vs 默认 480p
# ----------------------------------------------------------------------------

_CAPS_SCRIPT = r'''
import math, sys
import numpy as np
import capsules as CAP, geometry as GEO, model as GM

def view(target, pitch, yaw, dist):
    p, y = math.radians(pitch), math.radians(yaw)
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    r = np.array([-math.sin(y), math.cos(y), 0.0]); u = np.cross(f, r)
    c = np.asarray(target, float) - f * dist
    V = np.eye(4); V[:3, 0], V[:3, 1], V[:3, 2] = r, u, f; V[3, :3] = [-c @ r, -c @ u, -c @ f]
    return V

def cams():
    out = []
    for ortho in (True, False):
        V = view((0.0, 0.0, 0.0), -50.0, 30.0, 3000.0 if ortho else 2200.0)
        if ortho:
            w = 1500.0; hh = w * GEO.SRC_H / GEO.SRC_W
            P = np.array([[2 / w, 0, 0, 0], [0, 2 / hh, 0, 0], [0, 0, -2e-5, 0], [0, 0, 1, 1]], float)
        else:
            c = 1.0 / math.tan(math.radians(39.0) / 2)
            P = np.array([[c, 0, 0, 0], [0, c * GEO.SRC_W / GEO.SRC_H, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
        out.append(((V @ P).ravel(), V.ravel()))
    return out

CAPS = [(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 120.0]), 30.0, 10, 0),
        (np.array([0.0, 0.0, 120.0]), np.array([40.0, 10.0, 160.0]), 12.0, 10, 1),
        (np.array([10.0, 0.0, 60.0]), np.array([-60.0, -20.0, 90.0]), 10.0, 10, 5),
        (np.array([260.0, 80.0, 0.0]), np.array([260.0, 80.0, 180.0]), 45.0, 11, 0),
        (np.array([260.0, 80.0, 180.0]), np.array([300.0, 40.0, 120.0]), 15.0, 11, 2)]

def run():
    res = {}
    ys, xs = np.mgrid[0:GEO.DST_H, 0:GEO.DST_W]
    for k, (w2c, w2v) in enumerate(cams()):
        with np.errstate(invalid="ignore"):
            # pad 64 + 旧包围盒（端点 ± r 沿世界轴）= 不裁的参考：斜视下世界轴少算球的屏幕半径，默认 pad 2 时旧包围盒
            # 在两种分辨率下裁掉的边不同；这里先验"同一源像素 ⇒ 同一射线"
            rend = CAP.render_capsules(np.zeros((GEO.DST_H, GEO.DST_W), np.uint8), w2c, w2v, CAPS, pad_px=64.0)
            ids = GM.collapse_parts(rend)
            res[f"rend{k}"] = rend
            # 默认 pad 2：相机轴包围盒（cam_axes=True）/ 默认（832×480 关、其它开）/ 旧包围盒（cam_axes=False）
            z = np.zeros((GEO.DST_H, GEO.DST_W), np.uint8)
            res[f"rendc{k}"] = CAP.render_capsules(z, w2c, w2v, CAPS, cam_axes=True)
            res[f"rendd{k}"] = CAP.render_capsules(z, w2c, w2v, CAPS)
            res[f"rendo{k}"] = CAP.render_capsules(z, w2c, w2v, CAPS, cam_axes=False)
            res[f"label{k}"] = CAP.label_parts(ids, w2c, w2v, CAPS)
            res[f"ids{k}"] = ids
        res[f"gp{k}"] = CAP.ground_points(w2c, w2v, xs.ravel(), ys.ravel(), 0.0).reshape(GEO.DST_H, GEO.DST_W, 2)
    return res

if __name__ == "__main__":
    np.savez(sys.argv[1], **run())
'''


@pytest.mark.skipif(not DEFAULT, reason="父进程在默认分辨率下比")
def test_capsules_rays_consistent_across_res(tmp_path):
    """同一源像素中心发同一条射线：704p 上逐源像素算的部位 / 胶囊 / 地面点，按 480p 的 _IDX 取样 = 480p 上直接算的。"""
    script = tmp_path / "caps_res.py"
    script.write_text(_CAPS_SCRIPT)
    out = tmp_path / "caps704.npz"
    p = subprocess.run([sys.executable, str(script), str(out)], env=_sub_env("1280x704"), capture_output=True,
                       text=True, timeout=600, cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr[-3000:]
    sys.path.insert(0, str(tmp_path))
    try:
        import caps_res as C
    finally:
        sys.path.remove(str(tmp_path))
    here = C.run()
    with np.load(out) as z:
        far = {k: z[k] for k in z.files}
    iy, ix = GEO._IDX_Y[:, None], GEO._IDX_X[None, :]
    for k in range(2):
        assert (here[f"rend{k}"] > 0).sum() > 500, "合成胶囊没画进画面"
        assert np.array_equal(far[f"rend{k}"][iy, ix], here[f"rend{k}"]), k
        # 相机轴包围盒 + 默认 pad 2：两种分辨率都不再裁角色（= pad 64）；默认值 704p 开、832×480 关（与旧包围盒逐位相同）
        assert np.array_equal(here[f"rendc{k}"], here[f"rend{k}"]) and np.array_equal(far[f"rendc{k}"], far[f"rend{k}"]), k
        assert np.array_equal(far[f"rendd{k}"], far[f"rendc{k}"]) and np.array_equal(here[f"rendd{k}"], here[f"rendo{k}"]), k
        # label_parts：先在 704p 标再取样 = 先取样再在 480p 标（输入 ids 是同一份剪影的取样）
        assert np.array_equal(far[f"ids{k}"][iy, ix], here[f"ids{k}"])
        assert np.array_equal(far[f"label{k}"][iy, ix], here[f"label{k}"]), k
        assert len(set(np.unique(here[f"label{k}"]).tolist()) - {0}) >= 4, "部位没分出来"
        # 地面点（UNKNOWN 遮挡 / ext_canvas 用）：同一源像素 ⇒ 同一交点
        a, b = far[f"gp{k}"][iy, ix], here[f"gp{k}"]
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.isfinite(b).mean() > 0.5
        assert np.allclose(a, b, rtol=0, atol=1e-6, equal_nan=True), float(np.nanmax(np.abs(a - b)))


@pytest.mark.skipif(not DEFAULT, reason="父进程在默认分辨率下发起；子进程里不再递归")
def test_704_suite_subprocess():
    """GC_RES=1280x704 下把分辨率相关的 CPU 测试整套再跑一遍（合成数据 / tiny 模型 / dry run；真数据用例按 res 自动 skip）。"""
    files = ["test_res.py", "test_dataset.py", "test_train.py", "test_gcdit.py", "test_stage1.py", "test_stage2.py",
             "test_stage3.py", "test_prep.py", "test_sample_user.py", "test_firstframe_flux.py", "test_procmap.py"]
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + [os.path.join(HERE, f) for f in files]
    p = subprocess.run(cmd, env=_sub_env("1280x704"), capture_output=True, text=True, timeout=1800, cwd=M)
    tail = (p.stdout + p.stderr)[-4000:]
    assert p.returncode == 0 and " passed" in p.stdout, tail
