"""prep_v10 的快速单测（CPU）：块 / prompt 对齐、扩展有效性、split 规则、原子写 / resume / merge。

    PYTHONPATH=code/gc25d:code/arope .venv/bin/python -m pytest -q code/gc25d/tests/test_prep.py
最后一个用例在真实数据上跑一个 clip 的 worker（--no_vae 路径，≈10 s），数据不在就跳过。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

import geometry as GEO
import prep_v10 as P
import text as TX

N_FRAMES, N_BLOCKS = 4800, 1200


def synth_blocks(n=N_BLOCKS) -> np.ndarray:
    """block b 的 [w,a,s,d,attack] = b 的低 5 位：每块各不相同（模 32），对齐错一位就能看出来。"""
    b = np.arange(n)
    return np.stack([(b >> k) & 1 for k in range(5)], 1).astype(np.uint8)


# ---------------------------------------------------------------- 块 / prompt 对齐
@pytest.mark.parametrize("s", [0, 96, 1536, 4608])
def test_block_prompt_alignment(s):
    AB = synth_blocks()
    b20 = P.clip_blocks(AB, s)
    assert (b20 == AB[s // 4: s // 4 + 20]).all()
    cb = TX.cell_blocks(b20)
    assert cb.shape == (21, 5) and (cb[0] == cb[1]).all()
    for j in range(1, 21):                      # cell j ← block s//4 + j − 1（帧 4j−3 … 4j 由该块的按键产生）
        assert (cb[j] == AB[s // 4 + j - 1]).all()
    pr = TX.cell_prompts(b20)
    assert pr == [TX.block_to_prompt(*r.tolist()) for r in cb]
    eb = P.ext_blocks(AB, s)                    # 扩展 cell 21..27 ← block s//4+20 … s//4+26
    assert eb.shape == (7, 5)
    for k, j in enumerate(range(21, 28)):
        assert (eb[k] == AB[s // 4 + j - 1]).all()
    # cell 21..27 覆盖帧 81..108（= cond_ext 的 28 帧）
    assert (4 * 21 - 3, 4 * 27) == (P.F_CLIP, P.F_SPAN - 1) and P.F_EXT == 4 * P.N_EXT_CELLS


def test_ext_blocks_past_run_end():
    AB = synth_blocks()
    with pytest.raises(AssertionError):
        P.ext_blocks(AB, 4704)                  # 最后一个 clip：块 1196..1202 越界


# ---------------------------------------------------------------- 扩展有效性
def _ep_cam():
    ep = np.zeros(N_FRAMES, np.int32)
    cam = np.ones(N_FRAMES, np.int32)
    return ep, cam


def _valid(ep, cam, s, n_blocks=N_BLOCKS):
    return P.ext_valid(ep[s:s + P.F_SPAN], cam[s:s + P.F_SPAN], s, n_blocks)


def test_ext_valid_constant():
    ep, cam = _ep_cam()
    assert _valid(ep, cam, 0) and _valid(ep, cam, 96 * 48)          # 4608 + 108 = 4716 < 4800
    assert not _valid(ep, cam, 96 * 49)                              # 4704：帧 4812 越界（窗口不满 109）


@pytest.mark.parametrize("off,ok", [(81, False), (100, False), (108, False), (109, True), (1, False)])
def test_ext_valid_episode_change(off, ok):
    s = 960
    ep, cam = _ep_cam()
    ep[s + off:] = 1
    assert _valid(ep, cam, s) == ok


@pytest.mark.parametrize("off,ok", [(81, False), (95, False), (108, False), (109, True)])
def test_ext_valid_camera_change(off, ok):
    s = 960
    ep, cam = _ep_cam()
    cam[s + off:] = 2
    assert _valid(ep, cam, s) == ok


def test_ext_valid_blocks_and_short_window():
    ep, cam = _ep_cam()
    s = 960
    assert _valid(ep, cam, s, n_blocks=s // 4 + 27)
    assert not _valid(ep, cam, s, n_blocks=s // 4 + 26)              # 块 s//4+26 不存在
    assert not P.ext_valid(ep[s:s + 108], cam[s:s + 108], s, N_BLOCKS)


# ---------------------------------------------------------------- split / drop
def test_split_rule():
    val = ["R10_s1512", "R10_s3512", "R10_s5512", "R10_s7511", "R10_s9511",
           "P9_golden_s2909", "P9_autumn_s4908", "P9C_golden_s6946", "P9C_default_s8946", "P9C_snow_s10945"]
    assert set(val) == P.VAL_RUNS
    assert all(P.split_of(r) == "val" for r in val)
    for r in ("R10_s1501", "R10_s2512", "R10_s10511", "P9C_autumn_s1941", "P9_default_s1909", "P9C_snow_s10944"):
        assert P.split_of(r) == "train"
    assert all(r in P.SPLIT_RULE for r in val)


def _fake_src(tmp_path, runs):
    """假 src：<subset>/<scene>/<run>/index/{clips_81.json, episodes.npy}。"""
    src = tmp_path / "src"
    for sub, scene, run in runs:
        d = src / sub / scene / run / "index"
        d.mkdir(parents=True)
        clips = [{"start": 96 * k, "end": 96 * k + 81, "episode": 0, "camera_variant": 1, "look": 0, "look_name": "x",
                  "centered": True, "max_frame_offset_cm": 0.0, "events": [{}] * (k % 2)} for k in range(50)]
        (d / "clips_81.json").write_text(json.dumps(clips))
        np.save(d / "episodes.npy", np.zeros(N_FRAMES, np.int32))
    return str(src)


def test_plan_split_drop(tmp_path):
    src = _fake_src(tmp_path, [("v9_stylized", "02_B", "P9_golden_s2909"), ("v10_real", "01_A", "R10_s1501"),
                               ("v10_real", "01_A", "R10_s1512")])
    runs = P.list_runs(src)
    assert [r[2] for r in runs] == ["R10_s1501", "R10_s1512", "P9_golden_s2909"]   # v10_real 在前
    assert [r[2] for r in P.list_runs(src, ["P9_golden_s2909"])] == ["P9_golden_s2909"]
    with pytest.raises(SystemExit):
        P.list_runs(src, ["nope"])
    buried = {P.clip_id_of("01_A", "R10_s1501", 192)}
    tasks = P.plan_tasks(runs, None, buried)
    assert len(tasks) == 150 and len({t["clip_id"] for t in tasks}) == 150
    by = {t["clip_id"]: t for t in tasks}
    t = by["01_A__R10_s1501__000192"]
    assert t["drop"] and t["drop_reason"] == P.DROP_REASON and t["split"] == "train" and t["n_events"] == 0
    assert by["01_A__R10_s1512__000096"]["split"] == "val" and not by["01_A__R10_s1512__000096"]["drop"]
    assert by["01_A__R10_s1512__000096"]["n_events"] == 1
    assert {t["split"] for t in tasks if t["run"] == "P9_golden_s2909"} == {"val"}
    sub = P.plan_tasks(runs[:1], 4)
    assert [t["start"] for t in sub] == [0, 96 * 16, 96 * 33, 96 * 49]                # 均匀取，含首尾


# ---------------------------------------------------------------- 原子写 / resume / merge
def test_atomic_savez_and_torch_save(tmp_path):
    p = str(tmp_path / "a.npz")
    P.atomic_savez(p, ids=np.arange(10, dtype=np.uint8), s=np.array(["x", "yy"]))
    with np.load(p, allow_pickle=False) as z:
        assert (z["ids"] == np.arange(10)).all() and z["s"].tolist() == ["x", "yy"]
    q = str(tmp_path / "b.pt")
    P.atomic_torch_save(q, {"latent": torch.zeros(2, dtype=torch.bfloat16), "prompts": ["a"]})
    assert torch.load(q, weights_only=True)["prompts"] == ["a"]
    assert sorted(os.listdir(tmp_path)) == ["a.npz", "b.pt"]                          # 没有残留 tmp


def _row(cid, run="R10_s1501", scene="01_A", start=0, has_ext=True, subset="v10_real"):
    return {"clip_id": cid, "subset": subset, "scene": scene, "run": run, "start": start, "split": P.split_of(run),
            "drop": False, "has_ext": has_ext, "res": GEO.RES_TAG}


def test_resume_and_merge(tmp_path):
    out = str(tmp_path / "out")
    for d in ("index", "latents", "cond", "cond_ext", "meta"):
        os.makedirs(os.path.join(out, d))
    a, b, c = (P.clip_id_of("01_A", "R10_s1501", s) for s in (0, 96, 192))
    tasks = [{"clip_id": x} for x in (a, b, c)]
    pa, pb = P.clip_paths(out, a), P.clip_paths(out, b)
    for k in ("latent", "cond", "cond_ext", "meta"):
        open(pa[k], "wb").close()
    for k in ("cond", "meta"):
        open(pb[k], "wb").close()
    open(pb["latent"] + ".tmp123", "wb").close()           # 被打断的 latent 写入
    with open(os.path.join(out, "index", "rows_rank0.jsonl"), "w") as fh:
        fh.write(json.dumps(_row(a, start=0)) + "\n" + json.dumps(_row(b, start=96, has_ext=False)) + "\n")
        fh.write('{"clip_id": "trunc')                      # 被杀时的半行
    assert P.outputs_exist(pa, True, True) and not P.outputs_exist(pb, False, True) and P.outputs_exist(pb, False, False)
    assert P.done_clips(out, tasks, vae=True) == {a}
    assert P.done_clips(out, tasks, vae=False) == {a, b}
    # has_ext 的 clip 缺 cond_ext ⇒ 没完成
    os.remove(pa["cond_ext"])
    assert P.done_clips(out, tasks, vae=True) == set()
    # merge：两个 rank 文件、重复行去重、按 (subset, scene, run, start) 排序
    with open(os.path.join(out, "index", "rows_rank1.jsonl"), "w") as fh:
        fh.write(json.dumps(_row(c, start=192)) + "\n" + json.dumps(_row(a, start=0)) + "\n")
        fh.write(json.dumps(_row("02_B__P9_golden_s2909__000000", run="P9_golden_s2909", scene="02_B",
                                 subset="v9_stylized")) + "\n")
    with pytest.raises(SystemExit):
        P.merge(out, expect=[a, b, c, "missing"])
    rows = P.merge(out, expect=[a, b, c])
    assert [r["clip_id"] for r in rows] == [a, b, c, "02_B__P9_golden_s2909__000000"]
    lines = open(os.path.join(out, "index", "clips.jsonl")).read().splitlines()
    assert len(lines) == 4
    runs = json.load(open(os.path.join(out, "index", "runs.json")))
    assert runs["R10_s1501"] == "train" and runs["P9_golden_s2909"] == "val" and "_rule" in runs


def test_open_rows_seals_torn_line(tmp_path):
    """上次被杀留下半行 ⇒ resume 追加的第一行不能粘在半行后面（否则 merge 报缺行）。"""
    out = str(tmp_path / "out")
    os.makedirs(os.path.join(out, "index"))
    a, x = (P.clip_id_of("01_A", "R10_s1501", s) for s in (0, 96))
    p = os.path.join(out, "index", "rows_rank0.jsonl")
    with open(p, "w") as fh:
        fh.write(json.dumps(_row(a)) + "\n" + json.dumps(_row(x, start=96))[:40])
    for _ in range(2):                                     # 第二次打开：文件已以 '\n' 结尾，不再补空行
        with P.open_rows(p) as fh:
            fh.write(json.dumps(_row(x, start=96)) + "\n")
    assert set(P.read_rows(out)) == {a, x}
    assert open(p).read().count("\n") == 4                 # A、半行封口、X、X
    assert [r["clip_id"] for r in P.merge(out, expect=[a, x])] == [a, x]
    q = os.path.join(out, "index", "rows_rank1.jsonl")     # 新文件 / 空文件：不补
    with P.open_rows(q) as fh:
        pass
    assert open(q).read() == ""


def test_check_raw_ids():
    raw = np.zeros((704, 1280), np.uint8)
    raw[:, :7] = np.arange(7)
    raw[0, 100], raw[0, 101] = 10, 11
    P.check_raw_ids(raw, "ok")
    for v in (7, 12, 16, 21, 255):                          # 12..21 是 label_parts 的输出值，源里不该有
        bad = raw.copy()
        bad[5, 5] = v
        with pytest.raises(AssertionError, match=str(v)):
            P.check_raw_ids(bad, "bad")


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_die_with_parent():
    """子进程设 PDEATHSIG 后，父进程直接 _exit（模拟主进程被 SIGKILL）⇒ 子进程也退出，不变成孤儿。"""
    import signal
    import time
    r, w = os.pipe()
    mid = os.fork()
    if mid == 0:                                           # 中间进程（= 主进程）
        os.close(r)
        gc = os.fork()
        if gc == 0:                                        # 孙进程（= rank）
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            P.die_with_parent(os.getppid())
            os.write(w, b"ready")
            time.sleep(30)
            os._exit(0)
        os.write(w, f"{gc:010d}".encode())
        time.sleep(0.5)                                    # 等孙进程设好 prctl
        os._exit(0)
    os.close(w)
    gpid = int(os.read(r, 10).decode())
    os.waitpid(mid, 0)

    def alive(pid):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().split(")")[-1].split()[0] != "Z"
        except FileNotFoundError:
            return False
    t = time.time()
    while alive(gpid) and time.time() - t < 5:
        time.sleep(0.05)
    os.close(r)
    leaked = alive(gpid)
    if leaked:
        os.kill(gpid, signal.SIGKILL)
    assert not leaked, "父进程退出后子进程还活着（PDEATHSIG 没生效）"


def test_ground_z():
    caps = np.full((2, 2, 12, 7), np.nan, np.float32)
    caps[0, 0, 0] = [0, 0, 50, 0, 0, 100, 10]               # a.z 50, b.z 100, r 10 ⇒ 40
    caps[0, 0, 7] = [0, 0, 30, 0, 0, 20, 5]                 # min(30,20) − 5 = 15
    caps[0, 1, 0] = [0, 0, -500, 0, 0, -500, 1]             # boss 不算
    gz = P.ground_z(caps)
    assert gz.dtype == np.float32 and gz[0] == 15 and np.isnan(gz[1])


def test_vae_input_lut_matches_cpu_path():
    u8 = np.arange(256, dtype=np.uint8).reshape(1, 1, 256, 1).repeat(3, -1)
    ref = torch.from_numpy(u8).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)    # infer.encode_video 的 CPU 写法
    got = P._U8_TO_PM1[torch.from_numpy(u8).long()].permute(3, 0, 1, 2)
    assert torch.equal(ref, got) and ref.stride() == got.stride()


# ---------------------------------------------------------------- 真实数据一个 clip（无 VAE）
RUN_DIR = os.path.join(P.SRC_DEFAULT, "v10_real", "01_Sunlit_Mediterranean_Courtyard", "R10_s1501")


@pytest.mark.skipif(not os.path.isdir(RUN_DIR), reason="v10 数据不在")
def test_real_clip_no_vae(tmp_path):
    import model as GM
    runs = [r for r in P.list_runs(P.SRC_DEFAULT, ["R10_s1501"])]
    t = [x for x in P.plan_tasks(runs) if x["start"] == 96][0]
    out = str(tmp_path)
    for d in ("cond", "cond_ext", "meta"):
        os.makedirs(os.path.join(out, d))
    r = P.prep_clip(t, out, vae=False)
    assert not r["skip"] and r["rgb"] is None and r["row"]["has_ext"]
    with np.load(P.clip_paths(out, t["clip_id"])["cond"]) as z:
        ids = z["ids"]
    with np.load(P.clip_paths(out, t["clip_id"])["cond_ext"]) as z:
        assert z["ids"].shape == (28, GEO.DST_H, GEO.DST_W) and z["blocks"].shape == (7, 5) and len(z["prompts"]) == 7
    with np.load(P.clip_paths(out, t["clip_id"])["meta"]) as z:
        assert z["capsules"].shape == (109, 2, 12, 7) and z["world_to_clip"].shape == (109, 16)
        assert (z["player_px"][:81] == (GM.collapse_parts(ids) == 10).sum((1, 2))).all()
    assert ids.shape == (81, GEO.DST_H, GEO.DST_W) and set(np.unique(ids).tolist()) <= GM.valid_raw_ids("parts") - {255}
    assert r["row"]["res"] == GEO.RES_TAG
    assert r["pt"]["prompts"] == TX.cell_prompts(r["pt"]["actions_block"][1:])
    # 第二次：输出都在 ⇒ 只补行，不解码
    r2 = P.prep_clip(t, out, vae=False)
    assert r2["skip"] and r2["row"] == r["row"]
