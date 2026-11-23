#!/usr/bin/env python3
"""GcDiT（gcdistill/gc_dit.py）的蒸馏门禁：tiny CPU 模型，全部秒级。

* test_1  ★ GcDiT(causal=False) ≡ model.gc_forward 逐位相同（两个词表 × fp32 / bf16）—— 双向教师 / critic 的回归锚点
* test_2  ★ 流式单 cell（KV 缓存 + cell_frames）≡ 并行块因果，float64 ≤ 1e-9；滑窗不驱逐时逐位等于绝对模式；
            sink/窗口驱逐之前逐位相同、之后真的不同
* test_3  ★ teacher forcing 泄漏：clean f → noisy f 恰好 0、clean f → noisy <f 恰好 0、clean f → noisy f+1 > 0
* test_4  条件因果性：改 cell j 的语义帧，cell < j 的输出逐位不变（并行因果 / TF / 流式三条路径），sem 项逐 cell 局部
* test_5  条件切片：cell_frames / window_frames / frankenstein_frames 与手写下标逐位相同（生产几何 N=21..27, K=20）
* test_6  GcRolloutDataset：cond + cond_ext 拼成 109 帧 / 28 个 cell，has_ext / drop 过滤，词表 collapse
* test_7  empty_wan_dit 的构造参数与 DiffSynth 的 Wan2.2-TI2V-5B 配置一致（真权重逐位相同由冒烟脚本核）
"""
import json
import os
import sys

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import geometry as GEO  # noqa: E402
import model as GM  # noqa: E402
from gcdistill import data_gc as DG  # noqa: E402
from gcdistill import gc_dit as GD  # noqa: E402
from distill.arope_dit import unwrap  # noqa: E402

B, C, F, LH, LW, L_TXT = 1, 48, 6, 8, 10, 8
H, W = 16 * LH, 16 * LW
TOL_EQUIV = 1e-9


def _inputs(vocab="parts", seed=7, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, F, LH, LW, generator=g).to(dtype)
    clean = torch.randn(B, C, F, LH, LW, generator=g).to(dtype)
    ctx = torch.randn(B, F, L_TXT, 4096, generator=g).to(dtype)
    ids = torch.arange(F, dtype=torch.int64).view(1, F)            # 每个 cell 一条不同的文本
    sem = DG.syn_sem_ids(DG.frames_of(F), H, W, vocab, g, block=8)[None]
    ts_pf = torch.linspace(50.0, 950.0, F, dtype=dtype).view(1, F).clone()
    ts_pf[:, 0] = 0.0
    return dict(x=x, clean=clean, ctx=ctx, ids=ids, sem=sem, ts=torch.full((B,), 700.0, dtype=dtype), ts_pf=ts_pf)


def _model(causal, vocab="parts", dtype=torch.float64, seed=3):
    return DG.tiny_gcdit(causal=causal, seed=seed, vocab=vocab, dtype=dtype).eval()


def _perturb_cell(sem, j, vocab):
    """把 cell j 的语义帧（cell 0 = 帧 0，cell j = 帧 4j−3 … 4j）换成另一套合法 id。"""
    s = sem.clone()
    lo, hi = (0, 1) if j == 0 else (4 * j - 3, 4 * j + 1)
    vals = sorted(GM.valid_raw_ids(vocab))
    s[:, lo:hi] = torch.tensor(vals[(vals.index(int(s[0, lo, 0, 0])) + 3) % len(vals)], dtype=torch.uint8)
    return s


# ----------------------------------------------------------------------------
# 1. 双向 GcDiT ≡ gc_forward
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("vocab", ["parts", "ids"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_1_bidirectional_gcdit_is_bitwise_gc_forward(vocab, dtype):
    d = _inputs(vocab, dtype=dtype)
    m = _model(False, vocab, dtype)
    assert GM.sem_vocab(m.dit) == vocab
    with torch.no_grad():
        a = m(d["x"], d["ts"], d["ctx"], context_ids=d["ids"], mask_input=d["sem"], first_frame_cond=True)
        b = GM.gc_forward(m.dit, d["x"], d["ts"], d["ctx"], d["sem"], first_frame_cond=True, context_ids=d["ids"])
        c = m(d["x"], d["ts"], d["ctx"], context_ids=d["ids"], mask_input=_perturb_cell(d["sem"], 2, vocab),
              first_frame_cond=True)
    assert torch.equal(a, b), f"GcDiT 双向路径与 gc_forward 不再逐位相同：max|Δ|={float((a - b).abs().max()):.3e}"
    assert not torch.equal(a, c), "换了语义图输出却没变 —— 条件根本没接上"


# ----------------------------------------------------------------------------
# 2. 流式 ≡ 并行块因果
# ----------------------------------------------------------------------------

def _stream(m, d, *, kv_window=None, sink_size=0, rope_cap=0, sem=None):
    sem = d["sem"] if sem is None else sem
    cache = unwrap(m).init_kv_cache(kv_window=kv_window, sink_size=sink_size, rope_cap=rope_cap)
    outs = []
    for k in range(F):
        with torch.no_grad():
            outs.append(m(d["x"][:, :, k:k + 1], d["ts_pf"][:, k], d["ctx"][:, k:k + 1],
                          context_ids=d["ids"][:, k:k + 1], first_frame_cond=False, frame_index=k,
                          kv_cache=cache, commit=True, mask_in_k=GD.cell_frames(sem, k)))
    unwrap(m).release_kv_cache(cache)
    return torch.cat(outs, dim=2)


@pytest.mark.parametrize("vocab", ["parts", "ids"])
def test_2_streaming_equals_parallel_causal_float64(vocab):
    d = _inputs(vocab)
    m = _model(True, vocab)
    with torch.no_grad():
        par = m(d["x"], d["ts_pf"], d["ctx"], context_ids=d["ids"], mask_input=d["sem"], first_frame_cond=True)
    st = _stream(m, d)
    per = [float((par[:, :, k] - st[:, :, k]).abs().max()) for k in range(F)]
    assert max(per) < TOL_EQUIV, f"流式与并行块因果不是同一个函数：per-cell max|Δ| {per}"
    # 滑窗但不驱逐（窗口 ≥ F−1）⇒ 与绝对模式逐位相同
    st_w = _stream(m, d, kv_window=F - 1, sink_size=0, rope_cap=F - 1)
    assert torch.equal(st, st_w)


def test_2b_sink_window_eviction_matches_before_and_diverges_after():
    """部署几何的缩小版：sink 1 / window 2 / cap 3。cell ≤ 3 读不到驱逐（与绝对模式逐位相同），cell 5 读到了（必须不同）。"""
    d = _inputs()
    m = _model(True)
    ab = _stream(m, d)
    ev = _stream(m, d, kv_window=2, sink_size=1, rope_cap=3)
    pre = max(float((ab[:, :, k] - ev[:, :, k]).abs().max()) for k in range(4))
    post = float((ab[:, :, 5] - ev[:, :, 5]).abs().max())
    assert pre < TOL_EQUIV, f"驱逐之前就不同了：{pre:.3e}"
    assert post > 1e-6, f"驱逐之后完全没影响（{post:.3e}）—— 窗口没在滑"


# ----------------------------------------------------------------------------
# 3. teacher forcing 泄漏
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def leak_matrix():
    d = _inputs()
    m = _model(True)
    kw = dict(context_ids=d["ids"], mask_input=d["sem"], first_frame_cond=True)
    with torch.no_grad():
        base = m(d["x"], d["ts_pf"], d["ctx"], clean_x=d["clean"], **kw)
    rows = []
    for f in range(F):
        c2 = d["clean"].clone()
        c2[:, :, f] += 5.0
        with torch.no_grad():
            vv = m(d["x"], d["ts_pf"], d["ctx"], clean_x=c2, **kw)
        rows.append((vv - base).abs().amax(dim=(0, 1, 3, 4)))
    return torch.stack(rows)                     # M[f_clean, f_noisy]


def test_3a_tf_no_self_copy(leak_matrix):
    assert max(float(leak_matrix[f, f]) for f in range(F)) == 0.0, "noisy cell f 看得见自己的 clean 孪生"


def test_3b_tf_history_reaches_next_cell(leak_matrix):
    assert min(float(leak_matrix[f, f + 1]) for f in range(F - 1)) > 1e-9, "clean 历史根本没进 noisy 流"


def test_3c_tf_no_future_leak(leak_matrix):
    assert max((float(leak_matrix[f, g]) for f in range(F) for g in range(f)), default=0.0) == 0.0


# ----------------------------------------------------------------------------
# 4. 条件因果性
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("j", [1, 3])
def test_4_condition_causality(j):
    d = _inputs()
    m = _model(True)
    s2 = _perturb_cell(d["sem"], j, "parts")
    kw = dict(context_ids=d["ids"], first_frame_cond=True)
    with torch.no_grad():
        a = m(d["x"], d["ts_pf"], d["ctx"], mask_input=d["sem"], **kw)
        b = m(d["x"], d["ts_pf"], d["ctx"], mask_input=s2, **kw)
        a_tf = m(d["x"], d["ts_pf"], d["ctx"], mask_input=d["sem"], clean_x=d["clean"], **kw)
        b_tf = m(d["x"], d["ts_pf"], d["ctx"], mask_input=s2, clean_x=d["clean"], **kw)
    for name, p, q in (("并行因果", a, b), ("TF", a_tf, b_tf)):
        before = float((p[:, :, :j] - q[:, :, :j]).abs().max())
        at = float((p[:, :, j] - q[:, :, j]).abs().max())
        assert before == 0.0, f"{name}: 改 cell {j} 的语义帧动了更早的 cell（{before:.3e}）"
        assert at > 0.0, f"{name}: 改 cell {j} 的语义帧，cell {j} 自己没变"
    st_a, st_b = _stream(m, d), _stream(m, d, sem=s2)
    assert float((st_a[:, :, :j] - st_b[:, :, :j]).abs().max()) == 0.0
    # sem 项本身逐 cell 局部（时间方向的核全是 1）
    ref = torch.zeros(B, m.dit.dim, F, LH // 2, LW // 2, dtype=torch.float64)
    with torch.no_grad():
        e1, e2 = GD.sem_embed(m.dit, d["sem"], ref), GD.sem_embed(m.dit, s2, ref)
    diff = (e1 - e2).abs().amax(dim=(0, 1, 3, 4))
    assert [k for k in range(F) if float(diff[k]) > 0] == [j]


# ----------------------------------------------------------------------------
# 5. 条件切片
# ----------------------------------------------------------------------------

def _coded(T):
    """帧 t 的像素值 = t（uint8），切片后直接读出帧号。"""
    return torch.arange(T, dtype=torch.uint8).view(1, T, 1, 1).expand(1, T, 4, 4).contiguous()


def _frames(x):
    return [int(v) for v in x[0, :, 0, 0]]


@pytest.mark.parametrize("N", [21, 22, 24, 27])
def test_5_slicing_matches_manual_indices_at_production_geometry(N):
    total, K = 1 + N, 20
    sem = _coded(DG.frames_of(total))
    for k in (0, 1, 7, N):
        exp = [0] if k == 0 else list(range(4 * k - 3, 4 * k + 1))
        assert _frames(GD.cell_frames(sem, k)) == exp
    # frankenstein：cat([cell 0, cells a..N])，a = total − K
    a, t2f = DG.tail_slices(total, K, True)
    assert a == total - K and t2f == [0] + list(range(a, total))
    fr = DG.frankenstein_frames(sem, a, total)
    assert _frames(fr) == [0] + list(range(4 * a - 3, 4 * N + 1))
    assert fr.shape[1] == DG.frames_of(1 + K) == 81
    # 尾巴里的 cell j ≥ 1 拿到的恰好是绝对 cell t2f[j] 自己的 4 帧（_group 的 T=4f−3 分组）
    for j in range(1, 1 + K):
        assert _frames(fr[:, 4 * j - 3: 4 * j + 1]) == _frames(GD.cell_frames(sem, t2f[j]))
    # 连续尾巴：锚点是生成的 cell a（用它的最后一帧 4a），之后逐 cell 对齐
    a_c, t2f_c = DG.tail_slices(total, K, False)
    assert a_c == total - 1 - K and t2f_c == list(range(a_c, total))
    wf = GD.window_frames(sem, a_c, 1 + K)
    assert _frames(wf) == list(range(4 * a_c, 4 * N + 1))
    for j in range(1, 1 + K):
        assert _frames(wf[:, 4 * j - 3: 4 * j + 1]) == _frames(GD.cell_frames(sem, t2f_c[j]))
    # 文本 / id 与 latent 同一套 cat
    ids = torch.arange(total).view(1, total)
    assert DG.frankenstein_cells(ids, a).tolist()[0] == t2f
    lat = torch.arange(total, dtype=torch.float32).view(1, 1, total, 1, 1)
    assert DG.frankenstein_cells(lat, a, dim=2).flatten().tolist() == [float(i) for i in t2f]


def test_5b_slice_rollout_and_alignment_gate():
    n = 9
    batch = {"sem_ids": _coded(DG.frames_of(n)), "context": torch.zeros(1, n, 2, 3), "context_ids": torch.arange(n)[None]}
    sem, ctx, ids = DG.slice_rollout(batch, 6)
    assert sem.shape[1] == 25 and ctx.shape[1] == 7 and ids.shape[1] == 7
    with pytest.raises(ValueError):
        DG.slice_rollout(batch, 9)
    with pytest.raises(AssertionError):
        DG.assert_rollout_aligned_gc(sem[:, :-1], ctx, ids, 6)


# ----------------------------------------------------------------------------
# 6. GcRolloutDataset（假的 v10 布局，全尺寸形状；压缩后很小）
# ----------------------------------------------------------------------------

def _fake_root(tmp_path, table):
    root = tmp_path / "gc25d_fake"
    for sub in ("index", "latents", "cond", "cond_ext"):
        (root / sub).mkdir(parents=True)
    prompts = [k for k in sorted(table) if k != "In a 2.5D isometric view."]
    rows = []
    for i, (has_ext, drop, split) in enumerate([(True, False, "train"), (False, False, "train"),
                                               (True, True, "train"), (True, False, "val")]):
        cid = f"scene__run__{i:06d}"
        rows.append({"clip_id": cid, "split": split, "drop": drop, "has_ext": has_ext})
        torch.save({"latent": torch.full((48, 21, GEO.LAT_H, GEO.LAT_W), float(i), dtype=torch.bfloat16),
                    "prompts": [prompts[j % len(prompts)] for j in range(21)]}, root / "latents" / f"{cid}.pt")
        ids = np.zeros((81, GEO.DST_H, GEO.DST_W), np.uint8)
        ids[:, :, 400:] = 12                                       # 玩家左臂（parts）
        np.savez_compressed(root / "cond" / f"{cid}.npz", ids=ids)
        if has_ext:
            ext = np.full((28, GEO.DST_H, GEO.DST_W), 5, np.uint8)
            np.savez_compressed(root / "cond_ext" / f"{cid}.npz", ids=ext, blocks=np.zeros((7, 5), np.uint8),
                                prompts=np.array([prompts[(21 + j) % len(prompts)] for j in range(7)]))
    with open(root / "index" / "clips.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(root / "index" / "prep_args.json", "w") as fh:
        json.dump({"res": GEO.RES_TAG}, fh)                     # 数据根的分辨率（GcRolloutDataset 查它）
    return str(root), prompts


def test_6_rollout_dataset(tmp_path):
    table = {k: torch.full((512, 4096), float(i), dtype=torch.bfloat16)
             for i, k in enumerate(sorted(DG.TXT.all_prompts() + [DG.TXT.NULL_PROMPT]))}
    root, prompts = _fake_root(tmp_path, table)
    ds = DG.GcRolloutDataset(root, "train", table, vocab="parts")
    assert ds.clip_ids() == ["scene__run__000000"], "只要 has_ext 且未 drop 的 train clip"
    s = ds[0]
    assert tuple(s["z0"].shape) == (48, 1, GEO.LAT_H, GEO.LAT_W) and s["z0"].dtype == torch.bfloat16
    assert tuple(s["sem_ids"].shape) == (109, GEO.DST_H, GEO.DST_W) and s["sem_ids"].dtype == torch.uint8
    assert int(s["sem_ids"][80, 0, 500]) == 12 and int(s["sem_ids"][81, 0, 0]) == 5     # cond | cond_ext 的接缝
    assert tuple(s["context"].shape) == (28, 512, 4096) and tuple(s["context_ids"].shape) == (28,)
    kid = DG.key_to_id(table)
    exp = [prompts[j % len(prompts)] for j in range(21)] + [prompts[(21 + j) % len(prompts)] for j in range(7)]
    assert s["context_ids"].tolist() == [kid[p] for p in exp]
    assert float(s["context"][27, 0, 0]) == float(table[exp[27]][0, 0])
    ds_ids = DG.GcRolloutDataset(root, "train", table, vocab="ids")
    assert int(ds_ids[0]["sem_ids"][0, 0, 500]) == 10, "ids 词表要把 parts 的 12 collapse 成 10"
    b = DG.rollout_collate([s, s])
    assert tuple(b["sem_ids"].shape) == (2, 109, GEO.DST_H, GEO.DST_W) and b["clip_id"] == [s["clip_id"]] * 2


# ----------------------------------------------------------------------------
# 7. 骨架构造参数
# ----------------------------------------------------------------------------

def test_7_empty_wan_dit_config_matches_diffsynth():
    """DiffSynth 按 safetensors 的 hash 选构造参数；Wan2.2-TI2V-5B DiT 那一条是 1f5ab770…"""
    from diffsynth.configs.model_configs import wan_series
    hit = [c for c in wan_series if c["model_hash"] == "1f5ab7703c6fc803fdded85ff040c316"]
    assert len(hit) == 1 and hit[0]["model_class"].endswith("WanModel")
    assert hit[0]["extra_kwargs"] == GD.WAN22_5B_DIT


# ----------------------------------------------------------------------------
# 8. 真权重逐位一致（默认 skip；动过 gc_dit / arope_dit / model 之后在一张卡上跑）
#    GC_BITPARITY_REAL=1 GC_BITPARITY_CKPT=<gc ckpt> CUDA_VISIBLE_DEVICES=6 pytest code/gc25d/tests/test_gcdit.py -k real
# ----------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("GC_BITPARITY_REAL") != "1", reason="真权重闸：GC_BITPARITY_REAL=1 + GC_BITPARITY_CKPT")
def test_8_real_weights_bitparity():
    ckpt = os.environ["GC_BITPARITY_CKPT"]
    dev = "cuda"
    m = GD.build_gcdit(ckpt, causal=False, dit=GD.empty_wan_dit(), log=lambda *a: None).to(dev).eval()
    m.move_rope_table(dev)
    vocab = GM.sem_vocab(m.dit)
    g = torch.Generator().manual_seed(0)
    f = 3
    x = torch.randn(1, 48, f, GEO.LAT_H, GEO.LAT_W, generator=g).to(dev, torch.bfloat16)
    ctx = (torch.randn(1, f, 512, 4096, generator=g) * 0.1).to(dev, torch.bfloat16)
    ids = torch.arange(f).view(1, f).to(dev)
    sem = DG.syn_sem_ids(DG.frames_of(f), GEO.DST_H, GEO.DST_W, vocab, g)[None].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = m(x, torch.tensor([700.0], device=dev), ctx, context_ids=ids, mask_input=sem, first_frame_cond=True)
        b = GM.gc_forward(m.dit, x, torch.tensor([700.0], device=dev), ctx, sem, first_frame_cond=True, context_ids=ids)
    assert torch.equal(a, b), f"真权重下 GcDiT 与 gc_forward 不逐位相同：{float((a - b).abs().max()):.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
