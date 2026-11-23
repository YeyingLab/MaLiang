#!/usr/bin/env python3
"""stage3（dmd_gc.py，self-forcing DMD）的门禁 —— ARoPE tests/test_stage3_dmd.py 的 gc 移植。全部 CPU、tiny 模型。

* test_1  ★ 提交趟恒 t=0，且每个 frame_index 恰好一次（rollout 与 replay 两条路径都测）
* test_2  流式前向恒 first_frame_cond=False；条件 = 本 cell 的语义帧（cell 0 一帧、之后四帧）、没有 offset_win
* test_3  生产几何（K=H=20、N=21..27）下 grad 窗口落在 scored tail 内、不含 tail 的干净锚点（两种尾巴）
* test_4  score_tail 的切片 / 语义帧 / 文本 / grad_mask 对齐（生产几何，frankenstein 与 contiguous）
* test_5  梯度只穿过**一次**退出前向，其余全在 no_grad 下
* test_6  eval_idx 永不含 cell 0；critic ≡ teacher 时 DMD loss 恰好是 0
* test_7  ★ streamed 的逐帧 backward == fused 的一次性 backward
* test_8  ★ 部署几何 sink 3 / window 17 / cap 20 的整条 rollout（N=22，真的驱逐）：与 codeworld/live.py 同一个 _call
* test_9  argparse 的守卫与配方默认值（/dits/ 拒收、rope_cap 不变量、rollout 区间、frankenstein 默认）
* test_10 dry run 端到端：dits/ critic/ ema/ 都导出、metadata 带 tail_mode / 几何、EMA 是 fp32
"""
import os
import sys

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from gcdistill import data_gc as DG  # noqa: E402
from gcdistill import dmd_gc as S3  # noqa: E402
from distill import flow_pf, runlib  # noqa: E402

B = 1
_, _, LH, LW = DG.TINY_SHAPE
L_TXT = 8
KV, SINK, CAP = 3, 1, 4          # tiny 几何上的部署同构滑窗（sink>0；N=6 ⇒ 7 cell，会真的驱逐）
N_GEN = 6


def _deploy():
    return flow_pf.deploy_schedule(flow_pf.DEPLOY_DENOISE_LIST)


def _inputs(n_gen=N_GEN, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = 1 + n_gen
    z0 = torch.randn(B, 48, 1, LH, LW, generator=g)
    ctx = torch.randn(B, total, L_TXT, 4096, generator=g)
    ids = torch.arange(total, dtype=torch.int64).view(1, total)
    sem = DG.syn_sem_ids(DG.frames_of(total), 16 * LH, 16 * LW, "parts", g)[None]
    return z0, ctx, ids, sem


def _spy(model):
    calls = []
    orig = model.forward

    def spy(*a, **kw):
        x, t = a[0], a[1]
        calls.append(dict(frame_index=kw.get("frame_index"), commit=kw.get("commit"),
                          first_frame_cond=kw.get("first_frame_cond"), keys=set(kw),
                          mk=kw.get("mask_in_k"), t_max=float(torch.as_tensor(t).float().abs().max()),
                          grad_enabled=torch.is_grad_enabled()))
        return orig(*a, **kw)

    model.forward = spy
    return calls


def _model(seed=1234, causal=True):
    return DG.tiny_gcdit(causal=causal, seed=seed, dtype=torch.float32).eval()


def _roll(m, n_gen=N_GEN, **kw):
    z0, ctx, ids, sem = _inputs(n_gen)
    base = dict(kv_window=KV, rope_cap=CAP, sink_size=SINK, score_last_k=4, gen=torch.Generator().manual_seed(5))
    base.update(kw)
    return S3.self_forcing_rollout(m, z0, ctx, ids, sem, n_gen, _deploy(), base.pop("grad_horizon", 4), **base)


# ----------------------------------------------------------------------------
# 1 / 2. 流式前向的两条铁律
# ----------------------------------------------------------------------------

def test_1_commit_pass_is_always_t0_and_exactly_once_per_cell():
    m = _model()
    calls = _spy(m)
    _roll(m)
    commits = [c for c in calls if c["commit"]]
    for c in commits:
        assert c["t_max"] == 0.0, f"提交趟的 t 必须恒为 0，cell {c['frame_index']} 用了 t={c['t_max']}"
        assert not c["grad_enabled"], f"cell {c['frame_index']} 的提交趟没有关梯度"
    assert [c["frame_index"] for c in commits] == list(range(1 + N_GEN))


def test_1b_replay_commits_are_also_t0():
    m = _model()
    z0, ctx, ids, sem = _inputs()
    rec = {"exit": {}, "commit": {}}
    S3.self_forcing_rollout(m, z0, ctx, ids, sem, N_GEN, _deploy(), 4, kv_window=KV, rope_cap=CAP,
                            sink_size=SINK, score_last_k=4, gen=torch.Generator().manual_seed(5), record=rec)
    rec["n_eval"] = max(1, len(rec["exit"]))
    calls = _spy(m)
    S3.stream_grad_replay(m, z0, ctx, ids, sem, rec, _deploy(), frame_target=lambda k: None,
                          backward=lambda l: None, kv_window=KV, rope_cap=CAP, sink_size=SINK)
    commits = [c for c in calls if c["commit"]]
    assert [c["frame_index"] for c in commits] == list(range(1 + N_GEN))
    assert all(c["t_max"] == 0.0 for c in commits)


def test_2_streaming_forwards_never_enable_first_frame_cond_and_take_their_own_cell_frames():
    m = _model()
    calls = _spy(m)
    _roll(m)
    _, _, _, sem = _inputs()
    for c in calls:
        assert c["first_frame_cond"] is False
        assert "offset_win" not in c["keys"] and "offset_px" not in c["keys"] and "mask_input" not in c["keys"]
        k = c["frame_index"]
        want = sem[:, :1] if k == 0 else sem[:, 4 * k - 3: 4 * k + 1]
        assert torch.equal(c["mk"], want), f"cell {k} 的语义帧不是它自己的"


# ----------------------------------------------------------------------------
# 3 / 4. 生产几何下的下标算术
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("n_gen", [21, 22, 26, 27])
@pytest.mark.parametrize("tail_mode", ["frankenstein", "contiguous"])
def test_3_grad_window_sits_inside_the_scored_tail_at_production_geometry(n_gen, tail_mode):
    K = HOR = 20
    total = 1 + n_gen
    lo, hi = S3.grad_start_range(total, HOR, K)
    assert lo == hi == total - K
    a, t2f = DG.tail_slices(total, K, tail_mode == "frankenstein")
    grad_end = min(lo + HOR, total)
    assert grad_end == total and grad_end - lo == HOR
    scored = set(t2f[1:])                      # tail 的 cell 0 是不打分的干净锚点
    assert set(range(lo, grad_end)) == scored, "grad 窗口必须恰好是打分尾巴（去掉锚点）"
    assert t2f[0] not in range(lo, grad_end)


def test_3b_draw_grad_start_stays_in_range():
    for n_gen in (21, 24, 27):
        total = 1 + n_gen
        lo, hi = S3.grad_start_range(total, 20, 20)
        for seed in range(4):
            assert lo <= S3.draw_grad_start(total, 20, 20, gen=torch.Generator().manual_seed(seed)) <= hi


@pytest.mark.parametrize("n_gen", [21, 27])
@pytest.mark.parametrize("tail_mode", ["frankenstein", "contiguous"])
def test_4_score_tail_is_aligned_at_production_geometry(n_gen, tail_mode):
    K = HOR = 20
    total = 1 + n_gen
    gen_lat = torch.randn(B, 48, total, 2, 2)
    lo, _ = S3.grad_start_range(total, HOR, K)
    grad_mask = torch.tensor([lo <= k < min(lo + HOR, total) for k in range(total)])
    ctx = torch.randn(B, total, 2, 8)
    ids = torch.arange(total).view(1, total)
    sem = torch.arange(DG.frames_of(total), dtype=torch.uint8).view(1, -1, 1, 1).expand(1, -1, 4, 4).contiguous()
    tail, cond_t, gm_t, t2f = S3.score_tail(gen_lat, grad_mask, sem, ctx, ids, n_gen, K, tail_mode=tail_mode)
    assert tail.shape[2] == 1 + K == len(t2f) == gm_t.numel() == cond_t["context"].shape[1]
    assert cond_t["mask_input"].shape[1] == DG.frames_of(1 + K) == 81, "教师 / critic 看到的是一条 21-cell 的片子"
    for j, c in enumerate(t2f):
        assert torch.equal(tail[:, :, j], gen_lat[:, :, c]) and int(cond_t["context_ids"][0, j]) == c
        assert torch.equal(cond_t["context"][:, j], ctx[:, c])
    fr = [int(v) for v in cond_t["mask_input"][0, :, 0, 0]]
    for j in range(1, 1 + K):
        c = t2f[j]
        assert fr[4 * j - 3: 4 * j + 1] == list(range(4 * c - 3, 4 * c + 1)), f"tail cell {j} 的语义帧不是 cell {c} 的"
    if tail_mode == "frankenstein":
        assert t2f[0] == 0 and fr[0] == 0 and torch.equal(tail[:, :, 0], gen_lat[:, :, 0]), "锚点必须是真实 cell 0"
    else:
        assert t2f[0] == total - 1 - K and fr[0] == 4 * t2f[0], "连续尾巴的锚点用 cell a 的最后一帧"
    assert not bool(gm_t[0]) and bool(gm_t[1:].all())
    assert cond_t["first_frame_cond"] is True and "offset_px" not in cond_t


# ----------------------------------------------------------------------------
# 5. 梯度只穿过一次退出前向
# ----------------------------------------------------------------------------

def test_5_gradient_crosses_exactly_one_forward_per_grad_frame():
    m = _model()
    calls = _spy(m)
    with torch.enable_grad():
        gen_lat, grad_mask = _roll(m, grad_horizon=3, grad_start=4, exit_step=1)
    on = [c for c in calls if c["grad_enabled"]]
    assert len(on) == 3 and all(not c["commit"] for c in on)
    assert sorted(c["frame_index"] for c in on) == [4, 5, 6]
    assert int(grad_mask.sum()) == 3 and bool(grad_mask[4]) and not bool(grad_mask[3])
    assert gen_lat.requires_grad


# ----------------------------------------------------------------------------
# 6. eval_idx 与 step-0 的退化形态
# ----------------------------------------------------------------------------

def _tail(tail_mode="frankenstein"):
    K = 4
    total = 1 + N_GEN
    gen_lat = torch.randn(B, 48, total, LH, LW)
    lo, _ = S3.grad_start_range(total, K, K)
    grad_mask = torch.tensor([lo <= k < min(lo + K, total) for k in range(total)])
    _, ctx, ids, sem = _inputs()
    return S3.score_tail(gen_lat, grad_mask, sem, ctx, ids, N_GEN, K, tail_mode=tail_mode)


def test_6_eval_idx_excludes_the_anchor_and_dmd_is_zero_when_critic_equals_teacher():
    tail, cond_t, gm_t, _ = _tail()
    same = _model(seed=7, causal=False)
    loss, log = S3.dmd_generator_loss(same, same, tail, cond_t, gm_t, gen=torch.Generator().manual_seed(11))
    assert float(loss) == 0.0 and log["dmd_gnorm"] == 0.0
    tgt, eval_idx, _ = S3.dmd_targets(same, same, tail, cond_t, gm_t, gen=torch.Generator().manual_seed(11))
    assert int(eval_idx.min()) >= 1 and torch.allclose(tgt, tail)
    other = _model(seed=99, causal=False)
    loss2, log2 = S3.dmd_generator_loss(same, other, tail, cond_t, gm_t, gen=torch.Generator().manual_seed(11))
    assert float(loss2) > 0.0 and log2["dmd_gnorm"] > 0.0


def test_6b_eval_idx_drops_cell0_even_when_the_mask_asks_for_it():
    tail, cond_t, _, _ = _tail()
    same = _model(seed=7, causal=False)
    _, eval_idx, _ = S3.dmd_targets(same, same, tail, cond_t, torch.ones(tail.shape[2], dtype=torch.bool),
                                    gen=torch.Generator().manual_seed(11))
    assert eval_idx.tolist() == list(range(1, tail.shape[2]))
    loss_c, _ = S3.critic_loss(same, tail, cond_t, eval_idx, gen=torch.Generator().manual_seed(11),
                               use_gradient_checkpointing=False)
    assert torch.isfinite(loss_c)


def test_6c_native_scoring_gate_rejects_wrong_cond():
    tail, cond_t, _, _ = _tail()
    ts = torch.zeros(B, tail.shape[2])
    S3._assert_native_scoring(tail, ts, cond_t)
    for bad in (dict(cond_t, mask_input=cond_t["mask_input"][:, :-1]),
                dict(cond_t, mask_input=cond_t["mask_input"].float()),
                dict(cond_t, first_frame_cond=False),
                dict(cond_t, clean_x=tail)):
        with pytest.raises(AssertionError):
            S3._assert_native_scoring(tail, ts, bad)


# ----------------------------------------------------------------------------
# 7. streamed == fused
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("tail_mode", ["frankenstein", "contiguous"])
def test_7_streamed_per_frame_backward_equals_the_fused_backward(tail_mode):
    z0, ctx, ids, sem = _inputs()
    dep = _deploy()
    hor, gs, ex = 3, 4, 1
    kw = dict(kv_window=KV, rope_cap=CAP, sink_size=SINK, score_last_k=4, grad_start=gs, exit_step=ex)
    m_f = _model()
    torch.manual_seed(21)
    gen_f, mask_f = S3.self_forcing_rollout(m_f, z0, ctx, ids, sem, N_GEN, dep, hor,
                                            gen=torch.Generator().manual_seed(5), **kw)
    idx = torch.nonzero(mask_f).flatten()
    tgt = torch.randn(B, 48, 1 + N_GEN, LH, LW, generator=torch.Generator().manual_seed(33))
    loss_f = 0.5 * torch.nn.functional.mse_loss(gen_f[:, :, idx].float(), tgt[:, :, idx].float())
    loss_f.backward()
    g_f = {n: p.grad.detach().clone() for n, p in m_f.named_parameters() if p.grad is not None}

    m_s = _model()
    rec = {"exit": {}, "commit": {}}
    torch.manual_seed(21)
    gen_s, mask_s = S3.self_forcing_rollout(m_s, z0, ctx, ids, sem, N_GEN, dep, hor, record=rec,
                                            gen=torch.Generator().manual_seed(5), **kw)
    assert torch.equal(gen_s, gen_f.detach()) and torch.equal(mask_s, mask_f)
    rec["n_eval"] = int(idx.numel())
    tot = S3.stream_grad_replay(m_s, z0, ctx, ids, sem, rec, dep, frame_target=lambda k: tgt[:, :, k:k + 1],
                                backward=lambda l: l.backward(), kv_window=KV, rope_cap=CAP, sink_size=SINK)
    g_s = {n: p.grad.detach().clone() for n, p in m_s.named_parameters() if p.grad is not None}
    assert abs(tot - float(loss_f.detach())) < 1e-6
    assert set(g_s) == set(g_f) and g_f
    assert any(n.startswith("dit.gc_sem_cond.") for n in g_f), "语义条件模块没拿到梯度"
    num = sum(float(((g_s[n] - g_f[n]) ** 2).sum()) for n in g_f)
    den = sum(float((g_f[n] ** 2).sum()) for n in g_f)
    assert den > 0 and (num / den) ** 0.5 < 1e-6


# ----------------------------------------------------------------------------
# 8. 部署几何的整条 rollout
# ----------------------------------------------------------------------------

def test_8_deploy_geometry_rollout_sink3_window17_cap20():
    """N=22 ⇒ 23 个 cell：cell 20 的提交第一次驱逐，cell 21 / 22 读到驱逐后的缓存。调用口径必须与
    codeworld/live.py::LiveGame._call 完全相同（逐 cell 文本 + mask_in_k，没有 offset / mask_input）。"""
    n_gen = 22
    m = _model()
    calls = _spy(m)
    z0, ctx, ids, sem = _inputs(n_gen)
    caches = []
    orig_init = m.init_kv_cache

    def _init(**kw):
        c = orig_init(**kw)
        caches.append((c, kw))
        return c

    m.init_kv_cache = _init
    gen_lat, _ = S3.self_forcing_rollout(m, z0, ctx, ids, sem, n_gen, _deploy(), 20, kv_window=17, rope_cap=20,
                                         sink_size=3, score_last_k=20, gen=torch.Generator().manual_seed(1))
    assert torch.isfinite(gen_lat).all() and gen_lat.shape[2] == 1 + n_gen
    assert len(caches) == 1 and caches[0][1] == dict(kv_window=17, sink_size=3, rope_cap=20)
    live_keys = {"context_ids", "first_frame_cond", "frame_index", "kv_cache", "commit", "mask_in_k"}
    assert all(c["keys"] == live_keys for c in calls), {frozenset(c["keys"]) for c in calls}
    commits = [c["frame_index"] for c in calls if c["commit"]]
    assert commits == list(range(1 + n_gen))
    assert len([c for c in calls if not c["commit"]]) <= 4 * n_gen


# ----------------------------------------------------------------------------
# 9. argparse
# ----------------------------------------------------------------------------

def test_9_parse_args_guards_and_defaults(tmp_path):
    a = S3.parse_args(["--dry_run", "--output", str(tmp_path / "x")])
    assert (a.kv_window, a.sink_size, a.rope_cap) == (17, 3, 20)
    assert (a.rollout_min, a.rollout_max, a.dmd_score_frames, a.grad_horizon) == (21, 27, 20, 20)
    assert a.tail_mode == "frankenstein" and a.grad_pass == "streamed" and a.real_guidance_scale == 1.0
    assert (a.lr, a.lr_critic, a.betas, a.weight_decay, a.warmup, a.grad_clip) == (2e-6, 1e-6, (0.0, 0.999), 0.01, 100, 10.0)
    assert (a.critic_steps, a.ema_decay, a.ema_start, a.max_steps, a.save_every) == (5, 0.99, 100, 500, 50)
    ck = str(tmp_path / "dits" / "dit_step10.safetensors")
    with pytest.raises(SystemExit, match="dits"):
        S3.parse_args(["--output", str(tmp_path / "x"), "--student_ckpt", ck, "--teacher_ckpt", ck])
    with pytest.raises(SystemExit, match="rope_cap"):
        S3.parse_args(["--dry_run", "--output", str(tmp_path / "x"), "--rope_cap", "21"])
    with pytest.raises(SystemExit, match="rollout_min"):
        S3.parse_args(["--output", str(tmp_path / "x"), "--rollout_min", "20"])
    with pytest.raises(SystemExit, match="rollout_max"):
        S3.parse_args(["--output", str(tmp_path / "x"), "--rollout_max", "28"])


# ----------------------------------------------------------------------------
# 10. dry run 端到端
# ----------------------------------------------------------------------------

def test_10_dry_run_end_to_end(tmp_path):
    out = str(tmp_path / "s3")
    assert S3.main(["--dry_run", "--output", out, "--num_workers", "0"]) == 6
    for p in (runlib.dit_path(out, 6), runlib.critic_path(out, 6), runlib.ema_path(out, 6)):
        assert os.path.isfile(p), p
    _, meta = runlib.read_ckpt_header(runlib.ema_path(out, 6))
    for k, v in {"stage": "stage3_dmd", "role": "ema(deploy)", "tail_mode": "frankenstein", "sink_size": "1",
                 "kv_window": "3", "rope_cap": "4", "format": "gc25d", "vocab": "parts"}.items():
        assert meta.get(k) == v, (k, meta.get(k))
    hdr, _ = runlib.read_ckpt_header(runlib.ema_path(out, 6))
    assert {v["dtype"] for v in hdr.values()} == {"BF16"}, "导出一律 bf16（EMA 影子本身是 fp32）"
    rows = [ln for ln in open(os.path.join(out, "train_log.csv")).read().splitlines()[1:] if ln]
    assert len(rows) == 6


# ----------------------------------------------------------------------------
# 11. 部署孪生（eval_rollout.deploy_rollout = codeworld/live.py 的逐 cell 回路）
# ----------------------------------------------------------------------------

def test_11_eval_rollout_is_the_live_loop():
    from gcdistill import eval_rollout as ER
    m = DG.tiny_gcdit(causal=True, seed=3, dtype=torch.bfloat16).eval()
    calls = _spy(m)
    n = 5
    z0, ctx, ids, sem = _inputs(n)
    lat = ER.deploy_rollout(m, z0, ctx.bfloat16(), ids, sem, n, kv_window=KV, sink_size=SINK, rope_cap=CAP, seed=0)
    assert tuple(lat.shape) == (B, 48, 1 + n, LH, LW) and torch.isfinite(lat.float()).all()
    assert torch.equal(lat[:, :, 0].float(), z0[:, :, 0].bfloat16().float()), "cell 0 就是 z0"
    live_keys = {"context_ids", "first_frame_cond", "frame_index", "kv_cache", "commit", "mask_in_k"}
    assert all(c["keys"] == live_keys and c["first_frame_cond"] is False for c in calls)
    assert [c["frame_index"] for c in calls if c["commit"]] == list(range(1 + n))
    assert all(c["t_max"] == 0.0 for c in calls if c["commit"])
    _, _, t_grid = _deploy()
    for k in range(1, 1 + n):
        ts = [c["t_max"] for c in calls if c["frame_index"] == k and not c["commit"]]
        assert len(ts) == 4 and all(abs(a - float(b)) < 1e-3 for a, b in zip(ts, t_grid)), (k, ts)
    lat2 = ER.deploy_rollout(m, z0, ctx.bfloat16(), ids, sem, n, kv_window=KV, sink_size=SINK, rope_cap=CAP, seed=0)
    lat3 = ER.deploy_rollout(m, z0, ctx.bfloat16(), ids, sem, n, kv_window=KV, sink_size=SINK, rope_cap=CAP, seed=1)
    assert torch.equal(lat, lat2) and not torch.equal(lat, lat3), "同种子逐位复现、换种子要变"
    with pytest.raises(SystemExit):
        ER.refuse_dits("/tmp/maliang-test/x/stage2_cd/dits/dit_step500.safetensors")
    ER.refuse_dits("/tmp/maliang-test/x/stage2_cd/ema/ema_step500.safetensors")
    with pytest.raises(SystemExit):
        ER.parse_args(["--ckpt", "/x/dits/dit_step1.safetensors", "--out", "/tmp/maliang-test/x"])


def test_12_rollout_metrics_shapes_and_drift():
    from gcdistill import eval_rollout as ER
    g = np.full((13, 8, 16, 3), 100, np.uint8)
    g[9:] = 110                                                      # 最后一个 cell 亮了 10
    gt = np.full((9, 8, 16, 3), 100, np.uint8)
    met = ER.rollout_metrics(g, gt)
    assert met["frames"] == 13 and met["gt_frames"] == 9 and len(met["luma_gen_cell"]) == 4
    assert abs(met["drift"] - 10.0) < 1e-3 and abs(met["gt_luma_gap"]) < 1e-3
    assert met["psnr_cell"][0] is None, "帧 0 的 PSNR 是 inf，json 里记 None"
    json.dumps(met)


# ----------------------------------------------------------------------------
# 13 / 14 / 15 / 16. 训练 rollout 的数值 == 部署；重放逐位；续训的 lr 调度
# ----------------------------------------------------------------------------

class _Fp32Out(torch.nn.Module):
    """accelerate.prepare 给 forward 套的 convert_outputs_to_fp32 的 CPU 替身：FSDP 下学生的输出就是这样变成 fp32 的。"""

    def __init__(self, m):
        super().__init__()
        self.module = m                                   # unwrap() 走 .module 拿裸模型

    def forward(self, *a, **kw):
        return self.module(*a, **kw).float()


def _t_spy(monkeypatch):
    """记录每次单 cell 前向在模型内部用的 (latent dtype, t)（arope_dit._per_token_timestep_pf 之后的值）。"""
    import distill.arope_dit as AD
    seen, orig = [], AD._per_token_timestep_pf

    def hook(latents, timestep, n_tok, first_frame_cond):
        out = orig(latents, timestep, n_tok, first_frame_cond)
        if latents.shape[2] == 1:
            seen.append((latents.dtype, float(out.float().reshape(-1)[0])))
        return out

    monkeypatch.setattr(AD, "_per_token_timestep_pf", hook)
    return seen


def test_13_rollout_timesteps_and_dtype_equal_the_deploy_loop(monkeypatch):
    """★ 学生输出被抬成 fp32（accelerate 包装）时，rollout 仍是 bf16 latent、t = 1000/752/500/252 —— 与
    eval_rollout.deploy_rollout（= live.py）逐 cell 相同。负对照：去掉 _as_latent ⇒ t 变回 fp32 的 750/500.6/251.6。"""
    from gcdistill import eval_rollout as ER
    n = 3
    z0, ctx, ids, sem = _inputs(n)
    z0, ctx = z0.bfloat16(), ctx.bfloat16()
    m = DG.tiny_gcdit(causal=True, seed=3, dtype=torch.bfloat16).eval()
    seen = _t_spy(monkeypatch)
    kw = dict(kv_window=KV, rope_cap=CAP, sink_size=SINK, score_last_k=n, grad_start=1, exit_step=3,
              gen=torch.Generator().manual_seed(5))

    def roll(record):
        seen.clear()
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):     # = dmd_gc 的 ac_roll
            gen_lat, _ = S3.self_forcing_rollout(_Fp32Out(m), z0, ctx, ids, sem, n, _deploy(), n, record=record, **kw)
        return gen_lat, list(seen)

    seen.clear()
    ER.deploy_rollout(m, z0, ctx, ids, sem, n, kv_window=KV, sink_size=SINK, rope_cap=CAP, seed=0)
    dep = list(seen)
    want = [(torch.bfloat16, t) for t in (1000.0, 752.0, 500.0, 252.0, 0.0)]
    assert dep == [(torch.bfloat16, 0.0)] + want * n, dep
    for record in (None, {"exit": {}, "commit": {}}):                  # fused 与 streamed 两条
        gen_lat, tr = roll(record)
        assert gen_lat.dtype == torch.bfloat16 and tr == dep, tr
    monkeypatch.setattr(S3, "_as_latent", lambda v, x: v)                  # 负对照：上一版的数值
    with pytest.raises(AssertionError, match="dtype"):
        roll(None)
    bad = [s for s in seen if s[0] == torch.float32]
    assert bad and abs(bad[0][1] - 750.0) < 1e-3, seen[:6]


def test_14_streamed_replay_is_bitwise_when_target_is_the_rollout():
    """目标 = rollout 自己的输出（critic ≡ teacher 的 step 0）⇒ 重放的 loss、replay_dev、学生梯度都恰好 0；
    record 趟里 grad 帧的退出前向在 grad 模式下跑（与重放同一个 kernel）、但不留图。"""
    m = _model()
    calls = _spy(m)
    z0, ctx, ids, sem = _inputs()
    rec = {"exit": {}, "commit": {}}
    gen_lat, mask = S3.self_forcing_rollout(m, z0, ctx, ids, sem, N_GEN, _deploy(), 3, kv_window=KV, rope_cap=CAP,
                                            sink_size=SINK, score_last_k=4, grad_start=4, exit_step=2, record=rec,
                                            gen=torch.Generator().manual_seed(5))
    assert not gen_lat.requires_grad
    on = [c for c in calls if c["grad_enabled"]]
    assert sorted(c["frame_index"] for c in on) == [4, 5, 6] and not any(c["commit"] for c in on)
    assert sorted(rec["den"]) == [4, 5, 6]
    rec["n_eval"] = 3
    m.zero_grad(set_to_none=True)
    loss = S3.stream_grad_replay(m, z0, ctx, ids, sem, rec, _deploy(), lambda k: gen_lat[:, :, k:k + 1],
                                 lambda l: l.backward(), kv_window=KV, rope_cap=CAP, sink_size=SINK)
    assert loss == 0.0 and rec["replay_dev"] == 0.0
    gs = [p.grad for p in m.parameters() if p.grad is not None]
    assert gs and all(float(g.abs().max()) == 0.0 for g in gs)


def test_15_resume_fast_forwards_both_lr_schedules(tmp_path):
    """续训（只恢复权重）后第一步的 lr_g / lr_c == 不间断那次同一步的值（上游 dmd_arope 从 warmup 头重来）。"""
    import csv
    common = ["--dry_run", "--num_workers", "0", "--warmup", "10", "--log_every", "1"]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    S3.main(common + ["--output", a, "--max_steps", "5", "--save_every", "0"])
    S3.main(common + ["--output", b, "--max_steps", "3", "--save_every", "3"])
    S3.main(common + ["--output", b, "--max_steps", "5", "--save_every", "0", "--resume", "auto"])

    def rows(d):
        with open(os.path.join(d, "train_log.csv")) as fh:
            return {int(r["step"]): r for r in csv.DictReader(fh)}

    ra, rb = rows(a), rows(b)
    for s in (4, 5):
        for k, lr in (("lr_g", 2e-6), ("lr_c", 1e-6)):
            want = lr * min(1.0, (s + 1) / 10)                     # 第 s 步记的是 slr.step() s 次之后 = lr_lambda(s)
            assert abs(float(rb[s][k]) - want) < 1e-12 and float(ra[s][k]) == float(rb[s][k]), (s, k, rb[s][k])


@pytest.mark.skipif(os.environ.get("GC_GPU_FSDP") != "1", reason="GPU：GC_GPU_FSDP=1 + 2 张卡（CUDA_VISIBLE_DEVICES）")
def test_16_fsdp_rollout_numerics_equal_deploy_gpu(tmp_path):
    """★ 真 FSDP（2 进程 accelerate，同一份 yaml、同一个 _fsdp_gate）：学生 rollout 的 (dtype, t) == 部署；critic ≡ teacher
    ⇒ dmd_gnorm 恰好 0；目标 = rollout 输出 ⇒ 重放 loss / replay_dev / 学生梯度恰好 0，opt.step 之后（_free_fsdp_root）也是。"""
    import subprocess
    out = str(tmp_path / "res.json")
    cmd = [os.path.join(M, ".venv", "bin", "accelerate"), "launch", "--config_file",
           os.path.join(M, "configs", "accelerate_fsdp_dmd_2gpu.yaml"), "--num_processes", "2",
           "--main_process_port", os.environ.get("GC_GPU_FSDP_PORT", "29657"),
           os.path.join(M, "code", "gc25d", "tests", "fsdp_t_parity_worker.py"), out]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = json.load(open(out))
    assert res["world"] == 2 and res["cast_root_forward_inputs"] is False
    want = [["bfloat16", t] for t in (1000.0, 752.0, 500.0, 252.0, 0.0)]
    assert res["deploy_seq"] == [want] * 3, res["deploy_seq"]
    assert res["train_seq"] == res["deploy_seq"], res["train_seq"]
    assert res["gen_dtype"] == "bfloat16"
    assert res["dmd_gnorm"] == 0.0 and res["replay_loss"] == 0.0 and res["replay_dev"] == 0.0, res
    assert res["student_gnorm"] == 0.0, res
    # opt.step 之后：不释放 FSDP 根 ⇒ 根单元用上一步的参数（负对照）；_free_fsdp_root 之后仍逐位
    assert res["replay_dev_after_update_no_free"] > 0.0, res
    assert res["root_freed"] is True and res["replay_dev_after_update_free"] == 0.0, res


# ----------------------------------------------------------------------------
# 17 / 18. eval_rollout：按 run 分层选 clip、长链
# ----------------------------------------------------------------------------

def _rows(runs=(("v10_real", "R1"), ("v10_real", "R2"), ("v9_stylized", "P1"), ("v9_stylized", "P2")), n=14,
          ep_break=None, no_ext=()):
    out = []
    for sub, run in runs:
        for i in range(n):
            s = 96 * i
            out.append({"clip_id": f"sc_{run}__{run}__{s:06d}", "subset": sub, "scene": f"sc_{run}", "run": run,
                        "start": s, "episode": int(ep_break is not None and i >= ep_break), "camera_variant": 1,
                        "has_ext": (run, s) not in no_ext, "drop": False, "look": 0})
    return out


def test_17_eval_selection_is_stratified_by_run_and_chains_are_contiguous():
    from gcdistill import eval_rollout as ER
    rows = _rows()
    pick = ER.pick_clips(rows, 4)
    assert [c.split("__")[1] for c in pick] == ["R1", "P1", "R2", "P2"], pick       # subset 交错、每条不同 run
    assert all(c.endswith("__000000") for c in pick)
    pick6 = ER.pick_clips(rows, 6)
    assert len({c.split("__")[1] for c in pick6}) == 4 and pick6[4].endswith("__000096")
    # 241 cell = 帧 0..960 ⇒ 11 条 clip（0..960），除最后一条外都要 has_ext
    by = {r["start"]: r for r in rows if r["run"] == "R1"}
    ch = ER.find_chain(by, 0, 241)
    assert [r["start"] for r in ch] == [96 * i for i in range(11)]
    assert ER.find_chain(by, 96 * 4, 241) is None                                    # 只剩 10 条
    by_noext = {r["start"]: r for r in _rows(no_ext={("R1", 96 * 3)}) if r["run"] == "R1"}
    assert ER.find_chain(by_noext, 0, 241) is None                                    # 中间断了 ext
    by_ep = {r["start"]: r for r in _rows(ep_break=5) if r["run"] == "R1"}
    assert ER.find_chain(by_ep, 0, 241) is None and ER.find_chain(by_ep, 0, 97) is not None   # 换 episode
    chains = ER.pick_chains(rows, 4, 241)
    assert [c[0]["run"] for c in chains] == ["R1", "P1", "R2", "P2"] and all(len(c) == 11 for c in chains)
    # 长链的 GT：有效帧之外是 NaN，不进均值
    g = np.full((13, 8, 16, 3), 100, np.uint8)
    gt = g.copy()
    gt[5:9] = 0
    valid = np.ones(13, bool)
    valid[5:9] = False
    met = ER.rollout_metrics(g, gt, valid)
    assert met["gt_frames"] == 9 and met["psnr_cell"][2] is None and abs(met["gt_luma_gap"]) < 1e-6


_MINI = os.path.join(M, "data", "gc25d_v10_mini" if DG.GEO.RES_TAG == DG.GEO.DEFAULT_RES
                     else f"gc25d_v10_{DG.GEO.DST_H}_mini")


@pytest.mark.skipif(not (os.path.isfile(os.path.join(_MINI, "index", "clips.jsonl"))
                         and DG.GEO.root_res(_MINI) == DG.GEO.RES_TAG),
                    reason=f"需要 {os.path.basename(_MINI)}（res = GC_RES）")
def test_18_load_chain_matches_the_single_clip_sample_on_real_data():
    from gcdistill import eval_rollout as ER
    root = _MINI
    rows = DG.read_index(root, "val", keep_drop=True)
    ch = ER.pick_chains(rows, 1, 121)[0]
    tab = DG.load_text_table(os.path.join(root, "text_table_gc25d.pt"))
    s = ER.load_chain(root, ch, 121, tab, DG.key_to_id(tab), "parts")
    assert tuple(s["sem_ids"].shape[:1]) == (DG.frames_of(121),) and s["context"].shape[0] == 121
    one = DG.GcRolloutDataset(root, "val", tab, vocab="parts", n_cells=28).load(ch[0]["clip_id"])
    assert torch.equal(one["sem_ids"], s["sem_ids"][:DG.ROLLOUT_FRAMES]) and one["prompts"] == s["prompts"][:28]
    assert torch.equal(one["z0"], s["z0"])
    nxt = np.load(os.path.join(root, "cond", f"{ch[1]['clip_id']}.npz"))["ids"]
    assert np.array_equal(s["sem_ids"][96:96 + 81].numpy(), nxt)                    # clip 1 落在相对帧 96


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
