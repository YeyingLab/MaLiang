#!/usr/bin/env python3
"""stage2（cd_gc.py，因果一致性蒸馏）的门禁 —— ARoPE tests/test_stage2_cd.py 的 gc 移植。全部 CPU、秒级。

* test_1  ★ CD 轴的节点逐点就是部署读点（1000/750/500/250 → σ [1, .75, .5006, .2516]），轴覆盖部署表
* test_2  ★ 解析预言机（v ≡ ε − x0）把 cd_loss 钉成恒等于 0（每一对都测）；2b 负控制：cm_tn 读错 σ 必须 ≠ 0
* test_3  梯度只到学生，教师 / EMA 是 0
* test_4  三次前向共用**同一批** cond 对象（含 mask_input = 语义图），t / t_next 用对
* test_5  cell 0 在三条流里都被钉成 GT
* test_6  _keep_frame0 out-of-place、可反传
* test_7  stage1 metadata 配方不符必须 SystemExit；读不出 metadata 只 warn
* test_8  discrete 轴必须报出「部署读点掉出轴」
* test_9  ★ EMA 影子是 fp32（bf16 在 decay 0.99 下会把 1% 的增量舍掉），.to(bf16) 被拒；update 数学；硬拷贝
* test_10 dry run 端到端：dits/ + ema/ 都导出、ema 角色 metadata、续训从 progress.json 接上
"""
import os
import sys
import types

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import pytest  # noqa: E402
import torch  # noqa: E402

from gcdistill import cd_gc as S2  # noqa: E402
from gcdistill import data_gc as DG  # noqa: E402
from distill import flow_pf, runlib  # noqa: E402
from distill.ema import EmaShadow  # noqa: E402

TOL_ORACLE = 1e-10


def _batch(dtype=torch.float32, seed=0):
    c, f, h, w = DG.TINY_SHAPE
    b = DG.gc_collate_basic([DG.SynGcDataset(4, f, (h, w), seed=seed)[0]])
    b["input_latents"] = b["input_latents"].to(dtype)
    b["context"] = b["context"].to(dtype)
    return b


def _axis():
    return S2.build_cd_axis("deploy", flow_pf.DEPLOY_DENOISE_LIST, 48, 0.0, log=lambda *a: None)


class _Oracle:
    """流匹配的解析速度场 v(x, t) ≡ ε − x0：单个 Euler 步精确、x0 估计 x_σ − σ·v ≡ x0 对任何 σ 成立 ⇒ cd_loss ≡ 0。"""

    def __init__(self, x0, noise):
        self.v = noise - x0
        self.calls = []

    def __call__(self, x, t, **kw):
        kw.pop("use_gradient_checkpointing", None)
        self.calls.append((x, float(t), kw))
        return self.v.clone()


def test_1_deploy_axis_nodes_are_exactly_the_deploy_read_points():
    sig, tst, info = _axis()
    _, d_sig, d_t = flow_pf.deploy_schedule(flow_pf.DEPLOY_DENOISE_LIST)
    assert torch.equal(sig, d_sig) and torch.equal(tst, d_t)
    assert info["K"] == 4 and info["pairs"] == 3 and info["deploy_off_axis"] == []
    assert [round(float(s), 4) for s in sig] == [1.0, 0.75, 0.5006, 0.2516]
    flow_pf.assert_deploy_covered(sig, flow_pf.DEPLOY_DENOISE_LIST)
    # stage2 argparse 的默认表就是部署表
    a = S2.parse_args(["--dry_run", "--output", "/tmp/unused"])
    assert a.cd_axis == "deploy" and a.denoise == list(flow_pf.DEPLOY_DENOISE_LIST)


def test_2_oracle_makes_cd_loss_exactly_zero():
    batch = _batch()
    x0 = batch["input_latents"]
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(7))
    sig, tst, _ = _axis()
    for i in range(len(sig) - 1):
        m = _Oracle(x0, noise)
        loss, stats = S2.cd_loss(m, m, m, batch, sig_t=float(sig[i]), sig_tn=float(sig[i + 1]),
                                 t_v=float(tst[i]), tn_v=float(tst[i + 1]), noise=noise)
        assert float(loss) < TOL_ORACLE, (i, float(loss))
        assert float(stats["cm_err"]) < TOL_ORACLE and float(stats["tgt_err"]) < TOL_ORACLE


def test_2b_oracle_loss_is_nonzero_when_the_target_reads_the_wrong_sigma():
    batch = _batch()
    x0 = batch["input_latents"]
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(7))
    sig, _, _ = _axis()
    s_t, s_tn = float(sig[0]), float(sig[1])
    v = noise - x0
    lat_tn = (1.0 - s_tn) * x0 + s_tn * noise
    assert torch.allclose(lat_tn - s_tn * v, x0, atol=1e-6)
    assert float(torch.nn.functional.mse_loss((lat_tn - s_t * v)[:, :, 1:], x0[:, :, 1:])) > 1e-3


def test_3_gradient_reaches_only_the_student():
    student, teacher, ema = (DG.tiny_gcdit(causal=True) for _ in range(3))
    for m in (teacher, ema):
        m.eval().requires_grad_(False)
    student.train()
    sig, tst, _ = _axis()
    loss, _ = S2.cd_loss(student, ema, teacher, _batch(dtype=torch.bfloat16),
                         sig_t=float(sig[0]), sig_tn=float(sig[1]), t_v=float(tst[0]), tn_v=float(tst[1]),
                         gen=torch.Generator().manual_seed(3))
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    loss.backward()
    assert sum(1 for p in student.parameters() if p.grad is not None and p.grad.abs().sum() > 0) > 0
    for m in (teacher, ema):
        assert not [n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    g_cond = student.dit.gc_sem_cond.proj_out.weight.grad
    assert g_cond is not None and float(g_cond.abs().sum()) > 0, "语义条件模块没拿到梯度"


def test_4_the_three_forwards_share_one_cond():
    batch = _batch()
    x0 = batch["input_latents"]
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    sig, tst, _ = _axis()
    m = _Oracle(x0, noise)
    S2.cd_loss(m, m, m, batch, sig_t=float(sig[0]), sig_tn=float(sig[1]),
               t_v=float(tst[0]), tn_v=float(tst[1]), noise=noise)
    assert len(m.calls) == 3
    base = m.calls[0][2]
    for _, _, cond in m.calls[1:]:
        for k in ("context", "context_ids", "mask_input", "first_frame_cond", "clean_x"):
            a, b = base[k], cond[k]
            assert (a is b) if isinstance(a, torch.Tensor) else (a == b), k
    assert base["mask_input"] is batch["sem_ids"] and base["mask_input"].dtype == torch.uint8
    assert base["first_frame_cond"] is True and base["clean_x"] is x0 and "offset_px" not in base
    assert m.calls[0][1] == float(tst[0]) and m.calls[1][1] == float(tst[0]) and m.calls[2][1] == float(tst[1])


def test_5_cell0_is_pinned_to_ground_truth_in_all_three_streams():
    batch = _batch()
    x0 = batch["input_latents"]
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(2))
    sig, tst, _ = _axis()
    m = _Oracle(x0, noise)
    S2.cd_loss(m, m, m, batch, sig_t=float(sig[0]), sig_tn=float(sig[1]),
               t_v=float(tst[0]), tn_v=float(tst[1]), noise=noise)
    for x, _, _ in m.calls:
        assert torch.equal(x[:, :, 0], x0[:, :, 0])
    assert not torch.equal(m.calls[1][0][:, :, 1:], x0[:, :, 1:])


def test_6_keep_frame0_is_out_of_place_and_differentiable():
    a = torch.randn(1, 2, 4, 3, 3, requires_grad=True)
    clip = torch.randn(1, 2, 4, 3, 3)
    out = S2._keep_frame0(a, clip)
    assert out is not a and torch.equal(out[:, :, 0], clip[:, :, 0]) and torch.equal(out[:, :, 1:], a[:, :, 1:])
    out.sum().backward()
    assert float(a.grad[:, :, 0].abs().sum()) == 0.0 and float(a.grad[:, :, 1:].abs().sum()) > 0.0


def test_7_ckpt_metadata_gate_rejects_a_recipe_mismatch(tmp_path):
    from safetensors.torch import save_file
    p = str(tmp_path / "dit_step100.safetensors")
    save_file({"w": torch.zeros(2)}, p, metadata={"stage": "stage1_ar_tf", "step": "100", "teacher_forcing": "False",
                                                 "windowed_train": "0", "windowed_sink": "0"})
    args = types.SimpleNamespace(teacher_forcing=True, windowed_train=0, windowed_sink=0)
    with pytest.raises(SystemExit) as e:
        S2._ckpt_meta_gate(p, args, log=lambda *a: None)
    assert "teacher_forcing" in str(e.value)
    ok = types.SimpleNamespace(teacher_forcing=False, windowed_train=0, windowed_sink=0)
    assert S2._ckpt_meta_gate(p, ok, log=lambda *a: None)["teacher_forcing"] == "False"
    p3 = str(tmp_path / "garbage.safetensors")
    open(p3, "wb").write(b"not a safetensors file at all")
    assert S2._ckpt_meta_gate(p3, args, log=lambda *a: None) == {}


def test_8_discrete_axis_reports_the_deploy_points_that_fall_off_it():
    lines = []
    _, _, info = S2.build_cd_axis("discrete", flow_pf.DEPLOY_DENOISE_LIST, 48, 0.0, log=lines.append)
    assert info["K"] == 48 and info["deploy_off_axis"] and any("[axis][warn]" in ln for ln in lines)


def test_9_ema_shadow_is_fp32():
    s = DG.tiny_gcdit(causal=True)                                   # bf16 学生
    ema = EmaShadow(s)
    assert all(p.dtype is torch.float32 for p in ema.parameters()), "EMA 影子必须是 fp32"
    with pytest.raises(RuntimeError):
        ema.to(torch.bfloat16)
    p_s, p_e = next(s.parameters()), next(ema.parameters())
    before = p_e.detach().clone()
    with torch.no_grad():
        p_s.add_(1.0)
    ema.update(s, 0.99)
    exp = 0.99 * before + 0.01 * p_s.detach().float()
    assert torch.allclose(p_e, exp, atol=1e-6), "pe = 0.99·pe + 0.01·p"
    assert float((p_e - before).abs().max()) > 0, "1% 的增量被舍掉了（bf16 影子的典型症状）"
    ema.hard_copy(s)
    assert torch.equal(p_e, p_s.detach().float())


def test_10_dry_run_saves_dits_and_ema_and_resumes(tmp_path):
    out = str(tmp_path / "s2")
    common = ["--dry_run", "--output", out, "--num_workers", "0", "--save_every", "2", "--log_every", "1",
              "--warmup", "1", "--ema_start", "1"]
    assert S2.main(common + ["--max_steps", "2"]) == 2
    for p in (runlib.dit_path(out, 2), runlib.ema_path(out, 2)):
        assert os.path.isfile(p)
    _, meta = runlib.read_ckpt_header(runlib.ema_path(out, 2))
    assert meta["role"] == "ema_deploy" and meta["format"] == "gc25d" and meta["denoise_list"] == "1000,750,500,250"
    prog = runlib.read_progress(out)
    assert prog["axis_sigmas"] == [1.0, 0.75, 0.5006, 0.251597] and prog["grad_accum"] == 1
    assert S2.main(common + ["--max_steps", "4", "--resume", "auto", "--force_run_dir"]) == 4
    assert os.path.isfile(runlib.ema_path(out, 4))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
