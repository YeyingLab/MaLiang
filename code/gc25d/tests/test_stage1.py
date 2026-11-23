#!/usr/bin/env python3
"""stage1（ar_tf_gc.py，teacher forcing 因果化）的门禁。tiny CPU 模型 / 合成数据。

* test_1  ★ loss 的接线：语义图原样进 mask_input（同一个对象）、clean_x 是干净 x0、first_frame_cond=True；
            解析预言机（v ≡ ε − x0）下 loss 恰好 0；cell 0 上的任意误差不进 loss
* test_2  --no-teacher_forcing 回退：clean_x=None
* test_3  tf_clean_stream：0 ⇒ 精确干净；>0 ⇒ 近干净端 + cell 0 仍精确干净
* test_4  ★ dry run 端到端：导出 + progress.json + metadata（teacher_forcing / windowed_* / init_ckpt / vocab / format），
            stage2 的配方闸认它、也拦配方不符的 stage2
* test_5  --cond_lr 参数组
* test_6  argparse 的配方默认值
"""
import os
import sys

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from gcdistill import ar_tf_gc as S1  # noqa: E402
from gcdistill import cd_gc as S2  # noqa: E402
from gcdistill import data_gc as DG  # noqa: E402
import geometry as GEO  # noqa: E402
from distill import runlib  # noqa: E402
from distill.flow_pf import NUM_TRAIN_TIMESTEPS, add_noise_pf, t_of_idx  # noqa: E402


def _batch(seed=0):
    c, f, h, w = DG.TINY_SHAPE
    s = DG.SynGcDataset(4, f, (h, w), seed=seed)[0]
    return DG.gc_collate_basic([s])


class _Spy:
    """记下每次前向的 kwargs；返回给定的速度场。"""

    def __init__(self, v_fn):
        self.v_fn, self.calls = v_fn, []

    def __call__(self, x, t, ctx, **kw):
        self.calls.append(dict(x=x, t=t, ctx=ctx, **kw))
        return self.v_fn(x)


def test_1_loss_wiring_and_oracle():
    b = _batch()
    x0 = b["input_latents"].float()
    b["input_latents"] = x0
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(3))
    target = noise - x0
    spy = _Spy(lambda x: target.clone())
    loss, st = S1.ar_tf_loss(spy, b, noise=noise)
    assert float(loss) == 0.0 and float(st["mse"]) == 0.0, "解析预言机 v ≡ ε − x0 下 loss 必须恰好是 0"
    c = spy.calls[0]
    assert c["mask_input"] is b["sem_ids"], "语义图必须原样走 mask_input"
    assert c["clean_x"] is x0 and c["aug_t"] is None, "TF 的 clean 半流必须是精确干净的 x0"
    assert c["first_frame_cond"] is True and c["context_ids"] is b["context_ids"]
    assert torch.equal(c["x"][:, :, 0], x0[:, :, 0]) and float(c["t"][0, 0]) == 0.0, "cell 0 = 干净 x0 + t=0"
    assert "offset_px" not in c, "gc25d 没有世界坐标"
    # cell 0 上的任意误差不进 loss
    bad = target.clone()
    bad[:, :, 0] += 100.0
    loss0, _ = S1.ar_tf_loss(_Spy(lambda x: bad.clone()), b, noise=noise)
    assert float(loss0) == 0.0
    # 其它 cell 的误差要进
    bad2 = target.clone()
    bad2[:, :, 2] += 1.0
    loss2, st2 = S1.ar_tf_loss(_Spy(lambda x: bad2.clone()), b, noise=noise)
    assert float(loss2) > 0.0 and abs(float(st2["mse"]) - 1.0 / (x0.shape[2] - 1)) < 1e-6


def test_1b_sem_shape_gate():
    b = _batch()
    b["sem_ids"] = b["sem_ids"][:, :-4]
    with pytest.raises(AssertionError):
        S1.ar_tf_loss(_Spy(lambda x: torch.zeros_like(x)), b)


def test_2_no_teacher_forcing_fallback():
    b = _batch()
    spy = _Spy(lambda x: torch.zeros_like(x, dtype=torch.float32))
    S1.ar_tf_loss(spy, b, teacher_forcing=False)
    assert spy.calls[0]["clean_x"] is None


def test_3_tf_clean_stream():
    x0 = torch.randn(1, 48, 5, 8, 10)
    noise = torch.randn_like(x0)
    c, a = S1.tf_clean_stream(x0, noise, 0)
    assert c is x0 and a is None
    torch.manual_seed(0)
    c, a = S1.tf_clean_stream(x0, noise, 50)
    assert torch.equal(c[:, :, 0], x0[:, :, 0]) and float(a[0, 0]) == 0.0
    assert float(a[:, 1:].max()) <= float(t_of_idx(NUM_TRAIN_TIMESTEPS - 50)) + 1e-3, "增广必须在近干净端（高下标 ⇒ 小 t）"
    assert not torch.equal(c[:, :, 1:], x0[:, :, 1:])
    idx = torch.full((1, 5), NUM_TRAIN_TIMESTEPS - 1)
    assert float((add_noise_pf(x0, noise, idx) - x0).abs().max()) < 0.05


def test_4_dry_run_end_to_end_metadata_and_stage2_gate(tmp_path):
    out = str(tmp_path / "s1")
    step = S1.main(["--dry_run", "--output", out, "--num_workers", "0", "--max_steps", "4", "--save_every", "2",
                    "--log_every", "1", "--val_every", "0", "--warmup", "2"])
    assert step == 4
    p = runlib.dit_path(out, 4)
    assert os.path.isfile(p) and os.path.isfile(runlib.dit_path(out, 2))
    assert runlib.read_progress(out)["step"] == 4
    _, meta = runlib.read_ckpt_header(p)
    for k, v in {"stage": "stage1_ar_tf", "teacher_forcing": "True", "windowed_train": "0", "windowed_sink": "0",
                 "vocab": "parts", "format": "gc25d", "arm": "gc25d", "sem_cond": "gc_sem_cond",
                 "mask_channel": "False"}.items():
        assert meta.get(k) == v, (k, meta.get(k))
    assert "init_ckpt" in meta and "mask_p" in meta and "text_dropout" in meta
    ok = types.SimpleNamespace(teacher_forcing=True, windowed_train=0, windowed_sink=0)
    assert S2._ckpt_meta_gate(p, ok, log=lambda *a: None)["stage"] == "stage1_ar_tf"
    with pytest.raises(SystemExit):
        S2._ckpt_meta_gate(p, types.SimpleNamespace(teacher_forcing=False, windowed_train=0, windowed_sink=0),
                           log=lambda *a: None)


def test_5_cond_lr_param_group():
    m = DG.tiny_gcdit(causal=True)
    a = S1.parse_args(["--dry_run", "--output", "/tmp/unused", "--cond_lr", "1e-4"])
    groups = S1.param_groups(m, a)
    assert isinstance(groups, list) and len(groups) == 2 and groups[1]["lr"] == 1e-4
    n_cond = sum(p.numel() for p in groups[1]["params"])
    assert n_cond == sum(p.numel() for p in m.dit.gc_sem_cond.parameters())
    a0 = S1.parse_args(["--dry_run", "--output", "/tmp/unused"])
    assert not isinstance(S1.param_groups(m, a0), list), "cond_lr=0 ⇒ 一个参数组"


def test_6_recipe_defaults():
    a = S1.parse_args(["--dry_run", "--output", "/tmp/unused"])
    assert (a.lr, a.betas, a.weight_decay, a.warmup, a.grad_clip) == (1e-5, (0.9, 0.999), 0.01, 200, 10.0)
    assert (a.batch_size, a.grad_accum, a.max_steps, a.save_every) == (1, 1, 8000, 500)
    assert (a.text_dropout, a.cond_dropout, a.mask_p) == (0.1, 0.05, 0.5)
    assert a.teacher_forcing and a.windowed_train == 0 and a.windowed_sink == 0 and a.gradient_checkpointing


MINI = os.path.join(M, "data", "gc25d_v10_mini" if GEO.RES_TAG == GEO.DEFAULT_RES else f"gc25d_v10_{GEO.DST_H}_mini")


@pytest.mark.skipif(not (os.path.isfile(os.path.join(MINI, "READY")) and GEO.root_res(MINI) == GEO.RES_TAG),
                    reason=f"{os.path.basename(MINI)} 还没就绪（或 res ≠ {GEO.RES_TAG}）")
def test_7_data_contract_real_dataset_vs_synthetic():
    """SFT 的 GcLatentDataset（惰性 import）与全尺寸合成样本给出同一份契约：键 / 形状 / dtype，文本 id = 排序键下标；
    真样本直接过 ar_tf_loss 的形状闸。stage3 的 GcRolloutDataset 与 GcLatentDataset 的前 21 个 cell 逐位相同。"""
    ds, collate = DG.latent_dataset(MINI, "train", os.path.join(MINI, "text_table_gc25d.pt"), vocab="parts",
                                    text_dropout=0.0, cond_dropout=0.0, mask_p=0.0, limit=2)
    real = ds[0]
    syn = DG.SynGcDataset(1, 21, DG.LAT_HW, vocab="parts", text_table=ds.text_table)[0]
    for k in ("input_latents", "context", "context_ids", "sem_ids"):
        assert tuple(real[k].shape) == tuple(syn[k].shape) and real[k].dtype == syn[k].dtype, k
    assert DG.key_to_id(ds.text_table) == ds.key_to_id
    b = collate([real])
    spy = _Spy(lambda x: torch.zeros_like(x, dtype=torch.float32))
    S1.ar_tf_loss(spy, b)
    assert spy.calls[0]["mask_input"].shape == (1, 81, GEO.DST_H, GEO.DST_W)
    roll = DG.GcRolloutDataset(MINI, "train", ds.text_table, vocab="parts")
    cid = roll.clip_ids()[0]
    r = roll.load(cid)
    ds_all, _ = DG.latent_dataset(MINI, "train", ds.text_table, vocab="parts", text_dropout=0.0,
                                  cond_dropout=0.0, mask_p=0.0)
    full = ds_all[ds_all.clip_ids().index(cid)]
    assert torch.equal(r["z0"], full["input_latents"][:, :1])
    assert torch.equal(r["sem_ids"][:81], full["sem_ids"]) and torch.equal(r["context_ids"][:21], full["context_ids"])
    assert torch.equal(r["context"][:21], full["context"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
