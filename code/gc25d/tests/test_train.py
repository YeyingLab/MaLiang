"""train.py 的 CPU 测试：tiny WanModel 走 GcTrainModule 前向 / 反向、TI2V 首帧、sem 项从 0 长起来、参数组、warmup、
验证三支（cond / uncond / nulltext）、导出键集与 metadata、CLI 默认值 = 。

    PYTHONPATH=code/gc25d:code/arope .venv/bin/python -m pytest -q code/gc25d/tests/test_train.py
"""
from __future__ import annotations


import os as _mos


def _maliang_repo_root(start):
    """Locate the repository root by walking up until code/gc25d exists."""
    p = _mos.path.abspath(start)
    while True:
        if _mos.path.isdir(_mos.path.join(p, "code", "gc25d")):
            return p
        parent = _mos.path.dirname(p)
        if parent == p:
            return _mos.getcwd()
        p = parent


MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _maliang_repo_root(__file__)
import json
import struct

import pytest
import torch

import model as GM
import train as TR
from diffsynth.models.wan_video_dit import WanModel

# tiny 几何同 code/arope/distill/tests/preflight.py::TINY（head_dim 12 ⇒ RoPE 三段列数凑得上）：latent 4×4 → 2×2 token / cell
TINY = dict(dim=24, in_dim=48, ffn_dim=48, out_dim=48, text_dim=4096, freq_dim=256,
            eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=2, has_image_input=False)
F_LAT, LAT_H, LAT_W, L_TXT = 3, 4, 4, 6
N_IMG = 4 * (F_LAT - 1) + 1


def tiny_module(vocab: str = "parts", seed: int = 0, gc: bool = True) -> TR.GcTrainModule:
    torch.manual_seed(seed)
    dit = WanModel(**TINY).float()
    GM.install_sem_cond(dit, vocab)
    dit.train().requires_grad_(True)
    return TR.GcTrainModule(dit, first_frame_cond=True, gradient_checkpointing=gc)


def tiny_batch(seed: int = 0, b: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    raw = torch.tensor(sorted(GM.valid_raw_ids("parts")), dtype=torch.uint8)
    sem = raw[torch.randint(0, len(raw), (b, N_IMG, LAT_H * 16, LAT_W * 16), generator=g)]
    sem[..., :32, :] = 1                                      # 一块大面积的静态类，免得全是噪声
    return {
        "input_latents": torch.randn(b, 48, F_LAT, LAT_H, LAT_W, generator=g),
        "context": torch.randn(b, F_LAT, L_TXT, 4096, generator=g),
        "context_ids": torch.arange(F_LAT).repeat(b, 1),
        "sem_ids": sem,
        "clip_id": [f"c{seed}_{i}" for i in range(b)],
    }


# ----------------------------------------------------------------------------
# GcTrainModule
# ----------------------------------------------------------------------------

def test_forward_backward_finite_and_grads():
    m = tiny_module()
    batch = tiny_batch()
    loss, stats = m(batch, timestep_id=300)
    assert torch.isfinite(loss) and loss.item() > 0
    assert set(stats) == {"mse", "timestep", "sem_rms", "x_rms"}
    assert stats["sem_rms"].item() == 0.0 and stats["x_rms"].item() > 0     # proj_out 零初始化 ⇒ sem 项恰为 0
    loss.backward()
    enc = getattr(m.dit, GM.SEM_COND_NAME)
    assert enc.proj_out.weight.grad is not None and enc.proj_out.weight.grad.abs().sum() > 0
    assert m.dit.patch_embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for n, p in m.named_parameters() if not n.startswith(TR.COND_PREFIX))


def test_loss_matches_manual_and_excludes_frame0():
    m = tiny_module(gc=False)
    batch = tiny_batch()
    tid = 417
    noise = torch.randn_like(batch["input_latents"])
    loss, stats = m(batch, tid, noise=noise.clone())
    # 手算：x_t、帧 0 换成干净 x0、t 置 0（gc_forward 里）、loss = mse(帧 1..) × w(t)
    x0 = batch["input_latents"]
    t = m.scheduler.timesteps[tid]
    sig = m.scheduler.sigmas[tid]
    x_t = (1 - sig) * x0 + sig * noise
    x_t[:, :, 0:1] = x0[:, :, 0:1]
    pred = GM.gc_forward(m.dit, x_t, t.view(1), batch["context"], batch["sem_ids"], context_ids=batch["context_ids"])
    mse = ((pred[:, :, 1:] - (noise - x0)[:, :, 1:]) ** 2).mean()
    w = m.scheduler.training_weight(t.view(1))
    assert torch.allclose(stats["mse"], mse, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss, mse * w, rtol=1e-5, atol=1e-6)
    # 帧 0 的噪声不进输入、帧 0 不计 loss ⇒ 改帧 0 的噪声 loss 不变
    noise2 = noise.clone()
    noise2[:, :, 0] = torch.randn_like(noise2[:, :, 0]) * 7
    loss2, _ = m(batch, tid, noise=noise2)
    assert torch.equal(loss, loss2)


def test_sem_rms_grows_from_zero_and_condition_is_wired():
    m = tiny_module()
    groups = TR.sem_param_groups(m, lr=1e-4, cond_lr=1e-2, weight_decay=0.01)
    opt = torch.optim.AdamW(groups)
    batch = tiny_batch()
    rms = []
    for k in range(3):
        loss, stats = m(batch, timestep_id=500)
        rms.append(stats["sem_rms"].item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert rms[0] == 0.0 and 0 < rms[1] < rms[2], rms
    # 条件确实接上了：换 sem_ids（全 255）⇒ 预测变了
    with torch.no_grad():
        noise = torch.randn_like(batch["input_latents"])
        _, s1 = m(batch, 500, noise=noise, train=False)
        _, s2 = m(dict(batch, sem_ids=torch.full_like(batch["sem_ids"], 255)), 500, noise=noise, train=False)
    assert s1["mse"].item() != s2["mse"].item()


def test_vocab_ids_module_runs():
    m = tiny_module(vocab="ids")
    batch = tiny_batch()
    batch["sem_ids"] = GM.collapse_parts(batch["sem_ids"])
    loss, _ = m(batch, 100)
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------------
# 优化器 / 调度
# ----------------------------------------------------------------------------

def test_sem_param_groups():
    m = tiny_module()
    groups = TR.sem_param_groups(m, lr=1e-5, cond_lr=1e-4, weight_decay=0.01)
    assert len(groups) == 2
    assert [g["lr"] for g in groups] == [1e-5, 1e-4] and all(g["weight_decay"] == 0.01 for g in groups)
    names = {id(p): n for n, p in m.named_parameters()}
    cond = [names[id(p)] for p in groups[1]["params"]]
    rest = [names[id(p)] for p in groups[0]["params"]]
    assert len(cond) == 6 and all(n.startswith("dit.gc_sem_cond.") for n in cond)
    assert not any(n.startswith("dit.gc_sem_cond.") for n in rest)
    assert len(cond) + len(rest) == len(list(m.parameters()))
    opt = torch.optim.AdamW(groups, lr=1e-5, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, TR.warmup_lambda(4))
    lrs = []
    for _ in range(6):
        lrs.append(sch.get_last_lr())
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx([2.5e-6, 2.5e-5]) and lrs[3] == pytest.approx([1e-5, 1e-4]) and lrs[5] == lrs[3]


def test_warmup_lambda():
    f = TR.warmup_lambda(500)
    assert f(0) == pytest.approx(1 / 500) and f(499) == 1.0 and f(20000) == 1.0
    assert TR.warmup_lambda(0)(0) == 1.0


def test_cli_defaults_match_spec():
    a = TR.parse_args(["--output", _mos.path.join(MALIANG_ROOT, "outputs", "_x")])
    assert (a.lr, a.cond_lr, a.weight_decay, a.warmup_steps, a.grad_clip) == (1e-5, 1e-4, 0.01, 500, 1.0)
    assert (a.max_steps, a.batch_size, a.grad_accum, a.save_every, a.val_every, a.val_n) == (20000, 1, 2, 1000, 500, 16)
    assert a.val_timesteps == [200.0, 500.0, 800.0] and a.val_uncond and a.save_state and a.gradient_checkpointing
    assert (a.text_dropout, a.cond_dropout, a.mask_p, a.vocab, a.first_frame_cond) == (0.1, 0.05, 0.5, "parts", True)
    assert a.data_root.endswith("data/gc25d_v10") and a.text_table.endswith("data/gc25d_v10/text_table_gc25d.pt")


# ----------------------------------------------------------------------------
# 验证三支
# ----------------------------------------------------------------------------

class _FakeAccel:
    device = torch.device("cpu")
    process_index, num_processes = 0, 1

    @staticmethod
    def reduce(t, reduction="sum"):
        return t


def test_val_variants_and_run_validation():
    m = tiny_module(gc=False)
    null_ctx = torch.full((1, F_LAT, L_TXT, 4096), 0.5)
    null_ids = torch.full((1, F_LAT), 99, dtype=torch.int64)
    b = tiny_batch()
    vb = TR.val_variants(b, null_ctx, null_ids, uncond=True)
    assert set(vb) == {"cond", "uncond", "nulltext"}
    assert bool((vb["uncond"]["sem_ids"] == 255).all()) and torch.equal(vb["uncond"]["context"], b["context"])
    assert torch.equal(vb["nulltext"]["context"], null_ctx) and vb["nulltext"]["context_ids"].tolist() == [[99] * F_LAT]
    assert torch.equal(vb["nulltext"]["sem_ids"], b["sem_ids"])
    # val 集 = 未 collate 的样本列表（run_validation 自己 collate）
    val = []
    for i in range(3):
        s = tiny_batch(seed=10 + i)
        val.append({k: (v[0] if isinstance(v, (torch.Tensor, list)) else v) for k, v in s.items()})
    tids = [m.timestep_id_for(t) for t in (200.0, 800.0)]
    res = TR.run_validation(_FakeAccel(), m, m, val, tids, seed=0, null_context=null_ctx, null_ids=null_ids)
    assert len(res) == 2
    for t, r in res.items():
        assert r["n"] == 3 and all(torch.isfinite(torch.tensor(r[k])) for k in ("mse", "mse_uncond", "mse_nulltext"))
        assert r["gap_uncond"] == 0.0             # gc_sem_cond 零初始化 ⇒ 换成全 255 也逐位相同
        assert r["gap_nulltext"] != 0.0
    # 固定噪声 ⇒ 可复现
    res2 = TR.run_validation(_FakeAccel(), m, m, val, tids, seed=0, null_context=null_ctx, null_ids=null_ids)
    assert res == res2


# ----------------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------------

def _header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, hdr.pop("__metadata__", {})


def test_export_keys_and_metadata(tmp_path):
    from gcdistill.gc_dit import save_gc, SEM_KEYS
    m = tiny_module()
    args = TR.parse_args(["--output", str(tmp_path), "--vocab", "parts"])
    meta = TR.export_metadata(args, 7)
    assert all(isinstance(v, str) for v in meta.values())
    path = str(tmp_path / "step-7.safetensors")
    save_gc(None, m, path, meta, dry_run=True)                 # tiny 模型键数不是 831 ⇒ dry_run 只跳过键数闸门
    hdr, md = _header(path)
    assert set(hdr) == set(m.dit.state_dict())                 # 去掉了 dit. 前缀，没有多余键
    assert all(k in hdr for k in SEM_KEYS) and {v["dtype"] for v in hdr.values()} == {"BF16"}
    for k, v in {"format": "gc25d", "arm": "gc25d", "sem_cond": "gc_sem_cond", "vocab": "parts", "data": "v10",
                 "mask_channel": "False", "step": "7", "cond_lr": "0.0001", "mask_p": "0.5"}.items():
        assert md.get(k) == v, (k, md.get(k))
    # 真模型的键数：825 Wan + 6 gc_sem_cond
    assert TR.GC_DIT_KEYS == 831


def test_export_metadata_lineage(tmp_path):
    # 热启动：init_ckpt = 那个 safetensors；dir 续训：init_ckpt 由 main 从存档接回，续训点单列
    a = TR.parse_args(["--output", str(tmp_path), "--resume", "/x/warm.safetensors"])
    md = TR.export_metadata(a, 3)
    assert (md["init_ckpt"], md["resumed_from"], md["resumed_step"]) == ("/x/warm.safetensors", "", "0")
    b = TR.parse_args(["--output", str(tmp_path), "--resume", str(tmp_path / "state")])
    assert b.init_ckpt == ""                                   # 目录不是权重初始化


# ----------------------------------------------------------------------------
# 续训：数据位置 / 血统 / 日志
# ----------------------------------------------------------------------------

def test_epoch_steps_is_ceil():
    assert TR.epoch_steps(25, 2) == 13 and TR.epoch_steps(24, 2) == 12 and TR.epoch_steps(1494, 2) == 747
    assert TR.epoch_steps(5, 1) == 5 and TR.epoch_steps(0, 2) == 1


def test_resume_position():
    geo = {"num_processes": 4, "grad_accum": 2, "batch_size": 1, "n_micro": 25, "n_train": 100, "seed": 0}
    ts = {"step": 20, "epoch": 1, "epoch_micro": 14, **geo}
    assert TR.resume_position(ts, geo)[:2] == (1, 14)                       # 精确：接着 epoch 1 的第 15 个 micro
    assert TR.resume_position(dict(ts, epoch_micro=25), geo)[:2] == (2, 0)  # 存档时正好在 epoch 末（含 1 个 micro 的尾巴步）
    for k, v in (("num_processes", 8), ("n_micro", 12), ("seed", 1), ("n_train", 99)):
        assert TR.resume_position(dict(ts, **{k: v}), geo)[:2] == (2, 0), k  # 几何变了 ⇒ 下一个 epoch 开头
    # 旧存档（只有 step）：按向上取整的每 epoch 步数推（13 步 / epoch）
    assert TR.resume_position({"step": 10}, geo)[:2] == (0, 20)
    assert TR.resume_position({"step": 13}, geo)[:2] == (1, 0)
    assert TR.resume_position({"step": 27}, geo)[:2] == (2, 2)


def test_init_ckpt_of_state(tmp_path):
    run = tmp_path / "run"
    (run / "state_A").mkdir(parents=True)
    (run / "state").symlink_to("state_A")
    assert TR.init_ckpt_of_state(str(run / "state"), {"init_ckpt": "/w/a.safetensors"}) == "/w/a.safetensors"
    assert TR.init_ckpt_of_state(str(run / "state"), {"init_ckpt": ""}) == ""
    assert TR.init_ckpt_of_state(str(run / "state"), {"step": 5}) == ""             # 旧存档、没有 config.json
    (run / "config.json").write_text(json.dumps({"args": {"resume": "/w/b.safetensors"}}))
    assert TR.init_ckpt_of_state(str(run / "state"), {"step": 5}) == "/w/b.safetensors"
    (run / "config.json").write_text(json.dumps({"args": {"resume": str(run / "state"), "init_ckpt": ""}}))
    assert TR.init_ckpt_of_state(str(run / "state"), {"step": 5}) == ""


def test_rotate_csv_if_header_changed(tmp_path):
    p = tmp_path / "train_log.csv"
    assert TR.rotate_csv_if_header_changed(str(p), ["a", "b"]) is None           # 没有文件
    p.write_text("a,b\n1,2\n")
    assert TR.rotate_csv_if_header_changed(str(p), ["a", "b"]) is None and p.exists()
    old = TR.rotate_csv_if_header_changed(str(p), ["a", "b", "c"])
    assert old.endswith("train_log.old0.csv") and not p.exists() and (tmp_path / "train_log.old0.csv").read_text() == "a,b\n1,2\n"


class _Idx(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n, self.epoch = n, 0

    def __len__(self):
        return self.n

    def set_epoch(self, e):
        self.epoch = e

    def __getitem__(self, i):
        return {"i": torch.tensor(i), "e": torch.tensor(self.epoch)}


def test_seedable_sampler_skip_resume_and_sync_pattern():
    """真 accelerate（CPU 单进程）：SeedableRandomSampler + data_seed ⇒ 每 epoch 排列只取决于 (seed, epoch)；
    skip_first_batches 接上的顺序 = 不中断时的后半段；GA 2 × 5 个 micro ⇒ 同步 3 次（= epoch_steps 向上取整）。"""
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState
    from accelerate.utils import DataLoaderConfiguration

    def run(skip_epoch=None, skip=0):
        AcceleratorState._reset_state(True)
        GradientState._reset_state()
        torch.manual_seed(123 if skip_epoch is None else 456)     # 全局 RNG 不同也不影响排列
        acc = Accelerator(cpu=True, gradient_accumulation_steps=2,
                          dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=7))
        ds = _Idx(5)
        dl = acc.prepare(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, drop_last=True))
        lin = torch.nn.Linear(1, 1)
        orders, syncs = {}, {}
        for e in range(3):
            if skip_epoch is not None and e < skip_epoch:
                continue
            ds.set_epoch(e)
            dl.set_epoch(e)
            loader = dl
            if e == skip_epoch and skip:
                loader = acc.skip_first_batches(dl, skip)
                loader.set_epoch(e)
            orders[e], syncs[e] = [], []
            for b in loader:
                with acc.accumulate(lin):
                    syncs[e].append(acc.sync_gradients)
                orders[e].append(int(b["i"]))
                assert int(b["e"]) == e                          # dataset 的 epoch 进了 worker / 样本
        return orders, syncs

    try:
        full, syncs = run()
        assert all(sorted(o) == list(range(5)) for o in full.values()) and full[0] != full[1]
        assert syncs[0] == [False, True, False, True, True] and sum(syncs[0]) == TR.epoch_steps(5, 2)
        again, _ = run(skip_epoch=0)
        assert again == full                                       # 换了全局种子，排列不变
        resumed, rs = run(skip_epoch=1, skip=2)                    # 存档在 epoch 1 消费了 2 个 micro（1 步）之后
        assert resumed[1] == full[1][2:] and resumed[2] == full[2]
        assert rs[1] == [False, True, True]                        # 跳过的是 GA 的整数倍 ⇒ 同步边界不变
    finally:
        AcceleratorState._reset_state(True)
        GradientState._reset_state()
