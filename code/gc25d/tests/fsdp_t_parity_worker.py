#!/usr/bin/env python3
"""stage3 的训练 / 部署数值对齐（GPU，2 进程 accelerate FSDP；tests/test_stage3.py::test_16 拉起，不是 pytest 文件）。

与 dmd_gc.main 同一套包装（accelerate.prepare + dmd_gc._fsdp_gate，同一份 configs/accelerate_fsdp_dmd_2gpu.yaml），tiny GcDiT：
  1. 学生的 self-forcing rollout（streamed record 趟）里每次单 cell 前向模型内部看到的 (latent dtype, t)
     == eval_rollout.deploy_rollout（= codeworld/live.py，不包装的 bf16 模型）的同一序列：1000 / 752 / 500 / 252，bf16；
  2. critic ≡ teacher（critic FSDP 包装、teacher 不包装）⇒ dmd_gnorm 恰好 0；
  3. 目标 = rollout 自己的输出 ⇒ streamed 重放的 loss、replay_dev、学生梯度范数都恰好 0；
  4. 真的 opt.step 之后再测 3：不调 dmd_gc._free_fsdp_root ⇒ FSDP 根单元用上一步的参数，replay_dev > 0（负对照）；
     调了 ⇒ 仍恰好 0。
rank 0 把结果写进 argv[1]（json）。

    accelerate launch --config_file configs/accelerate_fsdp_dmd_2gpu.yaml --num_processes 2 \\
        code/gc25d/tests/fsdp_t_parity_worker.py <out.json>
"""
import json
import os
import sys

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402

import distill.arope_dit as AD  # noqa: E402
from distill.flow_pf import DEPLOY_DENOISE_LIST, deploy_schedule  # noqa: E402
from gcdistill import data_gc as DG  # noqa: E402
from gcdistill import dmd_gc as S3  # noqa: E402
from gcdistill import eval_rollout as ER  # noqa: E402

SEEN = []
_orig = AD._per_token_timestep_pf


def _hook(latents, timestep, n_tok, first_frame_cond):
    out = _orig(latents, timestep, n_tok, first_frame_cond)
    if latents.shape[2] == 1:                                   # 流式单 cell 前向
        SEEN.append((str(latents.dtype).replace("torch.", ""), float(out.float().reshape(-1)[0])))
    return out


def _cells(seen, n):
    """单 cell 前向序列 → 每个生成 cell 的 [(dtype, t) × 4 去噪 + 提交]（cell 0 只有一次提交）。"""
    assert seen[0][1] == 0.0, seen[0]
    rest = seen[1:]
    assert len(rest) == 5 * n, (len(rest), n)
    return [rest[5 * i: 5 * i + 5] for i in range(n)]


def main(out_path):
    AD._per_token_timestep_pf = _hook
    accel = Accelerator(step_scheduler_with_optimizer=False)
    S3._fsdp_gate(accel, (print if accel.is_main_process else (lambda *a: None)))
    dev = accel.device
    n, kv = 3, dict(kv_window=3, rope_cap=4, sink_size=1)
    student = DG.tiny_gcdit(causal=True, seed=0, dtype=torch.float32).train().requires_grad_(True)
    critic = DG.tiny_gcdit(causal=False, seed=5, dtype=torch.float32).train().requires_grad_(True)
    teacher = DG.tiny_gcdit(causal=False, seed=5, dtype=torch.bfloat16, device=dev).eval().requires_grad_(False)
    deploy_m = DG.tiny_gcdit(causal=True, seed=0, dtype=torch.bfloat16, device=dev).eval().requires_grad_(False)
    opt_g = torch.optim.AdamW(student.parameters(), lr=1e-6)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-6)
    student, critic, opt_g, opt_c = accel.prepare(student, critic, opt_g, opt_c)
    for m in (student, critic):
        AD.unwrap(m).move_rope_table(dev)

    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(1, 48, 1, 8, 10, generator=g).to(dev, torch.bfloat16)
    ctx = torch.randn(1, n + 1, 8, 4096, generator=g).to(dev, torch.bfloat16)
    ids = torch.zeros(1, n + 1, dtype=torch.int64, device=dev)
    sem = DG.syn_sem_ids(DG.frames_of(n + 1), 128, 160, "parts", g)[None].to(dev)
    deploy = deploy_schedule(DEPLOY_DENOISE_LIST)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)  # noqa: E731

    # 1. 训练 rollout（exit_step = 最后一档 ⇒ 每个 cell 走满 4 步）vs 部署孪生
    SEEN.clear()
    rec = {"exit": {}, "commit": {}}
    with ac():
        gen, gmask = S3.self_forcing_rollout(student, z0, ctx, ids, sem, n, deploy, n, score_last_k=n, record=rec,
                                             grad_start=1, exit_step=3, gen=torch.Generator(dev).manual_seed(1), **kv)
    train_seq = _cells(list(SEEN), n)
    SEEN.clear()
    ER.deploy_rollout(deploy_m, z0, ctx, ids, sem, n, **kv)
    deploy_seq = _cells(list(SEEN), n)

    # 2. critic ≡ teacher 的 DMD 目标；3. 目标 = rollout 输出的重放
    tail, cond_t, gm_t, t2f = S3.score_tail(gen, gmask, sem, ctx, ids, n, n)
    with ac():
        tgt, eval_idx, glog = S3.dmd_targets(teacher, critic, tail, cond_t, gm_t,
                                             gen=torch.Generator(dev).manual_seed(2))
    rec["n_eval"] = int(eval_idx.numel())
    f2t = {t2f[int(j)]: int(j) for j in eval_idx}
    with ac():
        loss = S3.stream_grad_replay(student, z0, ctx, ids, sem, rec, deploy,
                                     lambda k: (None if k not in f2t else tgt[:, :, f2t[k]:f2t[k] + 1]),
                                     accel.backward, **kv)
    gn = float(student.clip_grad_norm_(float("inf")))
    opt_g.zero_grad(set_to_none=True)

    # 4. 参数真的更新之后：rollout(record) → 重放（目标 = rollout + 噪声，有梯度）→ [释放根] → opt.step → 再测一次
    for g_ in opt_g.param_groups:
        g_["lr"] = 1e-3

    def _update_then_dev(free_root, seed):
        r1 = {"exit": {}, "commit": {}}
        with ac():
            g1, _ = S3.self_forcing_rollout(student, z0, ctx, ids, sem, n, deploy, n, score_last_k=n, record=r1,
                                            grad_start=1, exit_step=3, gen=torch.Generator(dev).manual_seed(seed), **kv)
        r1["n_eval"] = n
        noisy_t = g1.float() + 0.5 * torch.randn(g1.shape, device=dev, generator=torch.Generator(dev).manual_seed(seed))
        with ac():
            S3.stream_grad_replay(student, z0, ctx, ids, sem, r1, deploy, lambda k: noisy_t[:, :, k:k + 1],
                                  accel.backward, **kv)
        freed = S3._free_fsdp_root(student) if free_root else None
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        r2 = {"exit": {}, "commit": {}}
        with ac():
            g2, _ = S3.self_forcing_rollout(student, z0, ctx, ids, sem, n, deploy, n, score_last_k=n, record=r2,
                                            grad_start=1, exit_step=3, gen=torch.Generator(dev).manual_seed(seed + 1), **kv)
        r2["n_eval"] = n
        with ac():
            S3.stream_grad_replay(student, z0, ctx, ids, sem, r2, deploy, lambda k: g2[:, :, k:k + 1],
                                  accel.backward, **kv)
        opt_g.zero_grad(set_to_none=True)
        return r2["replay_dev"], freed

    dev_stale, _ = _update_then_dev(False, 10)
    dev_fixed, freed = _update_then_dev(True, 20)
    res = {"replay_dev_after_update_no_free": dev_stale, "replay_dev_after_update_free": dev_fixed, "root_freed": freed,"train_seq": train_seq, "deploy_seq": deploy_seq, "gen_dtype": str(gen.dtype).replace("torch.", ""),
           "dmd_gnorm": glog["dmd_gnorm"], "replay_loss": loss, "replay_dev": rec["replay_dev"],
           "student_gnorm": gn, "world": accel.num_processes,
           "cast_root_forward_inputs": accel.state.fsdp_plugin.mixed_precision_policy.cast_root_forward_inputs}
    if accel.is_main_process:
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res), flush=True)
    accel.wait_for_everyone()
    accel.end_training()


if __name__ == "__main__":
    main(sys.argv[1])
