#!/usr/bin/env python3
"""三处配方逐项相同：run_stage{1,2,3}.sh 的默认值 == {ar_tf_gc,cd_gc,dmd_gc}.py 的 argparse 默认值 == config_stage{1,2,3}.json。

上游 an earlier internal project 的 run.sh 与 argparse 有六处不一致，每一处都对应一次已知崩溃（PORT_设计 risk 11）；ARoPE 的
stage1 自己也有 grad_clip 1.0 / 10.0 的分叉。这里只允许存在一份配方：改一处不改另两处，在这里被拦。
（_check_three_places 移植自 ARoPE tests/test_stage{2_cd,3_dmd}.py，路径口径换成 maliang 的 $M。）
另外核对 钉死的数值（不只是三处彼此相等，而且等于 设计）。
"""
import json
import os
import re
import sys

M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(M, "code", "gc25d"), os.path.join(M, "code", "arope")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import pytest  # noqa: E402

from gcdistill import ar_tf_gc as S1  # noqa: E402
from gcdistill import cd_gc as S2  # noqa: E402
from gcdistill import dmd_gc as S3  # noqa: E402

GCD = os.path.join(M, "code", "gc25d", "gcdistill")


def _norm_path(x) -> str:
    t = str(x)
    for pre in (f"{M}/", "$M/", "${M}/"):
        if t.startswith(pre):
            return t[len(pre):]
    return t


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return _norm_path(a) == _norm_path(b)


def _check_three_places(stage: int, args: dict, alias: dict, skip: set, derived: set,
                        literals: set, flags: tuple) -> list:
    sh = open(os.path.join(GCD, f"run_stage{stage}.sh"), encoding="utf-8").read()
    cfg = json.load(open(os.path.join(GCD, f"config_stage{stage}.json"), encoding="utf-8"))
    env: dict = {}
    for name, val in re.findall(r'^\s*([A-Z_][A-Z0-9_]*)="\$\{\1:-([^}]*)\}"', sh, re.M):
        env.setdefault(name, []).append(val)
    checked, bad = [], []
    for name, vals in env.items():
        if name in skip:
            continue
        dest = alias.get(name, name.lower())
        if name in alias:
            assert dest in args, f"别名表把 {name} 映到了不存在的 argparse dest {dest!r}"
        if dest not in args or dest in derived:
            continue
        checked.append(dest)
        if not any(_same(v, args[dest]) for v in vals):
            bad.append(f"{dest}: run.sh={vals!r} argparse={args[dest]!r}")
    block = re.search(r"^ARGS=\((.*?)^\)", sh, re.M | re.S).group(1)
    lits = dict(re.findall(r'--([A-Za-z0-9_]+)\s+(?!["$-])(\S+)', block))
    assert literals <= set(lits), f"run.sh ARGS 里的硬编码字面量变了：{sorted(lits)}（应含 {sorted(literals)}）"
    for dest, v in lits.items():
        assert dest in args, f"run.sh ARGS 传了 --{dest}，argparse 里没有"
        checked.append(dest)
        if not _same(v, args[dest]):
            bad.append(f"{dest}: run.sh 字面量={v!r} argparse={args[dest]!r}")
    for flag in flags:
        assert re.search(rf"--{flag}(?=\s|$)", block, re.M), f"run.sh ARGS 里没有 --{flag}"
        checked.append(flag)
        if args[flag] is not True:
            bad.append(f"{flag}: run.sh 传了 --{flag}，argparse 默认却是 {args[flag]!r}")
    # config.json：凡是 argparse 有的标量键都要相等（不只是 run.sh 比到的那些）
    for dest, v in cfg.items():
        if dest.startswith("_") or dest in derived or dest in skip_cfg_keys or dest not in args:
            continue
        if isinstance(v, (dict, list)):
            continue
        if dest not in checked:
            checked.append(dest)
        if not _same(v, args[dest]):
            bad.append(f"{dest}: config.json={v!r} argparse={args[dest]!r}")
    # 每个 ARGS 里传了的 argparse dest（除路径外）config.json 都要记
    missing = sorted(d for d in set(checked) if d not in cfg and d not in derived and d not in ("output",))
    assert not missing, f"config_stage{stage}.json 缺这些配方键：{missing}"
    assert len(set(checked)) >= 20, f"只比到 {len(set(checked))} 个常量，正则大概率没匹配上 run.sh：{checked}"
    assert not bad, "三处配方不一致:\n  " + "\n  ".join(bad)
    return sorted(set(checked))


# 路径 / ckpt / 端口 / 续训开关不比。有意与 argparse 不同的只有两处（同上游）：RESUME=auto（argparse None）与
# stage1 的 --save_state（argparse 默认关：随手一跑不该写两个 ~60 GB 的 state 槽）
skip_cfg_keys = {"init_ckpt", "teacher_ckpt", "student_ckpt", "data_root", "text_table", "output", "resume", "save_state"}
_SKIP = {"SFT", "INIT", "AR", "STUDENT", "TEACHER", "STAGE1_OUT", "STAGE2_OUT", "OUT", "PORT", "ACC_CFG",
         "RESUME", "FORCE_RUN_DIR", "LIMIT", "TEXT_TABLE", "DATA_ROOT"}


def _args(mod, tmp_path):
    return vars(mod.parse_args(["--output", str(tmp_path / "_argparse_probe"), "--dry_run"]))


def _dry_fixups(a: dict) -> dict:
    """--dry_run 会把 vocab=auto 落成 parts、ckpt 清成 None；比配方时还原成 argparse 的原始默认值。"""
    a = dict(a)
    a["vocab"] = "auto"
    return a


def test_stage1_recipe_three_places(tmp_path):
    a = _dry_fixups(_args(S1, tmp_path))
    checked = _check_three_places(
        1, a, alias={"WINDOW": "windowed_train", "WINDOW_SINK": "windowed_sink", "KEEP_DITS": "keep_last_dits",
                     "TF_AUG": "tf_noise_aug_max_idx", "GRAD_ACCUM": "grad_accum"},
        skip=_SKIP, derived={"betas", "val_timesteps"}, literals={"batch_size"},
        flags=("teacher_forcing", "gradient_checkpointing"))
    print(f"stage1: {len(checked)} 个常量 {checked}")


def test_stage2_recipe_three_places(tmp_path):
    a = _dry_fixups(_args(S2, tmp_path))
    checked = _check_three_places(
        2, a, alias={"DISCRETE_N": "discrete_cd_N", "SOLVER_STEPS": "teacher_solver_steps",
                     "CFG_SCALE": "real_guidance_scale", "WINDOW": "windowed_train", "WINDOW_SINK": "windowed_sink",
                     "KEEP_EMAS": "keep_last_emas", "KEEP_DITS": "keep_last_dits", "GRAD_ACCUM": "grad_accum"},
        skip=_SKIP, derived={"betas", "denoise"}, literals={"batch_size"},
        flags=("teacher_forcing", "gradient_checkpointing"))
    print(f"stage2: {len(checked)} 个常量 {checked}")


def test_stage3_recipe_three_places(tmp_path):
    a = _args(S3, tmp_path)
    a = _dry_fixups(a)
    # --dry_run 会把几何缩成 tiny（_shrink_for_dry_run 只在 main 里调，parse_args 不缩）——这里拿的就是原始默认值
    checked = _check_three_places(
        3, a, alias={"SINK": "sink_size", "SCORE_FRAMES": "dmd_score_frames", "CLAMP": "dmd_timestep_clamp",
                     "CFG_SCALE": "real_guidance_scale", "KEEP_DITS": "keep_last_dits", "KEEP_EMAS": "keep_last_emas",
                     "KEEP_CRITICS": "keep_last_critics"},
        skip=_SKIP, derived={"betas", "denoise", "clamp"}, literals={"batch_size", "use_ema"},
        flags=("gradient_checkpointing",))
    print(f"stage3: {len(checked)} 个常量 {checked}")


@pytest.mark.parametrize("stage,mod,spec", [
    (1, S1, dict(lr=1e-5, adam_betas="0.9,0.999", weight_decay=0.01, warmup=200, grad_clip=10.0, max_steps=8000,
                 save_every=500, windowed_train=0, windowed_sink=0, text_dropout=0.1, cond_dropout=0.05, mask_p=0.5,
                 grad_accum=1, batch_size=1)),
    (2, S2, dict(lr=2e-6, adam_betas="0.0,0.999", weight_decay=0.0, warmup=100, grad_clip=10.0, max_steps=5000,
                 save_every=500, ema_decay=0.99, ema_start=200, teacher_solver_steps=1, real_guidance_scale=1.0,
                 cd_axis="deploy", denoise_list="1000,750,500,250", text_dropout=0.0, cond_dropout=0.0, mask_p=0.0,
                 grad_accum=1, batch_size=1)),
    (3, S3, dict(lr=2e-6, lr_critic=1e-6, adam_betas="0.0,0.999", weight_decay=0.01, warmup=100, grad_clip=10.0,
                 max_steps=500, save_every=50, critic_steps=5, ema_decay=0.99, ema_start=100, sink_size=3,
                 kv_window=17, rope_cap=20, dmd_score_frames=20, grad_horizon=20, rollout_min=21, rollout_max=27,
                 denoise_list="1000,750,500,250", real_guidance_scale=1.0, tail_mode="frankenstein")),
])
def test_recipe_matches_spec_section8(stage, mod, spec, tmp_path):
    a = _args(mod, tmp_path)
    bad = {k: (a[k], v) for k, v in spec.items() if not _same(a[k], v)}
    assert not bad, f"stage{stage} 的 argparse 默认值与 不符：{bad}"


def test_launchers_default_to_8_gpus_and_eff_batch_8():
    common = open(os.path.join(GCD, "common_gc.sh"), encoding="utf-8").read()
    assert 'DISTILL_NGPU="${DISTILL_NGPU:-8}"' in common
    for n in (1, 2, 3):
        sh = open(os.path.join(GCD, f"run_stage{n}.sh"), encoding="utf-8").read()
        assert "EFF_BATCH_OVERRIDE" in sh and "require_abs_out" in sh, f"run_stage{n}.sh 缺等效 batch 断言 / 绝对 OUT 检查"
        assert re.search(r'OUT="\$\{OUT:-\$M/outputs/distill/', sh), f"run_stage{n}.sh 的默认 OUT 不在 outputs/distill 下"
    for y in ("accelerate_zero2_8gpu_ga1.yaml", "accelerate_fsdp_dmd_8gpu.yaml",
              "accelerate_zero2_2gpu_ga1.yaml", "accelerate_fsdp_dmd_2gpu.yaml"):
        assert os.path.isfile(os.path.join(M, "configs", y)), y
    fsdp = open(os.path.join(M, "configs", "accelerate_fsdp_dmd_2gpu.yaml")).read()
    assert "fsdp_use_orig_params: true" in fsdp and "num_processes: 2" in fsdp


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
