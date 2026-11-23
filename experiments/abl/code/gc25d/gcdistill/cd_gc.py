#!/usr/bin/env python3
"""stage2：因果一致性蒸馏（Causal Consistency Distillation，CF++ 的 P1 忠实路线）—— gc25d 版。
（ARoPE distill/stage2_cd/cd_arope.py 的 fork，结构逐段对应；改动点见 distill_port.md §8 表。）

    # 正式训练（配方常量钉在 run_stage2.sh 里，argparse 默认值与它、config_stage2.json 逐项相同）
    AR=<stage1 的 dits/dit_step8000.safetensors> bash code/gc25d/gcdistill/run_stage2.sh

    # 30 秒 dry run（tiny 三模型 + 合成数据，不碰真权重；验三模型装配 / loss / EMA / 导出 / 续训）
    CUDA_VISIBLE_DEVICES="" $PY code/gc25d/gcdistill/cd_gc.py --dry_run --output <abs tmp>

这一阶段是什么（同上游）
---------------------
stage1 产出的是**多步**因果 AR 扩散模型。CD 把它压成**少步**学生：学生在噪声档 t 处给出的 x0 估计，必须等于
EMA 影子在「教师走了一个 Euler 步」之后的 t_next 处给出的 x0 估计。轴 = 部署采样表 1000/750/500/250
（σ [1.0, 0.75, 0.5006, 0.2516]，3 对一致性）。

★ 产物是 **EMA 影子**：`ema/ema_step{N}.safetensors`（部署 + stage3 的 --student_ckpt）。dits/ 只为续训。

一步训练
-------
    idx        ← 一致性轴上抽一格（[0, K-2]），**每个 rank 抽到同一格**
    lat_t      = (1−σ_t)·x0 + σ_t·ε          cell 0 钉回干净 GT（TI2V 锚点）
    lat_tn     = lat_t + (σ_tn − σ_t)·教师(lat_t, t)      一个 Euler 步，cell 0 再钉一次
    cm_t       = lat_t  − σ_t ·学生(lat_t , t )           带梯度
    cm_tn      = lat_tn − σ_tn·EMA (lat_tn, t_n)          无梯度
    loss       = MSE(cm_t[:,:,1:], cm_tn[:,:,1:].detach())

相对 cd_arope.py 的改动（distill_port §8）
---------------------------------------
* 三次前向共用的 cond 换成 `dict(context, context_ids, mask_input=sem_ids, first_frame_cond=True, clean_x=x0, aug_t=None)`
  —— 没有 offset_px / known 掩码；语义图在 TF 的两个半流上加同一份。
* 三个模型都是 GcDiT（词表按 stage1 ckpt 自动判断），导出走 gc_dit.save_gc（831 键，dits/ + ema/，metadata 带 res）。
  分辨率闸：teacher / student（及续训的 dits/ema）ckpt 与数据根的 res 必须 = GC_RES（data_gc.check_res）。
* 数据：dataset.GcLatentDataset，文本 / 语义图 / 遮挡 dropout 全 0（CD 不做 CFG）。
* build_cd_axis / _keep_frame0 / _ckpt_meta_gate / _assert_ready / _assert_run_dir_safe 逐字照抄
  （cd_arope.py 没有 vendored，而且它 import 的 actionrope.dataset 在 maliang 里不存在）。
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import socket
import sys
import time
import warnings

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
# 分配器要在 import torch 之前设：launch 时 export 的环境变量到不了 accelerate 拉起的子进程
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GC_ROOT not in sys.path:
    sys.path.insert(0, GC_ROOT)

from gcdistill import data_gc as DG  # noqa: E402
from gcdistill import gc_dit as GD  # noqa: E402
import geometry as GEO  # noqa: E402
from distill import runlib  # noqa: E402
from distill.arope_dit import MODEL_DIR, unwrap  # noqa: E402
from distill.data import to_device  # noqa: E402
from distill.ema import EmaShadow  # noqa: E402
from distill.flow_pf import (  # noqa: E402
    DEPLOY_DENOISE_LIST,
    assert_deploy_covered,
    cd_axis,
    deploy_schedule,
    idx_of_t,
    sigma_of_idx,
    t_of_idx,
)
from text import NULL_PROMPT  # noqa: E402

CELLS = 21          # 一条 clip 的 latent cell 数（数据集硬契约）
STAGE = "stage2_cd"


# ----------------------------------------------------------------------------
# 一致性轴（cd_arope.build_cd_axis 逐字照抄）
# ----------------------------------------------------------------------------

def build_cd_axis(mode: str, denoise_list, discrete_N: int, tail_t: float, device=None, log=print):
    """返回 (sigmas[K], timesteps[K], info)。K−1 个一致性对：(0→1)、(1→2)…（下标越大越干净）。

    mode="deploy"（默认）：轴的节点**就是**部署采样表落到 1000 点训练网格上的那几个点。K=4 ⇒ 3 对，每步 Δσ≈−0.25。
    mode="discrete"：an earlier internal project 的密轴 `cd_axis(N)`（对照用，750 / 500 这两个部署读点不在轴上 —— 会打印警告）。
    tail_t > 0 ⇒ 在轴的最低点之下再挂一个节点（只能用于新 run；见上游 CD-1 说明）。默认 0 = 轴与部署表逐点相同。
    """
    dep_sig = deploy_schedule(denoise_list)[1]                      # [K_dep] 部署读点的 σ
    if mode == "deploy":
        idx, sig, tst = deploy_schedule(denoise_list)
        src = f"deploy_schedule({[float(x) for x in denoise_list]}) → grid idx {idx.tolist()}"
    elif mode == "discrete":
        sig, tst = cd_axis(int(discrete_N), check_deploy=False)
        src = f"cd_axis(N={discrete_N})"
    else:
        raise SystemExit(f"--cd_axis 只有 deploy / discrete，得到 {mode!r}")

    if tail_t and float(tail_t) > 0:
        # 追加节点只允许在**更干净**的一侧，否则会插到部署读点之间、改掉一致性对
        ti = idx_of_t(float(tail_t))
        s_tail, t_tail = sigma_of_idx(ti).reshape(1), t_of_idx(ti).reshape(1)
        if float(s_tail) >= float(sig[-1]) - 1e-9:
            raise SystemExit(f"--cd_tail_t {tail_t} 的 σ={float(s_tail):.5f} 不低于轴的最低点 "
                             f"σ={float(sig[-1]):.5f}；尾节点必须更干净（t 更小）")
        sig, tst = torch.cat([sig, s_tail]), torch.cat([tst, t_tail])
        src += f" + tail t={float(t_tail):.1f}"

    assert bool((sig[:-1] > sig[1:]).all()), f"一致性轴必须严格降序：{sig.tolist()}"
    assert sig.shape[0] >= 2, "一致性轴至少要 2 个节点才有一对"
    assert_deploy_covered(sig, denoise_list)      # 部署读点必须落在轴的 σ 区间内（flow_pf 的门）
    # 训练/推理对齐的硬闸门：部署每个读点都必须**恰好是**轴上的节点
    off_axis = [float(s) for s in dep_sig if float((sig - s).abs().min()) > 1e-6]
    if mode == "deploy":
        assert not off_axis, f"deploy 轴居然漏了部署读点 σ={off_axis}（deploy_schedule 与 build_cd_axis 不一致）"
    elif off_axis:
        log(f"[axis][warn] 部署读点 σ={[round(s, 5) for s in off_axis]} 不是 {src} 的节点 —— "
            f"这些档从没被一致性约束过就要在推理时读 x0。配方是 --cd_axis deploy。")
    info = {"mode": mode, "source": src, "K": int(sig.shape[0]), "pairs": int(sig.shape[0]) - 1,
            "sigmas": [round(float(s), 6) for s in sig], "timesteps": [round(float(t), 3) for t in tst],
            "dsigma": [round(float(sig[i + 1] - sig[i]), 6) for i in range(sig.shape[0] - 1)],
            "deploy_denoise_list": [float(x) for x in denoise_list],
            "deploy_sigmas": [round(float(s), 6) for s in dep_sig],
            "deploy_off_axis": [round(s, 6) for s in off_axis]}
    if device is not None:
        sig, tst = sig.to(device), tst.to(device)
    return sig, tst, info


# ----------------------------------------------------------------------------
# 目标
# ----------------------------------------------------------------------------

def _keep_frame0(x: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """cell 0 换成 clip 的 cell 0（out-of-place，clip 带梯度时也安全）。"""
    f0 = (torch.arange(x.shape[2], device=x.device) == 0).view(1, 1, -1, 1, 1)
    return torch.where(f0, clip, x)


def cd_loss(student, ema, teacher, batch: dict, *, sig_t, sig_tn, t_v: float, tn_v: float,
            teacher_forcing: bool = True, real_guidance_scale: float = 1.0,
            uncond=None, uncond_ids=None, solver_steps: int = 1,
            use_gradient_checkpointing: bool = False,
            noise: torch.Tensor | None = None, gen: torch.Generator | None = None):
    """一次 micro batch 的一致性 loss。返回 (loss, stats)。

    三次前向共用**同一份** cond（文本 / 语义图 / clean_x）：一致性步里唯一变的是噪声档和被加噪的 latent。
    """
    x0 = batch["input_latents"]                                    # [b, 48, 21, H/16, W/16] bf16
    b, f = x0.shape[0], x0.shape[2]
    dev = x0.device
    # σ 一律转成 **python float**（弱类型提升：`1−σ` 在 double 里算完再降位，比 bf16 张量相减精确）
    s_t, s_tn = float(sig_t), float(sig_tn)
    sem = batch["sem_ids"]                                         # [b, 81, H, W] uint8
    assert sem.dtype == torch.uint8 and tuple(sem.shape[:2]) == (b, DG.frames_of(f)), \
        f"sem_ids 应为 uint8 [b,{DG.frames_of(f)},H,W]，得到 {sem.dtype} {tuple(sem.shape)}"

    # ★ cond 在三次前向之间逐字段不变。first_frame_cond=True + **标量** t ⇒ cell 0 的 t=0、其余 =t
    cond = dict(context=batch["context"], context_ids=batch["context_ids"],
                mask_input=sem, first_frame_cond=True,
                clean_x=(x0 if teacher_forcing else None), aug_t=None)
    cond_u = dict(cond, context=uncond, context_ids=uncond_ids) if uncond is not None else None

    if noise is None:
        noise = torch.randn(x0.shape, device=dev, dtype=x0.dtype, generator=gen)
    lat_t = _keep_frame0((1.0 - s_t) * x0 + s_t * noise, x0)

    ac = torch.autocast(device_type=dev.type, dtype=torch.bfloat16)

    # ---- 教师：从 σ_t 走到 σ_tn（solver_steps 个 Euler 子步，默认 1 = 上游原样）；累加在 fp32 里做 ----
    with torch.no_grad(), ac:
        lat = lat_t.float()
        x0f = x0.float()
        for m in range(int(solver_steps)):
            a = s_t + (s_tn - s_t) * (m / solver_steps)
            z = s_tn if m == solver_steps - 1 else s_t + (s_tn - s_t) * ((m + 1) / solver_steps)
            tv = t_v if m == 0 else a * 1000.0
            v_c = teacher(lat.to(x0.dtype), tv, **cond)
            if cond_u is None:
                v_pred = v_c
            else:
                v_u = teacher(lat.to(x0.dtype), tv, **cond_u)
                v_pred = v_u + real_guidance_scale * (v_c - v_u)
            lat = _keep_frame0(lat + (z - a) * v_pred.float(), x0f)
        lat_tn = lat.to(x0.dtype)

    # ---- 学生：t 处的 x0 估计（带梯度）----
    with ac:
        v_s = student(lat_t, t_v, **cond, use_gradient_checkpointing=use_gradient_checkpointing)
    cm_t = lat_t.float() - s_t * v_s.float()

    # ---- EMA 影子：t_next 处的 x0 估计（无梯度，目标网络）----
    with torch.no_grad(), ac:
        v_e = ema(lat_tn, tn_v, **cond)
    cm_tn = lat_tn.float() - s_tn * v_e.float()

    # cell 0 是干净锚点，三条流里都被钉成 GT，算进 loss 只会稀释信号
    loss = F.mse_loss(cm_t[:, :, 1:], cm_tn[:, :, 1:].detach())
    with torch.no_grad():
        gt = x0.float()[:, :, 1:]
        stats = {
            "sigma_t": s_t, "sigma_tn": s_tn, "t": float(t_v), "tn": float(tn_v),
            "cm_err": ((cm_t[:, :, 1:] - gt) ** 2).mean().detach(),
            "tgt_err": ((cm_tn[:, :, 1:] - gt) ** 2).mean().detach(),
        }
    return loss, stats


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="stage2：因果一致性蒸馏（CD，gc25d）")
    # 默认值 = run_stage2.sh 的实参 = config_stage2.json
    ap.add_argument("--teacher_ckpt", default=None,
                    help="冻结教师。P1 = stage1 的 dits/dit_step{N}.safetensors（与 --student_ckpt 同一份）")
    ap.add_argument("--student_ckpt", default=None, help="学生 + EMA 的共同 init。P1 = 同一份 stage1 检查点")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--base_init", choices=("skeleton", "wan"), default="skeleton",
                    help="skeleton：不读 Wan 基础权重、从 831 键 ckpt 严格装满（结果与 wan 逐位相同）")
    ap.add_argument("--output", required=True, help="run 目录（写绝对路径）")

    ap.add_argument("--data_root", default=DG.DATA_ROOT)
    ap.add_argument("--text_table", default=None, help="None ⇒ <data_root>/text_table_gc25d.pt")
    ap.add_argument("--vocab", default="auto", choices=("auto", "parts", "parts_fx", "ids"))
    ap.add_argument("--limit", type=int, default=None, help="smoke 用：只取这么多训练 clip")
    ap.add_argument("--synthetic", type=int, default=0, help=">0 ⇒ 全尺寸合成样本（prep 写完之前的真权重冒烟）")
    ap.add_argument("--text_dropout", type=float, default=0.0, help="0：CD 不做 CFG，就不需要无条件样本")
    ap.add_argument("--cond_dropout", type=float, default=0.0)
    ap.add_argument("--mask_p", type=float, default=0.0)

    ap.add_argument("--teacher_causal", type=int, default=1,
                    help="1 = P1 忠实（因果教师 = stage1 产物，师生同轨迹）。0 = P2 务实（双向教师）")
    ap.add_argument("--teacher_forcing", action=argparse.BooleanOptionalAction, default=True,
                    help="干净 GT 走 clean_x 通道穿进**三次**前向。只有 stage1 也是 TF 训的才能开（读 metadata 核对）")
    ap.add_argument("--windowed_train", type=int, default=0, help="0 = 全因果")
    ap.add_argument("--windowed_sink", type=int, default=0, help="[windowed_train>0 时] 恒可见的开头 cell 数")

    ap.add_argument("--cd_axis", default="deploy", choices=["deploy", "discrete"],
                    help="deploy = 一致性轴的节点就是 --denoise_list 的部署读点；discrete = 密轴对照")
    ap.add_argument("--denoise_list", default=",".join(str(int(x)) for x in DEPLOY_DENOISE_LIST),
                    help="部署采样表（原始 t 值，从噪到净严格降序）。stage2 的轴和 stage3 都必须是这一串")
    ap.add_argument("--discrete_cd_N", type=int, default=48, help="[--cd_axis discrete] 密轴点数")
    ap.add_argument("--cd_tail_t", type=float, default=0.0, help="0 = 关（轴与部署表逐点相同）")
    ap.add_argument("--teacher_solver_steps", type=int, default=1, help="教师从 σ_t 走到 σ_tn 用几个 Euler 子步")
    ap.add_argument("--real_guidance_scale", type=float, default=1.0,
                    help="教师 CFG。1.0 ⇒ 跳过无条件前向（CF++ 铁律：CD / DMD 都是 1.0）")

    ap.add_argument("--ema_decay", type=float, default=0.99, help="目标网络的 EMA 衰减（半衰期≈69 步）")
    ap.add_argument("--ema_start", type=int, default=200, help="前这么多步硬拷贝 ema←student，之后才按 decay 走")

    ap.add_argument("--lr", type=float, default=2e-6, help="CF causal_cd 的 2e-6")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_betas", default="0.0,0.999",
                    help="CF 约定 beta1=0（一致性目标随 EMA 漂移，动量会追着旧目标跑）。别'顺手修'成 0.9")
    ap.add_argument("--warmup", type=int, default=100, help="线性 warmup 步数，之后恒定 lr")
    ap.add_argument("--grad_clip", type=float, default=10.0, help="CF max_grad_norm_generator=10.0")
    ap.add_argument("--batch_size", type=int, default=1, help="每卡 micro batch；等效 batch = world × 它 × grad_accum")
    ap.add_argument("--grad_accum", type=int, default=1, help="必须与 accelerate yaml 的 gradient_accumulation_steps 相同")
    ap.add_argument("--max_steps", type=int, default=5000, help="优化器更新数（gc25d_v8：5000）")
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--save_every", type=int, default=500, help="0 ⇒ 只在训练结束时导出一次")
    ap.add_argument("--keep_last_emas", type=int, default=25, help="ema/ 保留几个（stage3 的 init 从这些里挑）")
    ap.add_argument("--keep_last_dits", type=int, default=2, help="dits/ 只为续训而存，留 2 个够了")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume", default=None,
                    help="auto ⇒ 有 <run>/progress.json 就按它续（学生←dits/、影子←ema/，只热启动权重）；none / 不传 ⇒ 从头")
    ap.add_argument("--force_run_dir", action="store_true",
                    help="--output 目录里 train.log 刚被写过时仍然继续（默认拒：多半是另一个 run 还活着）")
    ap.add_argument("--dry_run", action="store_true", help="tiny 三模型 + 合成数据，不碰真权重 / 真数据")
    ap.add_argument("--dry_run_full_hw", action="store_true",
                    help="[dry run] 合成数据用当前 GC_RES 的全尺寸 latent / 语义图网格（tiny 模型照旧），验分辨率链路")
    args = ap.parse_args(argv)

    if args.windowed_sink > 0 and args.windowed_train <= 0:
        raise SystemExit("--windowed_sink 只在 --windowed_train > 0 时有意义")
    if args.teacher_forcing and not args.teacher_causal:
        raise SystemExit("--teacher_forcing 需要因果教师：clean_x 双流掩码只对 causal=True 有定义")
    if args.teacher_solver_steps < 1:
        raise SystemExit(f"--teacher_solver_steps 至少是 1，得到 {args.teacher_solver_steps}")
    try:
        b1, b2 = (float(x) for x in args.adam_betas.split(","))
    except ValueError:
        raise SystemExit(f"--adam_betas 应形如 '0.0,0.999'，得到 {args.adam_betas!r}") from None
    args.betas = (b1, b2)
    try:
        args.denoise = [float(x) for x in args.denoise_list.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--denoise_list 应形如 '1000,750,500,250'，得到 {args.denoise_list!r}") from None
    if len(args.denoise) < 2:
        raise SystemExit(f"--denoise_list 至少要两个点，得到 {args.denoise}")
    if args.text_table is None:
        args.text_table = os.path.join(args.data_root, "text_table_gc25d.pt")
        if (args.synthetic or args.dry_run) and not os.path.isfile(args.text_table):
            args.text_table = DG.REF_TEXT_TABLE
    if args.dry_run:
        args.teacher_ckpt = args.student_ckpt = None
        args.vocab = "parts" if args.vocab == "auto" else args.vocab
    else:
        for name in ("teacher_ckpt", "student_ckpt"):
            p = getattr(args, name)
            if not p:
                raise SystemExit(f"--{name} 是必需的（P1 路线：两者都是 stage1 的 dits/dit_step{{N}}.safetensors）")
            if not os.path.isfile(p):
                raise SystemExit(f"--{name} 不存在：{p}")
        vs, vt = GD.ckpt_vocab(args.student_ckpt), GD.ckpt_vocab(args.teacher_ckpt)
        if vs is None or vs != vt:
            raise SystemExit(f"student / teacher 的 gc_sem_cond 词表不一致或缺失：{vs} / {vt}")
        if args.vocab == "auto":
            args.vocab = vs
        elif args.vocab != vs:
            raise SystemExit(f"--vocab {args.vocab} 与 ckpt 的词表 {vs} 不符")
        # 分辨率闸（加载 3×5B 之前）：两份 ckpt 的 metadata res、数据根的 res 都必须 = GC_RES
        DG.check_res(ckpts=[("--teacher_ckpt", args.teacher_ckpt), ("--student_ckpt", args.student_ckpt)],
                     root=None if args.synthetic else args.data_root)
    return args


# ----------------------------------------------------------------------------
# dry run：tiny 三模型 + 合成数据
# ----------------------------------------------------------------------------

def _tiny_model(causal: bool = True, vocab: str = "parts") -> GD.GcDiT:
    """固定种子 ⇒ 每次调用得到**逐位相同**的权重（P1 要求 teacher/student/ema 同初值）。"""
    return DG.tiny_gcdit(causal=causal, seed=0, vocab=vocab)


def _gc_model(ckpt: str, causal: bool, args, win: dict, log) -> tuple[GD.GcDiT, dict]:
    DG.check_res(ckpts=[("ckpt", ckpt)])              # 续训的 dits/ 也过一遍（metadata res = GC_RES）
    dit = GD.empty_wan_dit() if args.base_init == "skeleton" else None
    m = GD.GcDiT(dit=dit, causal=causal, model_dir=args.model_dir, vocab=args.vocab, **win)
    return m, GD.load_gc_ckpt(m, ckpt)


# ----------------------------------------------------------------------------
# 三模型装配的守卫（上游逐字）
# ----------------------------------------------------------------------------

def _assert_ready(model, teacher, ema, device, log) -> None:
    """第一步之前把「没进 prepare 的两个模型」逐条验一遍（设备 / dtype / dit.freqs / 冻结 / 不是同一个对象）。"""
    bare_s = unwrap(model)
    for name, m, want_dtype in (("teacher", teacher, torch.bfloat16), ("ema", unwrap(ema.module), torch.float32)):
        p = next(m.parameters())
        assert p.device.type == device.type and (device.index is None or p.device.index == device.index), \
            f"{name} 的参数在 {p.device}，应该在 {device}（它没进 prepare，要手工 .to(device)）"
        assert p.dtype is want_dtype, f"{name} 的参数是 {p.dtype}，应该是 {want_dtype}"
        assert m.dit.freqs[0].device.type == device.type, \
            f"{name}.dit.freqs 还在 {m.dit.freqs[0].device}：它是普通 tuple 属性不是 buffer，"\
            f".to(device) 搬不动，必须 move_rope_table(device)"
        assert not any(q.requires_grad for q in m.parameters()), f"{name} 还有参数 requires_grad=True"
        assert not m.training, f"{name} 还在 train() 模式"
        assert m is not bare_s, f"{name} 与学生是同一个对象 —— 三模型退化成一个，loss 恒 0"
    assert unwrap(model).dit.freqs[0].device.type == device.type, "学生的 dit.freqs 没搬到卡上"
    n_ema, n_stu = len(list(ema.parameters())), len(list(model.parameters()))
    assert n_ema == n_stu, f"EMA 与学生的参数个数不同（{n_ema} vs {n_stu}）—— zip EMA 会静默错位"
    log(f"[guard] 三模型就位：student(ZeRO-2, bf16) / teacher(bf16 冻结, 未 prepare) / "
        f"ema(fp32 影子, 未 prepare)，各 {n_stu} 个参数张量，freqs 都在 {device}")


def _ckpt_meta_gate(path: str, args, log) -> dict:
    """读 stage1 检查点的 metadata（只读头），核对配方是否接得上：stage1 关了 TF 而 stage2 开着，
    会训出一个「历史从没干净过」的学生（不报错、loss 照降）。ar_tf_gc.py 的 save_gc 写了这几个键。"""
    try:
        _, meta = runlib.read_ckpt_header(path)
    except Exception as e:          # struct.error / json 解析失败都不该拦住训练
        log(f"[ckpt][warn] 读不出 {os.path.basename(path)} 的 metadata（{type(e).__name__}: {e}），跳过配方核对")
        return {}
    if not meta:
        return {}
    want = {"teacher_forcing": str(bool(args.teacher_forcing)),
            "windowed_train": str(int(args.windowed_train)),
            "windowed_sink": str(int(args.windowed_sink))}
    bad = {k: (meta[k], v) for k, v in want.items() if k in meta and str(meta[k]) != v}
    if bad:
        raise SystemExit(
            "stage1 检查点的配方与 stage2 的参数对不上：" +
            "；".join(f"{k} 检查点={a} 本次={b}" for k, (a, b) in bad.items()) +
            f"\n  检查点：{path}\n  CD 要求师生同轨迹 —— 把 stage2 的对应参数改成检查点里的值。")
    log(f"[ckpt] {os.path.basename(path)} metadata: " +
        " ".join(f"{k}={meta[k]}" for k in ("stage", "step", "teacher_forcing", "windowed_train", "vocab") if k in meta))
    return meta


def _assert_run_dir_safe(run_dir: str, stage: str, force: bool, log=print, dry_run: bool = False) -> None:
    """开跑之前确认 `--output` 指的不是**别人的** run 目录（保存点会 rotate 掉那边的权重）。
    1. progress.json 里记着别的 stage ⇒ 直接拒；2. train.log 在最近 10 分钟里被写过 ⇒ 拒（--force_run_dir 放行）。"""
    prog = runlib.read_progress(run_dir)
    if prog and prog.get("stage") and str(prog["stage"]) != stage:
        raise SystemExit(
            f"ABORT: {run_dir} 是 stage={prog['stage']!r} 的 run 目录（progress.json 记在 step "
            f"{prog.get('step')}），本次是 {stage}。保存时会 rotate 掉那边的 dits/ema —— 换个 --output。")
    log_path = os.path.join(run_dir, "train.log")
    if os.path.isfile(log_path):
        age = time.time() - os.path.getmtime(log_path)
        if age < 600 and not (force or dry_run):
            raise SystemExit(
                f"ABORT: {log_path} 在 {age / 60:.1f} 分钟前还被写过 —— 多半有另一个 run 正活在这个目录里。"
                "换个 --output；如果确认上一次已经死了，加 --force_run_dir（launcher 里是 FORCE_RUN_DIR=1）。")
        if age < 600:
            log(f"[warn] {log_path} {age / 60:.1f} 分钟前刚被写过，--force_run_dir 强行继续。")


def _datasets(args, verbose: bool = False):
    if args.dry_run:
        c, f, h, w = DG.TINY_SHAPE
        h, w = DG.LAT_HW if args.dry_run_full_hw else (h, w)
        return DG.SynGcDataset(64, f, (h, w), vocab=args.vocab), DG.gc_collate_basic
    if args.synthetic:
        return (DG.SynGcDataset(args.synthetic, CELLS, DG.LAT_HW, vocab=args.vocab, text_table=args.text_table,
                                seed=args.seed), DG.gc_collate_basic)
    return DG.latent_dataset(args.data_root, "train", args.text_table, vocab=args.vocab,
                             text_dropout=args.text_dropout, cond_dropout=args.cond_dropout,
                             mask_p=args.mask_p, limit=args.limit, seed=args.seed, verbose=verbose)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    if args.dry_run and args.max_steps > 30:
        # 在打 banner 之前就改掉，否则日志里记的是一套没生效的配方
        args.max_steps, args.save_every, args.warmup, args.log_every, args.ema_start = 12, 6, 3, 2, 3
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    is_main = accelerator.is_main_process
    use_ds = accelerator.distributed_type == DistributedType.DEEPSPEED
    set_seed(args.seed, device_specific=True)   # 每个 rank 的 ε 抽签都不同；一致性 idx 另走同步的 CPU 流
    device = accelerator.device

    if use_ds:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        ds_cfg["gradient_clipping"] = float(args.grad_clip)
        ga = ds_cfg.get("gradient_accumulation_steps")
        if ga not in ("auto", args.grad_accum):
            raise SystemExit(f"--grad_accum={args.grad_accum} 与 accelerate yaml 的 "
                             f"gradient_accumulation_steps={ga} 不一致（等效 batch = num_processes × batch_size × GA，配方 8）")
        ds_cfg["gradient_accumulation_steps"] = args.grad_accum

    run_dir = runlib.resolve_run_dir(args.output)
    # ★ 在**任何**写之前查：--output 指的是不是别人的 run 目录。所有 rank 都在这里 stat，之后再 barrier
    _assert_run_dir_safe(run_dir, STAGE, bool(args.force_run_dir), dry_run=bool(args.dry_run))
    DG.check_res(run_dir=run_dir)
    accelerator.wait_for_everyone()
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    log = runlib.TextTee(os.path.join(run_dir, "train.log"), enabled=is_main)
    jsonl = runlib.JsonlLogger(os.path.join(run_dir, "train_log.jsonl"), enabled=is_main)
    eff_batch = accelerator.num_processes * args.batch_size * args.grad_accum
    log(f"=== {STAGE} (gc25d) | run={run_dir} world={accelerator.num_processes} "
        f"dist={accelerator.distributed_type} mp={accelerator.mixed_precision} ===")
    log(f"[recipe] res={GEO.RES_TAG}（{GEO.RES_ENV}）teacher_causal={args.teacher_causal} tf={args.teacher_forcing} "
        f"windowed={args.windowed_train}/{args.windowed_sink} cfg={args.real_guidance_scale} vocab={args.vocab} "
        f"ema={args.ema_decay}@{args.ema_start} lr={args.lr} betas={args.betas} wd={args.weight_decay} "
        f"warmup={args.warmup} clip={args.grad_clip} "
        f"bs={args.batch_size}×accum{args.grad_accum}×{accelerator.num_processes}卡 ⇒ 等效 batch {eff_batch} "
        f"max_steps={args.max_steps} gc={args.gradient_checkpointing}")
    low = runlib.warn_if_low_disk(run_dir)
    if low:
        log(f"[warn] {low}")

    # ---- 一致性轴（必须在建模型之前定下来：续训要拿它的指纹做守卫）----
    sig, tst, axis_info = build_cd_axis(args.cd_axis, args.denoise, args.discrete_cd_N,
                                        args.cd_tail_t, device=None, log=log)
    K = int(sig.shape[0])
    log(f"[axis] {axis_info['source']}  K={K}（{K - 1} 对一致性）")
    log(f"[axis] sigma = {axis_info['sigmas']}")
    log(f"[axis] t     = {axis_info['timesteps']}")
    log(f"[axis] Δsigma= {axis_info['dsigma']}  教师 Euler 子步 = {args.teacher_solver_steps}")
    log(f"[axis] 部署采样表 {axis_info['deploy_denoise_list']} → sigma {axis_info['deploy_sigmas']}"
        f"{'（逐点都是轴上的节点）' if not axis_info['deploy_off_axis'] else '（有读点不在轴上！）'}")

    # ---- 数据 ----
    train_ds, collate = _datasets(args, verbose=is_main)
    log(f"[data] {'dry_run 合成' if args.dry_run else ('合成 ×%d' % args.synthetic if args.synthetic else args.data_root)}"
        f"：train {len(train_ds)} clip（dropout text={args.text_dropout} cond={args.cond_dropout} mask_p={args.mask_p}）")
    dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate,
        **DG.loader_kwargs(train_ds, args.num_workers),
    )

    # ---- 续训（只热启动权重：学生 ← dits/、影子 ← ema/；优化器状态不保存）----
    resume_step, resume_micro = 0, 0
    student_init, ema_init = args.student_ckpt, None
    if args.resume and args.resume != "none":
        prog = runlib.read_progress(run_dir)
        if not prog or int(prog.get("step", 0)) <= 0:
            log(f"[resume] {run_dir} 里没有 progress.json，从头开始")
        else:
            n = int(prog["step"])
            dp, ep = runlib.dit_path(run_dir, n), runlib.ema_path(run_dir, n)
            miss = [p for p in (dp, ep) if not os.path.isfile(p)]
            if miss:
                raise SystemExit(f"progress.json 说停在 step {n}，但 {miss} 不存在。")
            old_ga = prog.get("grad_accum")
            if old_ga is not None and int(old_ga) != args.grad_accum:
                raise SystemExit(f"续训不许改 --grad_accum（存档 {old_ga} → 本次 {args.grad_accum}）："
                                 f"一致性 idx 每个 micro batch 抽一次，改了累积数这条流就在各 rank 之间错开了")
            old_axis = prog.get("axis_sigmas")
            if old_axis is not None and [round(float(s), 6) for s in old_axis] != axis_info["sigmas"]:
                raise SystemExit(f"续训不许换一致性轴（存档 {old_axis} → 本次 {axis_info['sigmas']}）")
            resume_step, student_init, ema_init = n, dp, ep
            resume_micro = int(prog.get("micro", n * args.grad_accum))
            log(f"[resume] step {resume_step}：student←{os.path.basename(dp)} ema←{os.path.basename(ep)}"
                f"（micro batch 计数 {resume_micro}）")

    # ---- 模型：三个 GcDiT，全部从同一份 stage1 权重起步 ----
    t0 = time.time()
    win = dict(train_local_attn_size=args.windowed_train, train_sink_size=args.windowed_sink)
    if args.dry_run:
        module = _tiny_model(causal=True, vocab=args.vocab)
        teacher = _tiny_model(causal=bool(args.teacher_causal), vocab=args.vocab)
        load_info = None
        if student_init:                       # dry_run 也走一遍续训的装载路径
            runlib.load_dit_ckpt(module, student_init, allow_missing_mask=False)
            log(f"[model] student ← {student_init}")
    else:
        _ckpt_meta_gate(student_init, args, log)
        if args.teacher_ckpt != student_init:      # P1 下两者同一份，只在真的分叉时再验一次
            _ckpt_meta_gate(args.teacher_ckpt, args, log)
        module, info_s = _gc_model(student_init, True, args, win, log)
        log(f"[model] student ← {student_init}（{info_s['n_tensors']} 个张量）")
        teacher, info_t = _gc_model(args.teacher_ckpt, bool(args.teacher_causal), args, win, log)
        log(f"[model] teacher ← {args.teacher_ckpt}（causal={bool(args.teacher_causal)}，{info_t['n_tensors']} 个张量）")
        load_info = {"student": info_s["path"], "teacher": info_t["path"]}
    teacher.to(torch.bfloat16).requires_grad_(False).eval()

    # EMA 影子：deepcopy 学生 —— step 0 时 ema 与 student 逐位相同（CF++ P1 的前提）
    ema = EmaShadow(module, device=None)          # 先在 CPU 上转 fp32，等 prepare 之后再上卡
    if ema_init is not None:
        if not args.dry_run:
            DG.check_res(ckpts=[("ema", ema_init)])
        info_e = runlib.load_dit_ckpt(ema, ema_init, allow_missing_mask=False)   # copy_ 进 fp32 参数
        ema._assert_fp32()
        log(f"[model] ema ← {ema_init}（{info_e['n_tensors']} 个张量）")
    n_params = sum(p.numel() for p in module.parameters())
    module.train().requires_grad_(True)
    log(f"[model] 三份 GcDiT 各 {n_params:,} 参数（vocab={args.vocab}，base_init={args.base_init}）；"
        f"构建 {time.time() - t0:.0f}s")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    warm = max(int(args.warmup), 0)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warm) if warm > 0 else 1.0)
    if resume_step:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(resume_step):
                lr_scheduler.step()

    # ★ 只把 **一个** nn.Module 交给 prepare（DeepSpeed 只留最后一个）。teacher / ema 都在这行之外。
    model, optimizer, dataloader = accelerator.prepare(module, optimizer, dataloader)
    unwrap(model).move_rope_table(device)         # dit.freqs 是普通 tuple 属性，prepare / .to() 都搬不动
    teacher.to(device).move_rope_table(device)
    ema.to(device)
    unwrap(ema.module).move_rope_table(device)
    _assert_ready(model, teacher, ema, device, log)
    if is_main and use_ds:
        clip_eff = getattr(getattr(model, "optimizer", None), "clip_grad", None)
        log(f"[env] grad_clip={args.grad_clip}（DeepSpeed 生效值 {clip_eff}）")

    # ---- CFG 的负条件（cfg==1 时根本不建）----
    uncond = None
    if abs(args.real_guidance_scale - 1.0) >= 1e-6:
        if args.dry_run:
            null_row = torch.randn(8, 4096, dtype=torch.bfloat16)   # tiny 的 L=8
        else:
            table = DG.load_text_table(args.text_table)
            if NULL_PROMPT not in table:
                raise SystemExit(f"--real_guidance_scale={args.real_guidance_scale} 需要负条件，但文本表里没有 "
                                 f"NULL_PROMPT={NULL_PROMPT!r}（{args.text_table}）")
            null_row = table[NULL_PROMPT].to(torch.bfloat16)
        uncond = null_row.view(1, 1, *null_row.shape).to(device)
        log(f"[cfg] real_guidance_scale={args.real_guidance_scale}，负条件 = NULL_PROMPT {tuple(null_row.shape)}")

    # ---- config.json：复现所需的一切 ----
    if is_main:
        cfg = {
            "stage": STAGE, "args": vars(args), "eff_batch": eff_batch, "axis": axis_info, "res": GEO.RES_TAG,
            "dataset": ({"synthetic": True} if args.dry_run else {
                "data_root": args.data_root, "synthetic": args.synthetic, "train_used": len(train_ds),
                "text_table": args.text_table, "vocab": args.vocab}),
            "model": {"n_params": n_params, "causal_student": True, "causal_teacher": bool(args.teacher_causal),
                      "cond": "gc_sem_cond", "vocab": args.vocab, "student_ckpt": student_init,
                      "teacher_ckpt": args.teacher_ckpt, "ema_ckpt": ema_init, "load_info": load_info,
                      "start_step": resume_step, "base_init": args.base_init},
            "product": {"deploy": "ema/ema_step{N}.safetensors",
                        "resume_only": "dits/dit_step{N}.safetensors",
                        "note": "stage3 的 --student_ckpt 从 ema/ 里挑，绝不要用 dits/"},
            "env": {"world_size": accelerator.num_processes, "distributed_type": str(accelerator.distributed_type),
                    "mixed_precision": accelerator.mixed_precision, "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "python": sys.executable, "platform": platform.platform(), "hostname": socket.gethostname(),
                    "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "deepspeed_config": (accelerator.state.deepspeed_plugin.deepspeed_config if use_ds else None),
        }
        try:
            import accelerate  # noqa: F401
            cfg["env"]["accelerate"] = accelerate.__version__
        except Exception:
            pass
        runlib.save_config(run_dir, cfg)

    # ---- 日志 ----
    csv_fields = ["step", "cd", "cm_err", "tgt_err", "idx", "sigma_t", "sigma_tn", "ema_decay",
                  "lr", "grad_norm", "step_time", "mem_gb", "epoch"]
    train_log = runlib.CsvLog(os.path.join(run_dir, "train_log.csv"), csv_fields, enabled=is_main)

    def do_save(step, loss_val=None, micro=0):
        # save_gc 是集合操作（accelerator.get_state_dict），每个 rank 都必须进来
        meta = {"stage": STAGE, "step": step, "teacher_forcing": args.teacher_forcing,
                "windowed_train": args.windowed_train, "windowed_sink": args.windowed_sink,
                "cd_axis": args.cd_axis, "denoise_list": args.denoise_list,
                "ema_decay": args.ema_decay, "student_ckpt": student_init or "", "teacher_ckpt": args.teacher_ckpt or "",
                "vocab": args.vocab, "data": "v10" if not (args.dry_run or args.synthetic) else "synthetic"}
        GD.save_gc(accelerator, model, runlib.dit_path(run_dir, step),
                   {**meta, "role": "raw_student_resume_only"}, dry_run=args.dry_run, log=log)
        # ★ 部署权重 / stage3 的 init：EMA。上一行的 dits/ 只为续训存在。
        GD.save_gc(accelerator, ema, runlib.ema_path(run_dir, step),
                   {**meta, "role": "ema_deploy"}, dry_run=args.dry_run, log=log)
        if is_main:
            gone_d = runlib.rotate(runlib.dit_glob(run_dir), args.keep_last_dits)
            gone_e = runlib.rotate(runlib.ema_glob(run_dir), args.keep_last_emas)
            if gone_d or gone_e:
                log(f"[rotate] 删掉 {len(gone_d)} 个旧 dit（留 {args.keep_last_dits}）、"
                    f"{len(gone_e)} 个旧 ema（留 {args.keep_last_emas}）")
            runlib.write_progress(run_dir, step, stage=STAGE, loss=loss_val, micro=micro,
                                  grad_accum=args.grad_accum, axis_sigmas=axis_info["sigmas"],
                                  deploy="ema/ema_step%d.safetensors" % step)

    # ---- 一致性 idx 的随机流：CPU 生成器，**所有 rank 同一个种子** ⇒ 每个 micro batch 抽到同一格 ----
    idx_rng = torch.Generator(device="cpu").manual_seed(1234 + args.seed)
    for _ in range(resume_micro):                 # 按 micro batch 数快进
        torch.randint(0, K - 1, (1,), generator=idx_rng)

    # ---- 训练循环 ----
    model.train()
    step, epoch, micro, last_saved = resume_step, 0, resume_micro, resume_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"[train] 从 step {step} 到 {args.max_steps}")
    micro_time, loss_ema = 0.0, None
    hits = [0] * (K - 1)
    while step < args.max_steps:
        if hasattr(train_ds, "set_epoch"):          # GcLatentDataset 的 dropout 按 (seed, epoch, index) 抽
            train_ds.set_epoch(epoch)
        for batch in dataloader:
            if step >= args.max_steps:
                break
            t_step = time.time()
            batch = to_device(batch, device)
            lr_now = lr_scheduler.get_last_lr()[0]
            with accelerator.accumulate(model):
                # 每个 micro batch 抽一次，所有 rank 同步（否则平均的是几条不同轨迹上的梯度）
                idx = int(torch.randint(0, K - 1, (1,), generator=idx_rng).item())
                micro += 1
                hits[idx] += 1
                b = batch["input_latents"].shape[0]
                loss, stats = cd_loss(
                    model, ema, teacher, batch,
                    sig_t=float(sig[idx]), sig_tn=float(sig[idx + 1]),
                    t_v=float(tst[idx]), tn_v=float(tst[idx + 1]),
                    teacher_forcing=args.teacher_forcing, real_guidance_scale=args.real_guidance_scale,
                    uncond=(None if uncond is None else uncond.expand(b, batch["context"].shape[1], -1, -1)),
                    uncond_ids=(None if uncond is None else torch.zeros_like(batch["context_ids"])),
                    solver_steps=args.teacher_solver_steps,
                    use_gradient_checkpointing=args.gradient_checkpointing)
                accelerator.backward(loss)   # DeepSpeed 下这一步已含 engine.step()（裁剪 + 更新）
                grad_norm = (accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                             if accelerator.sync_gradients else None)
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            micro_time += time.time() - t_step
            if not accelerator.sync_gradients:
                continue
            step += 1
            # ★ EMA 必须在优化器更新**之后**、每个优化器步只做一次；step<=ema_start 期间硬拷贝
            decay = 0.0 if step <= args.ema_start else args.ema_decay
            ema.update(model, decay)
            step_time, micro_time = micro_time, 0.0
            loss_mean = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            loss_ema = loss_mean if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_mean
            if is_main and (step % args.log_every == 0 or step == args.max_steps):
                gn = float(grad_norm) if grad_norm is not None and math.isfinite(float(grad_norm)) else float("nan")
                row = {"step": step, "cd": loss_mean, "cm_err": stats["cm_err"].item(),
                       "tgt_err": stats["tgt_err"].item(), "idx": idx,
                       "sigma_t": stats["sigma_t"], "sigma_tn": stats["sigma_tn"], "ema_decay": decay,
                       "lr": lr_now, "grad_norm": gn, "step_time": step_time,
                       "mem_gb": (torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0),
                       "epoch": epoch}
                train_log.write(row)
                jsonl.log({**row, "cd_ema": loss_ema, "micro": micro, "axis_hits": hits})
                log(f"step {step}/{args.max_steps}  cd {loss_mean:.5f} (ema {loss_ema:.5f})  "
                    f"σ {row['sigma_t']:.4f}→{row['sigma_tn']:.4f} [idx {idx}]  "
                    f"cm_err {row['cm_err']:.4f} tgt {row['tgt_err']:.4f}  "
                    f"lr {lr_now:.2e}  gnorm {gn:.3f}  decay {decay:.2f}  "
                    f"{step_time:.2f}s  mem {row['mem_gb']:.1f}GB")
            if args.save_every > 0 and step % args.save_every == 0:
                do_save(step, loss_mean, micro)
                last_saved = step
        epoch += 1

    if last_saved != step:
        do_save(step, loss_ema, micro)
    if device.type == "cuda":
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 1024 ** 3, device=device)
        peak_max = accelerator.reduce(peak, reduction="max").item()
    else:
        peak_max = 0.0
    log(f"[done] step {step}，一致性轴各格命中 {hits}，各进程峰值显存 {peak_max:.1f} GB。"
        f"下一步：从 ema/ 里挑一个 ema_step{{N}}.safetensors 做 stage3 的 --student_ckpt（不要用 dits/）")
    if is_main:
        runlib.write_progress(run_dir, step, stage=STAGE, finished=True, peak_mem_gb=peak_max,
                              micro=micro, grad_accum=args.grad_accum, axis_sigmas=axis_info["sigmas"],
                              deploy="ema/ema_step%d.safetensors" % step)
    train_log.close()
    jsonl.close()
    log.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return step


if __name__ == "__main__":
    main()
