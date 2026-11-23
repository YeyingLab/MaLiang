#!/usr/bin/env python3
"""stage1：teacher forcing 因果化 —— 把 gc25d 的双向 SFT 检查点训成因果 AR 扩散模型。
（ARoPE distill/stage1_ar/ar_tf.py 的 gc25d fork，结构逐段对应，改动点见 distill_port.md §8 表。）

    # 正式训练（配方常量钉在 run_stage1.sh 里，argparse 默认值与它、config_stage1.json 逐项相同）
    bash code/gc25d/gcdistill/run_stage1.sh

    # 30 秒 dry run（tiny GcDiT + 合成数据，不碰真权重；验循环 / 导出 / 日志 / 续训）
    CUDA_VISIBLE_DEVICES="" $PY code/gc25d/gcdistill/ar_tf_gc.py --dry_run --output <abs tmp>

这一阶段是什么
-------------
SFT（outputs/gc25d_v10/step-N）是**双向**全片模型。AR rollout 要因果模型，所以先「因果化」：同一份 flow-matching
目标，只把自注意力换成 teacher forcing 双流掩码 —— noisy cell f 只看**干净的** 0..f-1 加自己那一格。
产物 dits/dit_step{N}.safetensors 同时做 stage2 的 --teacher_ckpt 和 --student_ckpt。

相对 ARoPE ar_tf.py 的改动（distill_port §8）
-------------------------------------------
1. **条件**。没有 offset_px / known 掩码 / new 区权重图：语义图 sem_ids uint8 [b,81,H,W] 走 `mask_input`，
   GcDiT 把它在 TF 的**两个半流**上加同一份（clean cell j 与 noisy cell j 是同一组帧）。
2. **loss**。均匀权重的逐 cell MSE（`se[:,:,1:].mean(dim=(1,3,4))`），再乘逐 cell 的 bsmntw 钟形权重取平均；
   cell 0 = 干净 x0 + t=0、不计 loss（与 SFT 的 TI2V、stage2 / stage3 一路对齐）。
3. **dropout**。与 SFT 相同：文本 p=0.1 → NULL ×21、整段语义图 p=0.05 → 全 255、UNKNOWN 遮挡 p=0.5，
   都在 dataset.GcLatentDataset 的 worker 里做。ARoPE 的 action_dropout 删掉；anchor/hist/ctx 三档上下文
   dropout 与条件无关，保留（默认 0）。
4. **模型 / 导出**。GcDiT（mask_channel=False + gc_sem_cond，词表按 init ckpt 自动判断）+ gc_dit.save_gc（831 键）。
   metadata 必须带 teacher_forcing / windowed_train / windowed_sink / init_ckpt —— stage2 的配方闸读它们；
   res（GC_RES）由 save_gc 写，init / 热启动 ckpt 与数据根的 res 必须等于 GC_RES（data_gc.check_res）。
5. **grad_clip 三处都是 10.0**（ARoPE argparse / config.json 写的 1.0 是 run.sh 之外的第二套配方，RUNBOOK 说有效值是
   10.0：stage1 gnorm≈2.5，1.0 会步步都裁）。

两个必须守住的坑（同上游）
-------------------------
* `--tf_noise_aug_max_idx` 抽的是近干净端 `[1000-k, 1000)`，默认 0（精确干净）。
* `--grad_accum` 必须与 accelerate yaml 的 gradient_accumulation_steps 一致（SystemExit 守卫）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import warnings
import zlib

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
# 分配器要在 import torch 之前设：launch 时 export 的环境变量到不了 accelerate 拉起的子进程
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
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
from distill.flow_pf import (  # noqa: E402
    NUM_TRAIN_TIMESTEPS,
    add_noise_pf,
    idx_of_t,
    sample_pf_timestep_ids,
    t_of_idx,
    training_target,
    training_weight_pf,
)

CELLS = 21          # 一条 clip 的 latent cell 数（数据集硬契约）
STAGE = "stage1_ar_tf"


# ----------------------------------------------------------------------------
# 目标
# ----------------------------------------------------------------------------

def tf_clean_stream(x0: torch.Tensor, noise: torch.Tensor, max_idx: int):
    """TF 的 clean 半流。max_idx<=0 ⇒ (x0, None) 精确干净（CF 默认）。上游原样。

    max_idx>0 ⇒ history-noise 增广：近干净 = **高**下标（sigmas 递减），所以抽 [1000-max_idx, 1000)。
    cell 0 无论如何保持精确干净 + t=0（TI2V 锚点）。"""
    if max_idx is None or int(max_idx) <= 0:
        return x0, None
    k = int(max_idx)
    assert 0 < k < NUM_TRAIN_TIMESTEPS, f"--tf_noise_aug_max_idx 应在 (0, {NUM_TRAIN_TIMESTEPS}) 内，得到 {k}"
    b, f = x0.shape[0], x0.shape[2]
    idx_aug = torch.randint(NUM_TRAIN_TIMESTEPS - k, NUM_TRAIN_TIMESTEPS, (b, f), device=x0.device)
    clean_x = add_noise_pf(x0, noise, idx_aug)
    aug_t = t_of_idx(idx_aug, device=x0.device)
    clean_x[:, :, 0] = x0[:, :, 0]
    aug_t[:, 0] = 0.0
    return clean_x, aug_t


def ar_tf_loss(student, batch: dict, *, teacher_forcing: bool = True, tf_noise_aug_max_idx: int = 0,
               use_gradient_checkpointing: bool = False,
               hist_dropout: float = 0.0, ctx_dropout: float = 0.0, anchor_dropout: float = 0.0,
               noise: torch.Tensor | None = None, idx_pf: torch.Tensor | None = None,
               gen: torch.Generator | None = None):
    """一次 micro batch 的 loss。返回 (loss, stats)。

    teacher_forcing=True（默认）：干净历史走 clean_x 通道，[2S,2S] 双流掩码，S=21·tok/cell（832×480：21·390=8190；
    1280×704：21·880=18480）。
    语义图 batch["sem_ids"] uint8 [b, 4f−3, 16·lat_h, 16·lat_w] 走 mask_input（两个半流共用一份语义项）。
    三档上下文 dropout 与 ARoPE 同义（只改注意力窗口，cell 0 的 TI2V 锚点在输入侧始终保留），默认全 0。
    """
    x0 = batch["input_latents"]                                   # [b, 48, 21, H/16, W/16] bf16
    b, c, f, lat_h, lat_w = x0.shape                              # 几何从 latent 里取，dry_run 的小分辨率也能过
    dev = x0.device
    sem = batch["sem_ids"]                                        # [b, 81, H, W] uint8
    assert sem.dtype == torch.uint8 and tuple(sem.shape[:2]) == (b, DG.frames_of(f)), \
        f"sem_ids 应为 uint8 [b,{DG.frames_of(f)},H,W]，得到 {sem.dtype} {tuple(sem.shape)}"

    if noise is None:
        noise = torch.randn_like(x0)
    if idx_pf is None:
        idx_pf = sample_pf_timestep_ids(b, f, dev, gen)            # [b, 21] 严格均匀，逐 cell 独立
    noisy = add_noise_pf(x0, noise, idx_pf)                        # x_t=(1−σ)x0+σε
    ts_pf = t_of_idx(idx_pf, device=dev)                           # [b, 21] f32，(4.98, 1000]
    # TI2V 锚点：cell 0 用干净 x0 + t=0（模型侧 first_frame_cond=True 也会抹，这里显式写一遍让 stats 与输入一致）
    noisy[:, :, 0] = x0[:, :, 0]
    ts_pf[:, 0] = 0.0

    # ---- 上下文 dropout：逐 micro batch 抽签（batch_size 恒为 1，所以逐 batch == 逐样本）----
    def _coin(p):
        if p <= 0.0:
            return False
        return bool(torch.rand((), generator=gen, device="cpu").item() < p)

    # 三档互斥，按"砍得越狠越优先"排：ctx（全空）> hist（只剩 cell0）> anchor（只摘 cell0）
    drop_ctx = _coin(ctx_dropout)
    drop_hist = (not drop_ctx) and _coin(hist_dropout)
    drop_anchor = (not drop_ctx) and (not drop_hist) and _coin(anchor_dropout)
    clean_x, aug_t = (tf_clean_stream(x0, noise, tf_noise_aug_max_idx) if teacher_forcing else (None, None))

    bare = unwrap(student) if (drop_ctx or drop_hist or drop_anchor) else None
    saved = ((bare.train_local_attn_size, bare.train_sink_size, bare.train_drop_anchor)
             if bare is not None else None)
    if drop_ctx:
        bare.train_local_attn_size, bare.train_sink_size = 1, 0
    elif drop_hist:
        bare.train_local_attn_size, bare.train_sink_size = 1, 1
    elif drop_anchor:
        bare.train_drop_anchor = True
    try:
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            v = student(noisy, ts_pf, batch["context"], context_ids=batch["context_ids"],
                        mask_input=sem, first_frame_cond=True, clean_x=clean_x, aug_t=aug_t,
                        use_gradient_checkpointing=use_gradient_checkpointing)
    finally:
        if bare is not None:
            bare.train_local_attn_size, bare.train_sink_size, bare.train_drop_anchor = saved

    target = training_target(x0, noise)                            # ε − x0（rectified flow）
    se = (v.float() - target.float()) ** 2
    se = se[:, :, 1:]                                              # cell 0 是条件帧，不计 loss
    per_cell = se.mean(dim=(1, 3, 4))                              # [b, 20] 均匀权重（gc 没有 new 区权重图）
    tw = training_weight_pf(idx_pf[:, 1:]).float()                 # [b, 20] bsmntw 钟形
    loss = (tw * per_cell).mean()
    stats = {
        "mse": per_cell.mean().detach(),             # 不乘钟形权重的 MSE，跨配方可比
        "t_mean": ts_pf[:, 1:].mean().detach(),
        "tw_mean": tw.mean().detach(),
        "d_hist": torch.tensor(float(drop_hist), device=dev),
        "d_ctx": torch.tensor(float(drop_ctx), device=dev),
        "d_anc": torch.tensor(float(drop_anchor), device=dev),
    }
    return loss, stats


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="stage1：teacher forcing 因果化 gc25d SFT 检查点")
    # 默认值 = run_stage1.sh 的实参 = config_stage1.json（argparse 与 run.sh 不许有第二套配方）
    ap.add_argument("--init_ckpt", default=None, help="要因果化的双向 SFT 检查点（gc25d 831 键 bf16）")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--base_init", choices=("skeleton", "wan"), default="skeleton",
                    help="skeleton：不读 Wan 基础权重、直接从 831 键 ckpt 严格装满（gc_dit.empty_wan_dit，省 19 GB 读盘）；"
                         "wan：先读基础权重再装 ckpt（上游做法，结果逐位相同）")
    ap.add_argument("--output", required=True, help="run 目录（写绝对路径；相对路径会落进代码树）")

    ap.add_argument("--data_root", default=DG.DATA_ROOT)
    ap.add_argument("--text_table", default=None, help="None ⇒ <data_root>/text_table_gc25d.pt")
    ap.add_argument("--vocab", default="auto", choices=("auto", "parts", "parts_fx", "ids"),
                    help="auto ⇒ 按 init ckpt 的 gc_sem_cond 输入通道判断（数据侧按它 collapse）")
    ap.add_argument("--limit", type=int, default=None, help="smoke 用：只取这么多训练 clip")
    ap.add_argument("--synthetic", type=int, default=0,
                    help=">0 ⇒ 用这么多条全尺寸合成样本代替真数据（真权重冒烟在 prep 写完之前用；文本取自真文本表）")
    ap.add_argument("--text_dropout", type=float, default=0.1, help="SFT 同值：p → NULL_PROMPT ×21")
    ap.add_argument("--cond_dropout", type=float, default=0.05, help="SFT 同值：p → 语义图全 255（与文本独立）")
    ap.add_argument("--mask_p", type=float, default=0.5, help="SFT 同值：世界锚定的 UNKNOWN 遮挡")

    ap.add_argument("--teacher_forcing", action=argparse.BooleanOptionalAction, default=True,
                    help="默认开：干净历史走 clean_x 通道 + [2S,2S] 双流掩码。--no-teacher_forcing 回退成 "
                         "diffusion forcing（单流、便宜 2.5×，但 stage2 必须跟着一起关）")
    ap.add_argument("--tf_noise_aug_max_idx", type=int, default=0,
                    help="clean 半流的 history-noise 增广宽度（近干净档 = 高下标）。0 = 精确干净（CF 默认）")
    ap.add_argument("--anchor_dropout", type=float, default=0.0, help="只把 clean cell 0 从视野里摘掉（上游同义）")
    ap.add_argument("--hist_dropout", type=float, default=0.0, help="注意力上下文砍成只剩 cell 0（上游同义）")
    ap.add_argument("--ctx_dropout", type=float, default=0.0, help="注意力上下文砍成什么都看不到（上游同义）")
    ap.add_argument("--windowed_train", type=int, default=0,
                    help="训练掩码的滑动窗口（cell 数）。0 = 全因果 —— F=21 下 sink=0 的 W=20 会把 cell 0 挤出视野")
    ap.add_argument("--windowed_sink", type=int, default=0, help="[windowed_train>0 时] 恒可见的开头 cell 数")

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cond_lr", type=float, default=0.0,
                    help=">0 ⇒ dit.gc_sem_cond.* 单独一个参数组用这个 lr（SFT 的 cond_lr 口径）；0 = 与 --lr 相同")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--adam_betas", default="0.9,0.999", help="SFT 的优化器口径（stage1 = SFT 目标 + 因果掩码）")
    ap.add_argument("--warmup", "--warmup_steps", dest="warmup", type=int, default=200,
                    help="线性 warmup 步数，之后恒定 lr（每个 save_every 边界都是合法的停止点）")
    ap.add_argument("--grad_clip", type=float, default=10.0,
                    help="三处都是 10.0（RUNBOOK：stage1 gnorm≈2.5，1.0 会步步都裁）；DeepSpeed 下覆盖 yaml")
    ap.add_argument("--batch_size", type=int, default=1, help="每卡 micro batch；等效 batch = world × 它 × grad_accum")
    ap.add_argument("--grad_accum", type=int, default=1, help="必须与 accelerate yaml 的 gradient_accumulation_steps 相同"
                                                              "（等效 batch 8）")
    ap.add_argument("--max_steps", type=int, default=8000, help="优化器更新数（gc25d_v8：8000）")
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--save_every", type=int, default=500, help="0 ⇒ 只在训练结束时导出一次")
    ap.add_argument("--keep_last_dits", type=int, default=6, help="dits/ 只留最新几个（0 = 不删）")
    ap.add_argument("--save_state", action="store_true", help="同时 accelerator.save_state（state_A/state_B 轮换，可续训）")
    ap.add_argument("--val_every", type=int, default=500, help="0 ⇒ 不验证")
    ap.add_argument("--val_n", type=int, default=16)
    ap.add_argument("--val_timesteps", default="200,500,800")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume", default=None,
                    help="*.safetensors ⇒ 只热启动权重（step 归 0）；目录 ⇒ accelerator.load_state 续训；"
                         "auto ⇒ 有 <run>/state 就续，没有就从头")
    ap.add_argument("--dry_run", action="store_true", help="tiny 模型 + 合成数据，不碰真权重 / 真数据")
    ap.add_argument("--dry_run_full_hw", action="store_true",
                    help="[dry run] 合成数据用当前 GC_RES 的全尺寸 latent / 语义图网格（tiny 模型照旧），验分辨率链路")
    args = ap.parse_args(argv)

    if args.windowed_sink > 0 and args.windowed_train <= 0:
        raise SystemExit("--windowed_sink 只在 --windowed_train > 0 时有意义")
    try:
        b1, b2 = (float(x) for x in args.adam_betas.split(","))
    except ValueError:
        raise SystemExit(f"--adam_betas 应形如 '0.9,0.999'，得到 {args.adam_betas!r}") from None
    args.betas = (b1, b2)
    args.val_timesteps = [float(t) for t in args.val_timesteps.split(",") if t.strip()]
    if args.text_table is None:
        args.text_table = os.path.join(args.data_root, "text_table_gc25d.pt")
        if args.synthetic and not os.path.isfile(args.text_table):
            args.text_table = DG.REF_TEXT_TABLE          # 同一张 19 键表；prep 写完之前的冒烟用
    if args.dry_run:
        args.init_ckpt = None
        args.vocab = "parts" if args.vocab == "auto" else args.vocab
    else:
        if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
            raise SystemExit(f"--init_ckpt 不存在：{args.init_ckpt}")
        ck = GD.ckpt_vocab(args.init_ckpt)
        if ck is None:
            raise SystemExit(f"--init_ckpt {args.init_ckpt} 没有 gc_sem_cond 键：不是 gc25d SFT 导出")
        if args.vocab == "auto":
            args.vocab = ck
        elif args.vocab != ck:
            raise SystemExit(f"--vocab {args.vocab} 与 init ckpt 的词表 {ck} 不符")
        # 分辨率闸（加载 5B 之前）：init / 热启动 ckpt 的 metadata res、数据根的 res 都必须 = GC_RES
        warm = args.resume if args.resume and args.resume.endswith(".safetensors") else None
        DG.check_res(ckpts=[("--init_ckpt", args.init_ckpt), ("--resume", warm)],
                     root=None if args.synthetic else args.data_root)
    return args


# ----------------------------------------------------------------------------
# dry run：tiny 模型 + 合成数据
# ----------------------------------------------------------------------------

def _tiny_model(args) -> GD.GcDiT:
    return DG.tiny_gcdit(causal=True, seed=0, vocab=args.vocab,
                         train_local_attn_size=args.windowed_train, train_sink_size=args.windowed_sink)


def _datasets(args, is_main: bool):
    """(train_ds, val_ds, collate)。真数据走 dataset.GcLatentDataset（SFT 同一份，惰性 import）。"""
    if args.dry_run:
        c, f, h, w = DG.TINY_SHAPE
        h, w = DG.LAT_HW if args.dry_run_full_hw else (h, w)
        return DG.SynGcDataset(64, f, (h, w), vocab=args.vocab), None, DG.gc_collate_basic
    if args.synthetic:
        ds = DG.SynGcDataset(args.synthetic, CELLS, DG.LAT_HW, vocab=args.vocab, text_table=args.text_table,
                             seed=args.seed)
        val = (DG.SynGcDataset(min(args.val_n, 4), CELLS, DG.LAT_HW, vocab=args.vocab, text_table=ds.table,
                               seed=args.seed + 1) if args.val_every > 0 and args.val_n > 0 else None)
        return ds, val, DG.gc_collate_basic
    train_ds, collate = DG.latent_dataset(args.data_root, "train", args.text_table, vocab=args.vocab,
                                          text_dropout=args.text_dropout, cond_dropout=args.cond_dropout,
                                          mask_p=args.mask_p, limit=args.limit, seed=args.seed, verbose=is_main)
    val_ds = None
    if args.val_every > 0 and args.val_n > 0:
        # 验证不做任何 dropout / 遮挡（固定 clip × 固定噪声档 × 固定噪声，曲线只随模型变）
        val_ds, _ = DG.latent_dataset(args.data_root, "val", args.text_table, vocab=args.vocab,
                                      text_dropout=0.0, cond_dropout=0.0, mask_p=0.0, limit=args.val_n,
                                      seed=args.seed, verbose=is_main)
    return train_ds, val_ds, collate


def _build_student(args, log) -> tuple[GD.GcDiT, dict | None]:
    dit = GD.empty_wan_dit() if args.base_init == "skeleton" else None
    module = GD.GcDiT(dit=dit, causal=True, model_dir=args.model_dir, vocab=args.vocab,
                      train_local_attn_size=args.windowed_train, train_sink_size=args.windowed_sink)
    # 因果化的起点：双向 SFT 检查点。strict 831 键；assign=False ⇒ copy_ 进现有参数，保持模型自己的 dtype
    info = GD.load_gc_ckpt(module, args.init_ckpt)
    log(f"[model] 双向 SFT ← {args.init_ckpt}（{info['n_tensors']} 个张量，vocab={args.vocab}，base_init={args.base_init}）")
    return module, info


def param_groups(module, args) -> list[dict] | object:
    """--cond_lr>0 ⇒ gc_sem_cond 单独一组（SFT 的 cond_lr 口径）；否则一个组。"""
    if args.cond_lr <= 0 or abs(args.cond_lr - args.lr) < 1e-15:
        return module.parameters()
    pre = f"dit.{GD.GM.SEM_COND_NAME}."
    cond = [p for n, p in module.named_parameters() if n.startswith(pre)]
    rest = [p for n, p in module.named_parameters() if not n.startswith(pre)]
    assert cond, f"没有 {pre}* 参数"
    return [{"params": rest, "lr": args.lr}, {"params": cond, "lr": args.cond_lr}]


# ----------------------------------------------------------------------------
# 验证
# ----------------------------------------------------------------------------

@torch.no_grad()
def run_validation(accelerator: Accelerator, model, val_ds, collate, tids: list[int], args) -> dict:
    """固定 clip × 固定噪声档 × 固定噪声，各进程分摊 clip，最后 reduce（集合操作，人人都要到）。
    一次验证把**所有** cell 放在同一个噪声档 tid 上（cell 0 依旧干净 + t=0）。"""
    model.eval()
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    acc = torch.zeros(len(tids), 2, device=device, dtype=torch.float64)   # mse 和 / 计数
    for i in range(rank, len(val_ds), world):
        sample = val_ds[i]
        batch = to_device(collate([sample]), device)
        x0 = batch["input_latents"]
        for j, tid in enumerate(tids):
            g = torch.Generator(device=device)
            g.manual_seed((zlib.crc32(str(sample["clip_id"]).encode()) * 1009 + tid * 7 + args.seed) % (2 ** 31))
            noise = torch.randn(x0.shape, generator=g, device=device, dtype=x0.dtype)
            idx_pf = torch.full((x0.shape[0], x0.shape[2]), int(tid), device=device, dtype=torch.int64)
            _, st = ar_tf_loss(model, batch, teacher_forcing=args.teacher_forcing, tf_noise_aug_max_idx=0,
                               use_gradient_checkpointing=False, noise=noise, idx_pf=idx_pf)
            acc[j, 0] += st["mse"].double()
            acc[j, 1] += 1
    acc = accelerator.reduce(acc, reduction="sum")
    model.train()
    out = {}
    for j, tid in enumerate(tids):
        n = max(acc[j, 1].item(), 1.0)
        out[float(t_of_idx(int(tid)))] = {"mse": acc[j, 0].item() / n, "n": int(acc[j, 1].item())}
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    if args.dry_run and args.max_steps > 30:
        # 在打 banner 之前就改掉，否则日志里记的是一套没生效的配方
        args.max_steps, args.save_every, args.warmup, args.log_every, args.val_every = 12, 6, 3, 2, 0
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    is_main = accelerator.is_main_process
    use_ds = accelerator.distributed_type == DistributedType.DEEPSPEED
    set_seed(args.seed, device_specific=True)   # 每个 rank 的 ε / 噪声档 / dropout 抽签都不同
    device = accelerator.device

    if use_ds:
        # 让 --grad_clip 成为唯一入口；yaml 里的 gradient_clipping 只是默认值
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        ds_cfg["gradient_clipping"] = float(args.grad_clip)
        ga = ds_cfg.get("gradient_accumulation_steps")
        if ga not in ("auto", args.grad_accum):
            raise SystemExit(f"--grad_accum={args.grad_accum} 与 accelerate yaml 的 "
                             f"gradient_accumulation_steps={ga} 不一致：要么传 --grad_accum {ga}，要么换 configs/ 里 GA 匹配的 yaml")
        ds_cfg["gradient_accumulation_steps"] = args.grad_accum

    run_dir = runlib.resolve_run_dir(args.output)
    DG.check_res(run_dir=run_dir)                # 打开 train.log 之前：别往另一个分辨率的 run 里追加
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    log = runlib.TextTee(os.path.join(run_dir, "train.log"), enabled=is_main)
    jsonl = runlib.JsonlLogger(os.path.join(run_dir, "train_log.jsonl"), enabled=is_main)
    eff_batch = accelerator.num_processes * args.batch_size * args.grad_accum
    log(f"=== {STAGE} (gc25d) | run={run_dir} world={accelerator.num_processes} "
        f"dist={accelerator.distributed_type} mp={accelerator.mixed_precision} ===")
    if max(args.hist_dropout, args.ctx_dropout, args.anchor_dropout) > 0:
        log(f"[recipe] 上下文 dropout: anchor={args.anchor_dropout} hist={args.hist_dropout} ctx={args.ctx_dropout}")
    log(f"[recipe] res={GEO.RES_TAG}（{GEO.RES_ENV}）tf={args.teacher_forcing} tf_aug={args.tf_noise_aug_max_idx} "
        f"windowed={args.windowed_train}/{args.windowed_sink} vocab={args.vocab} "
        f"dropout text={args.text_dropout} cond={args.cond_dropout} mask_p={args.mask_p}（数据集做） "
        f"lr={args.lr} cond_lr={args.cond_lr or args.lr} betas={args.betas} wd={args.weight_decay} "
        f"warmup={args.warmup} clip={args.grad_clip} "
        f"bs={args.batch_size}×accum{args.grad_accum}×{accelerator.num_processes}卡 ⇒ 等效 batch {eff_batch} "
        f"max_steps={args.max_steps} gc={args.gradient_checkpointing}")
    low = runlib.warn_if_low_disk(run_dir)
    if low:
        log(f"[warn] {low}")

    # ---- 数据 ----
    train_ds, val_ds, collate = _datasets(args, is_main)
    log(f"[data] {'dry_run 合成' if args.dry_run else ('合成 ×%d' % args.synthetic if args.synthetic else args.data_root)}："
        f"train {len(train_ds)} clip，val {len(val_ds) if val_ds is not None else 0} clip")
    dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate,
        **DG.loader_kwargs(train_ds, args.num_workers),
    )

    # ---- 模型 ----
    t0 = time.time()
    if args.dry_run:
        module, resume_info = _tiny_model(args), None
    else:
        module, resume_info = _build_student(args, log)
    # 只热启动权重的续训必须在 prepare **之前**（DeepSpeed 在 initialize 时就建好了 fp32 主权重副本）
    warm_start = args.resume if (args.resume and args.resume.endswith(".safetensors")) else None
    if warm_start:
        info = (runlib.load_dit_ckpt(module, warm_start, allow_missing_mask=False) if args.dry_run
                else GD.load_gc_ckpt(module, warm_start))
        log(f"[resume] 只热启动权重 {warm_start}（step 从 0 起）：{info['n_tensors']} 个张量")
    n_params = sum(p.numel() for p in module.parameters())
    module.train().requires_grad_(True)
    log(f"[model] GcDiT causal={module.causal} vocab={GD.GM.sem_vocab(module.dit)} "
        f"{n_params:,} 参数，加载 {time.time() - t0:.0f}s")

    optimizer = torch.optim.AdamW(param_groups(module, args), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=args.betas)
    warm = max(int(args.warmup), 0)
    # lr scheduler 故意不进 prepare：accelerate 包装后每步会 step num_processes 次，warmup 被悄悄压缩
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warm) if warm > 0 else 1.0)

    # 只把 **一个** nn.Module 交给 prepare（DeepSpeed 分支只留最后一个，多传会静默丢）
    model, optimizer, dataloader = accelerator.prepare(module, optimizer, dataloader)
    # dit.freqs 是普通 tuple 属性（不是 buffer），prepare / .to() 都搬不动它 —— 必须在裸模块上搬一次
    unwrap(model).move_rope_table(device)
    if is_main and use_ds:
        clip_eff = getattr(getattr(model, "optimizer", None), "clip_grad", None)
        log(f"[env] grad_clip={args.grad_clip}（DeepSpeed 生效值 {clip_eff}）")

    # ---- 续训（accelerate 存档：权重 + 优化器 + 数据顺序，必须在 prepare 之后）----
    start_step = 0
    if args.resume and not warm_start:
        rp = args.resume
        if rp == "auto":
            link = os.path.join(run_dir, "state")
            rp = link if os.path.exists(link) else None
            log(f"[resume] auto ⇒ {rp or '没有存档，从头开始'}")
        if rp:
            with open(os.path.join(rp, "trainer_state.json"), encoding="utf-8") as fh:
                ts = json.load(fh)
            if ts.get("res", GEO.LEGACY_RES) != GEO.RES_TAG:
                raise SystemExit(f"ABORT: {rp} 是 {ts.get('res', GEO.LEGACY_RES)} 的存档，本次 {GEO.RES_ENV}={GEO.RES_TAG}")
            accelerator.load_state(rp)
            start_step = int(ts["step"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_step):
                    lr_scheduler.step()
            log(f"[resume] accelerator.load_state({rp})，从 step {start_step} 续训")

    # ---- config.json：复现所需的一切 ----
    if is_main:
        cfg = {
            "stage": STAGE, "args": vars(args), "eff_batch": eff_batch, "res": GEO.RES_TAG,
            "dataset": ({"synthetic": True} if args.dry_run else {
                "data_root": args.data_root, "synthetic": args.synthetic, "train_used": len(train_ds),
                "text_table": args.text_table, "vocab": args.vocab,
                "val_used": len(val_ds) if val_ds is not None else 0,
                "val_clip_ids": (val_ds.clip_ids() if hasattr(val_ds, "clip_ids") else [])}),
            "model": {"n_params": n_params, "causal": True, "cond": "gc_sem_cond", "vocab": args.vocab,
                      "init_ckpt": args.init_ckpt, "base_init": args.base_init,
                      "resume_info": ({k: v for k, v in resume_info.items() if k != "metadata"} if resume_info else None),
                      "start_step": start_step},
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
    csv_fields = ["step", "loss", "mse", "t_mean", "lr", "grad_norm", "step_time", "mem_gb", "epoch"]
    train_log = runlib.CsvLog(os.path.join(run_dir, "train_log.csv"), csv_fields, enabled=is_main)
    val_log = runlib.CsvLog(os.path.join(run_dir, "val_log.csv"), ["step", "timestep", "mse", "n"], enabled=is_main)
    val_tids = [int(idx_of_t(t)) for t in args.val_timesteps]

    def do_validation(step):
        if val_ds is None:
            return
        tv = time.time()
        res = run_validation(accelerator, model, val_ds, collate, val_tids, args)
        if is_main:
            for t, r in res.items():
                val_log.write({"step": step, "timestep": t, **r})
            mean = sum(r["mse"] for r in res.values()) / len(res)
            log("[val] step {}  ".format(step)
                + "  ".join(f"t={t:.0f}: mse={r['mse']:.4f} (n={r['n']})" for t, r in res.items())
                + f"  | mean {mean:.4f}  {time.time() - tv:.0f}s")

    def do_save(step, loss_val=None):
        # save_gc 是集合操作（accelerator.get_state_dict），每个 rank 都必须进来
        GD.save_gc(accelerator, model, runlib.dit_path(run_dir, step),
                   {"stage": STAGE, "step": step, "teacher_forcing": args.teacher_forcing,
                    "tf_noise_aug_max_idx": args.tf_noise_aug_max_idx,
                    "hist_dropout": args.hist_dropout, "ctx_dropout": args.ctx_dropout,
                    "anchor_dropout": args.anchor_dropout,
                    "windowed_train": args.windowed_train, "windowed_sink": args.windowed_sink,
                    "text_dropout": args.text_dropout, "cond_dropout": args.cond_dropout, "mask_p": args.mask_p,
                    "vocab": args.vocab, "data": "v10" if not (args.dry_run or args.synthetic) else "synthetic",
                    "init_ckpt": args.init_ckpt or ""},
                   dry_run=args.dry_run, log=log)
        if is_main:
            gone = runlib.rotate(runlib.dit_glob(run_dir), args.keep_last_dits)
            if gone:
                log(f"[rotate] 删掉 {len(gone)} 个旧 dit（保留最新 {args.keep_last_dits} 个）")
            runlib.write_progress(run_dir, step, stage=STAGE, loss=loss_val)
        if args.save_state:
            runlib.save_state_rotating(accelerator, run_dir, {"step": step, "stage": STAGE, "res": GEO.RES_TAG})

    # ---- 训练循环 ----
    model.train()
    step, epoch, last_saved = start_step, 0, start_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"[train] 从 step {start_step} 到 {args.max_steps}")
    micro_time, loss_ema = 0.0, None
    while step < args.max_steps:
        if hasattr(train_ds, "set_epoch"):          # GcLatentDataset 的 dropout 按 (seed, epoch, index) 抽
            train_ds.set_epoch(epoch)
        for batch in dataloader:
            if step >= args.max_steps:
                break
            t_step = time.time()
            batch = to_device(batch, device)
            lr_now = lr_scheduler.get_last_lr()[0]     # 本步用的 lr（step 之后拿到的是下一步的）
            with accelerator.accumulate(model):
                loss, stats = ar_tf_loss(
                    model, batch, teacher_forcing=args.teacher_forcing,
                    tf_noise_aug_max_idx=args.tf_noise_aug_max_idx,
                    hist_dropout=args.hist_dropout, ctx_dropout=args.ctx_dropout,
                    anchor_dropout=args.anchor_dropout,
                    use_gradient_checkpointing=args.gradient_checkpointing)
                accelerator.backward(loss)   # DeepSpeed 下这一步已含 engine.step()（裁剪 + 更新）
                grad_norm = (accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                             if accelerator.sync_gradients else None)
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()      # DeepSpeed 的 scheduler 包装不认累积边界，只能在这里守门
                optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            micro_time += time.time() - t_step
            if not accelerator.sync_gradients:
                continue                     # 还在累积，不算一步
            step += 1
            step_time, micro_time = micro_time, 0.0
            loss_mean = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            loss_ema = loss_mean if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_mean
            if is_main and (step % args.log_every == 0 or step == args.max_steps):
                gn = float(grad_norm) if grad_norm is not None and math.isfinite(float(grad_norm)) else float("nan")
                row = {"step": step, "loss": loss_mean, "mse": stats["mse"].item(),
                       "t_mean": stats["t_mean"].item(), "lr": lr_now, "grad_norm": gn,
                       "step_time": step_time,
                       "mem_gb": (torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0),
                       "epoch": epoch}
                train_log.write(row)
                jsonl.log({**row, "loss_ema": loss_ema})
                log(f"step {step}/{args.max_steps}  loss {loss_mean:.4f} (ema {loss_ema:.4f})  "
                    f"mse {row['mse']:.4f} t̄ {row['t_mean']:.0f}  "
                    f"lr {row['lr']:.2e}  gnorm {gn:.3f}  {step_time:.2f}s  mem {row['mem_gb']:.1f}GB")
            if args.val_every > 0 and step % args.val_every == 0:
                do_validation(step)
            if args.save_every > 0 and step % args.save_every == 0:
                do_save(step, loss_mean)
                last_saved = step
        epoch += 1

    if last_saved != step:
        do_save(step, loss_ema)
    if device.type == "cuda":
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 1024 ** 3, device=device)
        peak_max = accelerator.reduce(peak, reduction="max").item()
    else:
        peak_max = 0.0
    log(f"[done] step {step}，各进程峰值显存 {peak_max:.1f} GB。"
        f"下一步：把 dits/dit_step{step}.safetensors 同时作为 stage2 的 --teacher_ckpt 与 --student_ckpt")
    if is_main:
        runlib.write_progress(run_dir, step, stage=STAGE, finished=True, peak_mem_gb=peak_max)
    train_log.close()
    val_log.close()
    jsonl.close()
    log.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return step


if __name__ == "__main__":
    main()
