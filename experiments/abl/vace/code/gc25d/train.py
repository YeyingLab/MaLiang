#!/usr/bin/env python3
"""gc25d 双向 SFT：Wan2.2-TI2V-5B 全参数微调 + 零初始化 gc_sem_cond（逐像素 parts id 条件），latent 与文本都预计算。

ARoPE `actionrope/train.py` 的 fork（结构逐段对应，便于 diff）：AropeTrainModule → GcTrainModule（model.gc_forward），
arope / 动作臂 / CNS / 画布 / 权重图全部去掉，换成 sem 条件；数据集换成 dataset.GcLatentDataset（三种 dropout 在 worker 里抽）。

    # 单进程（smoke，不用 DeepSpeed；AdamW 直接更新 bf16 参数、无 fp32 主权重，数字不代表训练效果）
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python code/gc25d/train.py --limit 8 --max_steps 4 --output outputs/_smoke/x

    # 多卡：accelerate + DeepSpeed ZeRO-2（启动器 scripts/train_sft.sh 选 / 生成 configs/accelerate_zero2*.yaml）
    OUT=outputs/sft_v10 bash scripts/train_sft.sh

与 ARoPE 的差别（都在 GcTrainModule / main 里）
--------------------------------------------------
* 前向 = model.gc_forward（plain RoPE、逐 cell 文本、patch embedding 后加 gc_sem_cond(sem_ids)），aux 带回 sem_rms / x_rms
  （sem 项与 patch embedding 的 RMS；proj_out 零初始化 ⇒ step 1 的 sem_rms 恰为 0，之后应单调长起来）。
* loss = MSE(帧 1..20) × training_weight(t)（权重图恒为 1，所以 ARoPE 的 loss_w 就是 mse，不再单列）。
* 两个 AdamW 参数组：`dit.gc_sem_cond.*` 用 --cond_lr（默认 1e-4 = 10×；零初始化的输入通道在 lr 1e-5 下学得极慢，
  train_recipe §6），其余 --lr；wd 相同。LambdaLR 的 warmup 系数同时乘到两组上。
* 验证：固定 val clip × 固定 t × 固定噪声，每个 (clip, t) 跑三遍：cond（真 ids + 真文本）/ uncond（ids 全 255）/
  nulltext（NULL_PROMPT × 21）；记 mse 与两个 gap（uncond − cond、nulltext − cond）—— gap 是条件有没有被用上的直接读数。
* 导出 = gcdistill.gc_dit.save_gc（831 个 bf16 键 = 825 Wan + 6 gc_sem_cond，键不带 dit.，metadata 带 vocab 等），
  写完用 check_gc_ckpt 体检；save_state 走 distill.runlib.save_state_rotating（state_A / state_B 两槽 + state 软链）。
* 热启动 --resume X.safetensors = model.load_gc_weights：gc25d 导出（词表必须相同）或普通 Wan / plain 导出
  （缺全部 gc_sem_cond.* ⇒ 保持零初始化）。--resume <dir> = accelerator.load_state 续训，**数据位置精确接上**：
  trainer_state.json 记 (epoch, 本 epoch 已消费的 micro batch 数, 几何)，续训用 skip_first_batches 跳过已消费的部分
  （ARoPE 从 epoch 开头重放）；导出 metadata 的 init_ckpt 沿用首次启动的热启动权重，续训另记 resumed_from / resumed_step。
* 分辨率 = geometry（env GC_RES，默认 832x480；scripts/train_sft.sh 透传）：--data_root 的 res 必须等于它（dataset 构造时查），
  导出 metadata / trainer_state 记 res；dir 续训的存档 res 必须相同；safetensors 热启动的 res 不同只打印提示（跨分辨率热启动合法）。
* 数据顺序用 accelerate 的 SeedableRandomSampler（data_seed = --seed）：每个 epoch 的排列 = f(seed, epoch)，各 rank 同一排列再分片；
  不用它时排列来自 sampler generator 的累积状态，load_state 不存（只给 IterableDataset 存），续训没法接上。
* step_time 是整步墙钟（含等数据），data_time 单列等 DataLoader 的部分（epoch 开头重新 fork worker 也在这里）。

学习率调度不交给 accelerate.prepare：accelerate 包装后的 scheduler 每步会 step num_processes 次
（它假设 scheduler 是按样本数建的），warmup 会被悄悄压缩；这里手动 step，一步就是一步。

随机数：set_seed(seed, device_specific=True) 让每个 rank 的 torch / CUDA RNG 都不同 —— 噪声 ε（randn_like）各 rank 独立；
t 由各 rank 的 cpu_gen 抽；dropout 抽签按 (seed, epoch, index) 在 dataset 里定（与 rank / worker 无关）。数据分片不受影响：
SeedableRandomSampler 用显式 data_seed，不吃 device_specific 的种子。不加 device_specific 时 8 张卡每步抽到同一份 ε。
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

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedType, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                    # maliang 根目录
AROPE_ROOT = os.environ.get("AROPE_ROOT", os.path.join(ROOT, "code", "arope"))
for _p in (HERE, AROPE_ROOT):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.append(_p)
import geometry as GEO  # noqa: E402
import model as GM  # noqa: E402
from dataset import CELLS, UNKNOWN_ID, GcLatentDataset, gc_collate  # noqa: E402
from text import NULL_PROMPT  # noqa: E402
from gcdistill.gc_dit import GC_DIT_KEYS, GC_META, check_gc_ckpt, cond_spec, save_gc  # noqa: E402
from distill.runlib import CsvLog, save_state_rotating  # noqa: E402
from diffsynth.diffusion.flow_match import FlowMatchScheduler  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models/Wan2.2-TI2V-5B")
DATA_ROOT = os.path.join(ROOT, "data/gc25d_v10")
DIT_FILES = [os.path.join(MODEL_DIR, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]
# 导出的 safetensors 只含 DiT（含 gc_sem_cond），键名与原模型一致：包装 module 里 DiT 挂在 .dit 下，导出时把这个前缀去掉
DIT_PREFIX = "dit."
COND_PREFIX = cond_spec("ids")["prefix"]                          # "dit.gc_sem_cond."
VAL_VARIANTS = ("cond", "uncond", "nulltext")


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="gc25d 双向 SFT")
    ap.add_argument("--vocab", choices=tuple(GM.VOCABS), default="parts",
                    help="条件词表：parts（v10 默认，K=20）/ ids（旧 gc25d，K=10，parts 经 collapse_parts 折叠）")
    ap.add_argument("--cond_type", choices=("ids",), default="ids", help="条件类型（gc25d v10 只有语义 id；记进 metadata）")
    ap.add_argument("--cond_mode", choices=("ids", "color_light", "color_vae", "color_vace"), default="ids",
                    help="paper 接口消融：ids=③ 整数直接编码；color_light=② 上色代理视频+轻量编码器；color_vae=① 上色代理视频+Wan VAE")
    ap.add_argument("--cond_key", choices=("ids",), default="ids", help="cond/*.npz 读哪个键（记进 metadata）")
    ap.add_argument("--text_dropout", type=float, default=0.1, help="逐样本：21 个 cell 全换成 NULL_PROMPT")
    ap.add_argument("--cond_dropout", type=float, default=0.05, help="逐样本：ids 全 255（与 text dropout 独立）")
    ap.add_argument("--mask_p", type=float, default=0.5, help="逐样本：世界锚定的 UNKNOWN 遮挡（cond 未 drop 时）")
    ap.add_argument("--include_dropped", action="store_true", help="训练集保留 clips.jsonl 里 drop=true 的 clip（默认剔除）")
    ap.add_argument("--first_frame_cond", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--data_root", default=DATA_ROOT, help="latents/ cond/ meta/ index/clips.jsonl text_table_gc25d.pt")
    ap.add_argument("--text_table", default=None, help="默认 <data_root>/text_table_gc25d.pt")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--limit", type=int, default=None, help="smoke 用：只取这么多训练 clip")

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cond_lr", type=float, default=1e-4, help=f"{COND_PREFIX}* 参数组的 lr")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500, help="线性 warmup 步数（0 ⇒ 常数 lr），之后常数")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="梯度裁剪阈值；DeepSpeed 下写进 deepspeed_config.gradient_clipping（覆盖 yaml），"
                         "日志里的 grad_norm 是裁剪前的全局范数")
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=1, help="每卡 micro batch（一个 micro batch 共用一个 t）")
    ap.add_argument("--grad_accum", type=int, default=2,
                    help="梯度累积步数；step 计的是优化器更新数。DeepSpeed 下必须与 accelerate yaml 的 "
                         "gradient_accumulation_steps 一致（configs/accelerate_zero2_ga2.yaml 是 2）")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--save_state", action=argparse.BooleanOptionalAction, default=True,
                    help="每次导出权重时同时 accelerator.save_state（含优化器分片）；"
                         "state_A / state_B 两个槽轮流写，<output>/state 是指向最新完整槽的软链")
    ap.add_argument("--val_every", type=int, default=500, help="0 ⇒ 不验证")
    ap.add_argument("--val_n", type=int, default=16)
    ap.add_argument("--val_timesteps", default="200,500,800")
    ap.add_argument("--val_uncond", action=argparse.BooleanOptionalAction, default=True,
                    help="验证时每个 (clip, t) 再跑 uncond（ids 全 255）与 nulltext（NULL_PROMPT×21）两遍")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--output", required=True, help="outputs/<run>")
    ap.add_argument("--resume", default=None,
                    help="*.safetensors ⇒ 只热启动权重（step 从 0 起；普通 Wan 导出 ⇒ gc_sem_cond 零初始化）；"
                         "目录 ⇒ accelerator.load_state（续训）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.text_table is None:
        args.text_table = os.path.join(args.data_root, "text_table_gc25d.pt")
    args.val_timesteps = [float(t) for t in args.val_timesteps.split(",") if t.strip()]
    if not os.path.isabs(args.output):
        args.output = os.path.join(ROOT, args.output)
    # 权重血统（导出 metadata）：init_ckpt = 热启动的 safetensors（dir 续训时由 main 从存档里接回来），续训另记
    args.init_ckpt = args.resume if args.resume and args.resume.endswith(".safetensors") else ""
    args.resumed_from, args.resumed_step = "", 0
    return args


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------

def load_dit(model_dir: str, device: str = "cpu"):
    """只加载 DiT（不挂 T5 / VAE），bf16。先落在 CPU，交给 accelerator.prepare 搬上卡。（= model.load_dit）"""
    return GM.load_dit(model_dir, device)


def load_weights_into_dit(dit: nn.Module, path: str, vocab: str) -> dict:
    """从导出的 safetensors 热启动（model.load_gc_weights：严格键集，只放行"全部 gc_sem_cond.* 缺失且模块仍零初始化"）。
    ckpt 带 gc_sem_cond 时词表必须与 --vocab 相同。"""
    info = GM.load_gc_weights(dit, path)
    if info["vocab"] != vocab:
        raise RuntimeError(f"{path} 的 gc_sem_cond 词表是 {info['vocab']!r}，本次训练 --vocab {vocab!r}")
    return {k: v for k, v in info.items() if k != "metadata"} | {"ckpt_metadata": info.get("metadata", {})}


def sem_param_groups(module: nn.Module, lr: float, cond_lr: float, weight_decay: float = 0.01,
                     prefix: str = COND_PREFIX) -> list[dict]:
    """可训参数按名字前缀分两组：[其余（lr）, prefix*（cond_lr）]；prefix 组为空时只返回一组。wd 两组相同。
    prefix 默认 "dit.gc_sem_cond."（GcTrainModule / gcdistill.GcDiT 都把 WanModel 挂在 .dit 下）。"""
    rest, cond = [], []
    for name, p in module.named_parameters():
        if p.requires_grad:
            (cond if name.startswith(prefix) else rest).append(p)
    groups = [{"params": rest, "lr": lr, "weight_decay": weight_decay}]
    if cond:
        groups.append({"params": cond, "lr": cond_lr, "weight_decay": weight_decay})
    return groups


def warmup_lambda(warmup_steps: int):
    """LambdaLR 系数：min(1, (s+1)/warm) 线性 warmup 后常数（warm=0 ⇒ 恒 1）。"""
    warm = max(int(warmup_steps), 0)
    return lambda s: min(1.0, (s + 1) / warm) if warm > 0 else 1.0


def proxy_palette(seed: int = 0) -> torch.Tensor:
    """Fixed id -> RGB colour table (uint8 [256,3]); 255 (UNKNOWN) -> white, no other id is pure white."""
    g = torch.Generator().manual_seed(seed)
    pal = torch.randint(16, 240, (256, 3), generator=g, dtype=torch.int64).to(torch.uint8)
    pal[UNKNOWN_ID] = 255
    return pal


ABL_COND_NAME = {"ids": GM.SEM_COND_NAME, "color_light": GM.WB_COND_NAME, "color_vae": "gc_vae_cond", "color_vace": "gc_vace"}


class GcTrainModule(nn.Module):
    """把 DiT + 扩散 loss 包成一个 module，forward 直接返回 loss —— DeepSpeed 引擎要求前向在 module 内。
    gc_sem_cond 挂在 dit 上（install_sem_cond），是这个 module 的子模块 ⇒ 进 ZeRO 分片、拿得到梯度。"""

    def __init__(self, dit, first_frame_cond: bool, gradient_checkpointing: bool, cond_mode: str = "ids", vae=None):
        super().__init__()
        self.dit = dit
        self.cond_mode = cond_mode
        self.register_buffer("pal", proxy_palette(), persistent=False)
        self.__dict__["_vae"] = vae          # frozen Wan VAE (not a submodule: no grads, not in optimizer / ZeRO)
        self.first_frame_cond = first_frame_cond
        self.gradient_checkpointing = gradient_checkpointing
        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(1000, training=True)

    def timestep_id_for(self, t: float) -> int:
        return int(torch.argmin((self.scheduler.timesteps - t).abs()))

    def forward(self, batch: dict, timestep_id: int, noise: torch.Tensor | None = None, train: bool = True):
        x0 = batch["input_latents"]  # [b, 48, 21, H/16, W/16] bf16（832×480：30×52）
        timestep = self.scheduler.timesteps[timestep_id].view(1)  # float32 cpu，与 DiffSynth 的抽法一致
        if noise is None:
            noise = torch.randn_like(x0)
        # x_t = (1−σ) x0 + σ ε；目标 ε − x0
        x_t = self.scheduler.add_noise(x0, noise, timestep)
        target = self.scheduler.training_target(x0, noise, timestep)
        # TI2V 首帧条件：帧 0 用干净 x0 替换，模型侧对帧 0 的 t 置 0（first_frame_cond）
        x_t[:, :, 0:1] = x0[:, :, 0:1]

        aux = {}
        sem, ctype, extra, vace_z = batch["sem_ids"], None, None, None
        if self.cond_mode == "color_light":
            sem, ctype = self.pal.to(sem.device)[sem.long()], "whitebox"           # uint8 [b,T,H,W,3]
        elif self.cond_mode in ("color_vae", "color_vace"):
            rgb = self.pal.to(sem.device)[sem.long()]
            with torch.no_grad():
                xv = rgb.permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1.0)      # [b,3,T,H,W] in [-1,1]
                zc = torch.stack([self._vae.single_encode(xv[i:i + 1], x0.device).squeeze(0) for i in range(xv.shape[0])])
            if self.cond_mode == "color_vae":
                extra = self.dit.gc_vae_cond(zc.to(x0.dtype))
            else:
                vace_z = zc.to(x0.dtype)
            sem = None
        pred = GM.gc_forward(
            self.dit, x_t, timestep.to(device=x0.device, dtype=x0.dtype), batch["context"], sem,
            first_frame_cond=self.first_frame_cond, context_ids=batch["context_ids"],
            use_gradient_checkpointing=self.gradient_checkpointing and train, aux=aux, cond_type=ctype, extra=extra,
            vace_z=vace_z,
        )
        # 帧 0 是条件帧，不计 loss
        pred, target = pred[:, :, 1:], target[:, :, 1:]
        se = (pred.float() - target.float()) ** 2
        mse = se.mean()
        loss = mse * self.scheduler.training_weight(timestep).to(mse.device)
        stats = {
            "mse": mse.detach(),
            "timestep": float(timestep.item()),
            "sem_rms": aux["sem_rms"].detach(),
            "x_rms": aux["x_rms"].detach(),
        }
        return loss, stats


# ----------------------------------------------------------------------------
# 导出 / 日志
# ----------------------------------------------------------------------------

def export_metadata(args, step: int, extra: dict | None = None) -> dict:
    """step-N.safetensors 的 metadata（save_gc 先填 GC_META，再盖这些；全部转成字符串）。"""
    meta = dict(GC_META)
    meta.update({
        "vocab": args.vocab, "data": "v10", "data_root": args.data_root, "step": step, "res": GEO.RES_TAG,
        "cond_type": args.cond_type, "cond_key": args.cond_key, "stage": "sft_bidirectional", "cond_mode": args.cond_mode,
        "text_dropout": args.text_dropout, "cond_dropout": args.cond_dropout, "mask_p": args.mask_p,
        "lr": args.lr, "cond_lr": args.cond_lr, "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps,
        "first_frame_cond": args.first_frame_cond, "seed": args.seed,
        # init_ckpt = 权重从哪来（热启动 safetensors，"" = 底模 + 零初始化 gc_sem_cond），续训不改它；续训点单列
        "init_ckpt": args.init_ckpt, "resumed_from": args.resumed_from, "resumed_step": args.resumed_step,
        "model_dir": args.model_dir, "trainer": "gc25d/train.py",
    })
    meta.update(extra or {})
    return {k: str(v) for k, v in meta.items()}


def export_dit_weights(accelerator: Accelerator, model, path: str, metadata: dict) -> int:
    mode = metadata.get("cond_mode", "ids")
    if mode != "ids":
        from safetensors.torch import save_file
        sd = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            out = {k[len(DIT_PREFIX):]: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in sd.items()
                   if k.startswith(DIT_PREFIX)}
            tmp = path + ".tmp"
            save_file(out, tmp, metadata={k: str(v) for k, v in metadata.items()})
            os.replace(tmp, path)
            print(f"[save] {path} ({len(out)} tensors, cond_mode={mode})", flush=True)
            return os.path.getsize(path)
        return 0
    return _export_dit_weights_ids(accelerator, model, path, metadata)


def _export_dit_weights_ids(accelerator: Accelerator, model, path: str, metadata: dict) -> int:
    """取完整权重（ZeRO-2 下参数本就不分片），主进程去掉 'dit.' 前缀写 safetensors（gcdistill.gc_dit.save_gc：
    断言 831 个 bf16 键、.tmp + rename、chmod 644），写完 check_gc_ckpt 体检。返回文件字节数（主进程）。"""
    size = save_gc(accelerator, model, path, metadata)
    if accelerator.is_main_process:
        res = check_gc_ckpt(path)
        if res["metadata"].get("vocab") != metadata["vocab"]:
            raise RuntimeError(f"{path}: metadata vocab {res['metadata'].get('vocab')!r} ≠ {metadata['vocab']!r}")
        print(f"[save] check_gc_ckpt OK：{res['n_tensors']} 个 {res['dtypes']} 张量，vocab={res['metadata']['vocab']}", flush=True)
    return size


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def rotate_csv_if_header_changed(path: str, fields: list[str]) -> str | None:
    """续训进同一个 OUT、而 CSV 列变了（代码升级）⇒ 旧文件改名 <name>.old<N>.csv，免得 CsvLog 往旧表头下追加错位的行。"""
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return None
    with open(path, encoding="utf-8") as fh:
        if fh.readline().rstrip("\r\n").split(",") == list(fields):
            return None
    stem, ext = os.path.splitext(path)
    n = 0
    while os.path.exists(f"{stem}.old{n}{ext}"):
        n += 1
    os.replace(path, f"{stem}.old{n}{ext}")
    return f"{stem}.old{n}{ext}"


# ----------------------------------------------------------------------------
# 续训：数据位置与权重血统
# ----------------------------------------------------------------------------

def epoch_steps(n_micro: int, grad_accum: int) -> int:
    """每 epoch 的优化器步数。accelerate 在 dataloader 末尾强制同步（end_of_dataloader），不足 GA 个 micro batch 的尾巴
    也算一步 ⇒ 向上取整（每进程 25 个 micro、GA 2 ⇒ 13 步，最后一步只有 1 个 micro）。"""
    return max(math.ceil(n_micro / max(grad_accum, 1)), 1)


def resume_position(ts: dict, geometry: dict) -> tuple[int, int, str]:
    """dir 续训的数据位置 → (epoch, 该 epoch 开头要跳过的 micro batch 数, 说明)。

    数据顺序 = f(seed, epoch)（SeedableRandomSampler），dropout 抽签 = f(seed, epoch, index) ⇒ 几何（进程数 / GA / micro batch /
    每进程 micro 数 / 训练集大小 / seed）不变时，跳过已消费的 micro batch 就与不中断的训练逐样本相同。
    几何变了 ⇒ 放弃当前 epoch 剩下的部分、从下一个 epoch 开头起（换排列、换抽签，不会重放同一份样本）。
    旧存档（trainer_state 没有 epoch）⇒ 按 step 与当前几何的每 epoch 步数推 epoch（不重放 epoch 0 的抽签，顺序不保证）。"""
    n_micro, ga = int(geometry["n_micro"]), int(geometry["grad_accum"])
    if "epoch" in ts:
        epoch, used = int(ts["epoch"]), int(ts.get("epoch_micro", 0))
        if any(ts.get(k) != v for k, v in geometry.items()):
            return epoch + 1, 0, "几何变了：从下一个 epoch 开头起"
        if used >= n_micro:
            return epoch + 1, 0, "精确（存档时正好在 epoch 末）"
        return epoch, used, "精确"
    spe = epoch_steps(n_micro, ga)
    step = int(ts["step"])
    epoch = step // spe
    return epoch, (step - epoch * spe) * ga, "旧存档：按 step 推算"


def init_ckpt_of_state(state_dir: str, ts: dict) -> str:
    """dir 续训时接回首次启动的 init_ckpt：trainer_state.json 里记的；旧存档没记 ⇒ 读同一 run 的 config.json
    （state_A/B 的上一级）里首次启动的 --resume（是 safetensors 才算）。"""
    if "init_ckpt" in ts:
        return ts["init_ckpt"] or ""
    try:
        with open(os.path.join(os.path.dirname(os.path.realpath(state_dir)), "config.json"), encoding="utf-8") as fh:
            a = json.load(fh)["args"]
    except (OSError, ValueError, KeyError):
        return ""
    r = a.get("init_ckpt") or a.get("resume") or ""
    return r if str(r).endswith(".safetensors") else ""


# ----------------------------------------------------------------------------
# 验证
# ----------------------------------------------------------------------------

def val_variants(batch: dict, null_context: torch.Tensor | None, null_ids: torch.Tensor | None, uncond: bool) -> dict:
    """一条 val 样本 → {cond, uncond, nulltext} 三份输入（uncond / nulltext 只换一种条件，与 infer 的两个 CFG 负支同形）。"""
    out = {"cond": batch}
    if uncond:
        out["uncond"] = dict(batch, sem_ids=torch.full_like(batch["sem_ids"], UNKNOWN_ID))
        b = batch["context"].shape[0]
        out["nulltext"] = dict(batch, context=null_context.expand(b, *null_context.shape[1:]),
                               context_ids=null_ids.expand(b, -1))
    return out


@torch.no_grad()
def run_validation(accelerator: Accelerator, model, module: GcTrainModule, val_ds, tids: list[int], seed: int,
                   null_context=None, null_ids=None, uncond: bool = True):
    """固定 clip 集 × 固定 timestep 集 × 固定噪声（三个 variant 共用同一份噪声），各进程分摊 clip，主进程汇总。

    ZeRO-2 的前向没有集合通信，所以各进程 clip 数不等也没关系；最后的 reduce 是集合操作，人人都要到。
    返回 {t: {"mse", "mse_uncond", "mse_nulltext", "gap_uncond", "gap_nulltext", "n"}}（所有进程都拿到同样的值）。"""
    model.eval()
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    variants = VAL_VARIANTS if uncond else VAL_VARIANTS[:1]
    acc = torch.zeros(len(tids), len(variants) + 1, device=device, dtype=torch.float64)  # 各 variant 的 mse 和, 计数
    for i in range(rank, len(val_ds), world):
        sample = val_ds[i]
        batch = to_device(gc_collate([sample]), device)
        vb = val_variants(batch, null_context, null_ids, uncond)
        for j, tid in enumerate(tids):
            g = torch.Generator(device=device)
            g.manual_seed((zlib.crc32(sample["clip_id"].encode()) * 1009 + tid * 7 + seed) % (2 ** 31))
            noise = torch.randn(batch["input_latents"].shape, generator=g, device=device, dtype=batch["input_latents"].dtype)
            for v, name in enumerate(variants):
                _, stats = model(vb[name], tid, noise=noise, train=False)
                acc[j, v] += stats["mse"].double()
            acc[j, -1] += 1
    acc = accelerator.reduce(acc, reduction="sum")
    model.train()
    out = {}
    for j, tid in enumerate(tids):
        n = max(acc[j, -1].item(), 1.0)
        r = {("mse" if name == "cond" else f"mse_{name}"): acc[j, v].item() / n for v, name in enumerate(variants)}
        if uncond:
            r["gap_uncond"] = r["mse_uncond"] - r["mse"]
            r["gap_nulltext"] = r["mse_nulltext"] - r["mse"]
        r["n"] = int(acc[j, -1].item())
        out[float(module.scheduler.timesteps[tid])] = r
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    # SeedableRandomSampler + 显式 data_seed：每 epoch 的排列只取决于 (seed, epoch)，续训才接得上（见 resume_position）
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=args.seed))
    is_main = accelerator.is_main_process
    use_ds = accelerator.distributed_type == DistributedType.DEEPSPEED
    set_seed(args.seed, device_specific=True)
    if use_ds:
        # accelerate 的 clip_grad_norm_ 在 DeepSpeed 下只取回范数不裁剪，裁剪由引擎按 config 做；
        # 让 --grad_clip 成为唯一入口，yaml 里的 gradient_clipping 只是默认值
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] = float(args.grad_clip)
        ga = accelerator.state.deepspeed_plugin.deepspeed_config.get("gradient_accumulation_steps")
        if ga not in ("auto", args.grad_accum):
            raise SystemExit(f"--grad_accum={args.grad_accum} 与 accelerate yaml 的 gradient_accumulation_steps={ga} 不一致，"
                             f"请换 configs/accelerate_zero2_ga{args.grad_accum}.yaml（scripts/train_sft.sh 会自动选 / 生成）")
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = args.grad_accum
    if is_main:
        os.makedirs(args.output, exist_ok=True)
        print(f"[env] world={accelerator.num_processes} distributed={accelerator.distributed_type} "
              f"mixed_precision={accelerator.mixed_precision} device={accelerator.device} "
              f"res={GEO.RES_TAG}（{GEO.RES_ENV}，latent {GEO.LAT_H}×{GEO.LAT_W}，{GEO.TOK_PER_CELL} token/cell）", flush=True)

    # ---- 数据 ----
    train_ds = GcLatentDataset(args.data_root, "train", text_table=args.text_table, vocab=args.vocab,
                               text_dropout=args.text_dropout, cond_dropout=args.cond_dropout, mask_p=args.mask_p,
                               limit=args.limit, seed=args.seed, include_dropped=args.include_dropped, verbose=is_main)
    val_ds = None
    if args.val_every > 0 and args.val_n > 0:
        # 验证不 dropout（val split 强制）；同一张文本表，别再读一遍；drop=true 的 val clip 一律不进
        val_ds = GcLatentDataset(args.data_root, "val", text_table=train_ds.text_table, vocab=args.vocab,
                                 limit=args.val_n, seed=args.seed, verbose=is_main)
        leak = set(train_ds.clip_ids()) & set(val_ds.clip_ids())
        runs_tr = {train_ds.rows[c]["run"] for c in train_ds.clip_ids()}
        runs_va = {val_ds.rows[c]["run"] for c in val_ds.clip_ids()}
        if leak or (runs_tr & runs_va):
            raise SystemExit(f"train / val 泄漏：共用 clip {sorted(leak)[:3]}，共用 run {sorted(runs_tr & runs_va)[:3]}")
    # persistent_workers=False：每个 epoch 重新 fork worker，dataset.set_epoch 才传得进去（dropout 按 (seed, epoch, index) 抽）
    dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=gc_collate,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False,
    )

    # ---- 模型 ----
    t0 = time.time()
    dit = load_dit(args.model_dir, device="cpu")
    vae = None
    if args.cond_mode == "ids":
        GM.install_sem_cond(dit, args.vocab)
    elif args.cond_mode == "color_light":
        GM.install_wb_cond(dit)
        dit.cond_type = "whitebox"
    else:
        if args.cond_mode == "color_vace":
            GM.install_vace(dit)
        else:
            ref = dit.patch_embedding.weight
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234)
                vc = nn.Conv3d(48, dit.dim, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
            nn.init.zeros_(vc.weight); nn.init.zeros_(vc.bias)
            dit.add_module("gc_vae_cond", vc.to(device=ref.device, dtype=ref.dtype))
        import prep_v10 as PV
        PV.VAE_PATH = os.path.join(args.model_dir, "Wan2.2_VAE.pth")
        vae = PV.load_vae(str(accelerator.device)).vae
        vae.model.requires_grad_(False)
    resume_info = None
    if args.resume and args.resume.endswith(".safetensors"):
        resume_info = load_weights_into_dit(dit, args.resume, args.vocab)
        resume_info["res"] = GEO.ckpt_res(args.resume)
        if is_main:
            print(f"[resume] 热启动权重 {args.resume}: n_tensors={resume_info['n_tensors']} "
                  f"has_sem_cond={resume_info['has_sem_cond']} missing={len(resume_info['missing'])}", flush=True)
            if resume_info["res"] != GEO.RES_TAG:
                print(f"[resume] 注意：热启动 ckpt 的 res={resume_info['res']}，本次 {GEO.RES_ENV}={GEO.RES_TAG}"
                      f"（跨分辨率热启动；DiT / gc_sem_cond 对分辨率通用）", flush=True)
    dit.train().requires_grad_(True)
    module = GcTrainModule(dit, args.first_frame_cond, args.gradient_checkpointing, args.cond_mode, vae)
    cname = ABL_COND_NAME[args.cond_mode]
    enc = getattr(dit, cname)
    n_params = sum(p.numel() for p in dit.parameters())
    n_sem_params = sum(p.numel() for p in enc.parameters())
    if is_main:
        print(f"[model] cond_mode={args.cond_mode} DiT 加载 {time.time() - t0:.0f}s，{n_params:,} 参数（{cname} {n_sem_params:,}，"
              f"vocab={args.vocab}）", flush=True)

    groups = sem_param_groups(module, args.lr, args.cond_lr, args.weight_decay, prefix=f"dit.{cname}.")
    assert len(groups) == 2 and sum(len(g["params"]) for g in groups) == len(list(module.parameters())), \
        "参数组应恰为 [DiT, gc_sem_cond] 且覆盖全部参数"
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda(args.warmup_steps))

    model, optimizer, dataloader = accelerator.prepare(module, optimizer, dataloader)
    device = accelerator.device
    if is_main:
        clip_eff = getattr(getattr(model, "optimizer", None), "clip_grad", None) if use_ds else args.grad_clip
        print(f"[env] grad_clip={args.grad_clip}（生效值 {clip_eff}）", flush=True)
    # RoPE 表搬上卡：每步要查 [f*h*w,1,64] 的 complex128 表，留在 CPU 每步都要拷
    module.dit.freqs = tuple(t.to(device) for t in module.dit.freqs)

    # 数据几何：续训时与存档比对，一致才能按 micro batch 精确接上
    geometry = {"num_processes": accelerator.num_processes, "grad_accum": args.grad_accum, "batch_size": args.batch_size,
                "n_micro": len(dataloader), "n_train": len(train_ds), "seed": args.seed}
    start_step, start_epoch, skip_micro = 0, 0, 0
    if args.resume and not args.resume.endswith(".safetensors"):
        with open(os.path.join(args.resume, "trainer_state.json"), encoding="utf-8") as fh:
            ts = json.load(fh)
        if ts.get("vocab", args.vocab) != args.vocab:
            raise SystemExit(f"{args.resume} 是 vocab={ts.get('vocab')!r} 的存档，本次 --vocab {args.vocab!r}")
        if ts.get("res", GEO.LEGACY_RES) != GEO.RES_TAG:
            raise SystemExit(f"{args.resume} 是 res={ts.get('res', GEO.LEGACY_RES)} 的存档（缺失 = {GEO.LEGACY_RES}），"
                             f"本次 {GEO.RES_ENV}={GEO.RES_TAG}")
        accelerator.load_state(args.resume)
        start_step = int(ts["step"])
        with warnings.catch_warnings():  # 优化器还没 step 就快进 scheduler，PyTorch 会提醒一句，这里是有意的
            warnings.simplefilter("ignore")
            for _ in range(start_step):
                lr_scheduler.step()
        start_epoch, skip_micro, how = resume_position(ts, geometry)
        args.init_ckpt = init_ckpt_of_state(args.resume, ts)
        args.resumed_from, args.resumed_step = os.path.realpath(args.resume), start_step
        if is_main:
            print(f"[resume] accelerator.load_state({args.resume} → {args.resumed_from})，从 step {start_step} 续训；"
                  f"数据从 epoch {start_epoch} 的第 {skip_micro} 个 micro batch 接上（{how}）；init_ckpt={args.init_ckpt or '（底模）'}",
                  flush=True)
            if how.startswith("几何"):
                print(f"[resume] 存档几何 { {k: ts.get(k) for k in geometry} } ≠ 本次 {geometry}", flush=True)

    # ---- config.json：复现所需的一切 ----
    if is_main:
        cfg = {
            "args": vars(args),
            "res": GEO.RES_TAG,
            "dataset": {"root": args.data_root, "res": train_ds.res, "train_total": train_ds.n_total,
                        "train_dropped_rows": train_ds.n_dropped,
                        "train_used": len(train_ds), "text_table_keys": len(train_ds.text_table),
                        "val_used": len(val_ds) if val_ds is not None else 0,
                        "val_clip_ids": val_ds.clip_ids() if val_ds is not None else [],
                        "geometry": geometry, "steps_per_epoch": epoch_steps(len(dataloader), args.grad_accum),
                        "start_epoch": start_epoch, "skip_micro": skip_micro},
            "model": {"dit_files": [os.path.join(args.model_dir, os.path.basename(f)) for f in DIT_FILES],
                      "n_params": n_params, "n_sem_cond_params": n_sem_params, "vocab": args.vocab,
                      "sem_cond_in_channels": enc.in_channels, "export_keys": GC_DIT_KEYS,
                      "param_groups": [{"lr": g["lr"], "n_params": sum(p.numel() for p in g["params"])} for g in groups],
                      "resume_info": resume_info, "start_step": start_step},
            "env": {"world_size": accelerator.num_processes, "distributed_type": str(accelerator.distributed_type),
                    "mixed_precision": accelerator.mixed_precision, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "python": sys.executable, "platform": platform.platform(), "hostname": socket.gethostname(),
                    "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "global_batch": accelerator.num_processes * args.batch_size * args.grad_accum},
            "deepspeed_config": accelerator.state.deepspeed_plugin.deepspeed_config if use_ds else None,
        }
        try:
            import accelerate, deepspeed, diffsynth  # noqa: F401
            cfg["env"]["accelerate"] = accelerate.__version__
            cfg["env"]["deepspeed"] = deepspeed.__version__
        except Exception:
            pass
        with open(os.path.join(args.output, "config.json" if start_step == 0 else f"config_resume{start_step}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False, default=str)

    # ---- 日志 ----
    train_log = val_log = tb = None
    if is_main:
        # clips = 本步 rank 0 各 micro batch 的 clip_id（";" 连接），续训接没接上一眼可查
        log_files = {"train_log.csv": ["step", "loss", "mse", "timestep", "lr", "cond_lr", "grad_norm", "sem_rms", "x_rms",
                                       "text_drop", "cond_drop", "mask_frac", "step_time", "data_time", "mem_gb", "epoch",
                                       "clips"],
                     "val_log.csv": ["step", "timestep", "mse", "mse_uncond", "mse_nulltext", "gap_uncond", "gap_nulltext", "n"]}
        for name, fields in log_files.items():
            old = rotate_csv_if_header_changed(os.path.join(args.output, name), fields)
            if old:
                print(f"[log] {name} 的列变了，旧文件改名 {old}", flush=True)
        train_log = CsvLog(os.path.join(args.output, "train_log.csv"), log_files["train_log.csv"])
        val_log = CsvLog(os.path.join(args.output, "val_log.csv"), log_files["val_log.csv"])
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.output, "tensorboard"))

    # 每个进程各抽各的 timestep 与噪声（与 DiffSynth 一致），种子按 rank 错开，避免 8 张卡同一步抽到同一个 t
    cpu_gen = torch.Generator().manual_seed(args.seed * 1000 + accelerator.process_index + start_step)
    val_tids = [module.timestep_id_for(t) for t in args.val_timesteps]
    null_context = null_ids = None
    if val_ds is not None and args.val_uncond:
        null_context, null_ids = train_ds.context_for([NULL_PROMPT] * CELLS)      # [21,512,4096] / [21]，加 batch 维
        null_context, null_ids = null_context.unsqueeze(0).to(device), null_ids.unsqueeze(0).to(device)

    def do_validation(step):
        if val_ds is None:
            return
        tv = time.time()
        res = run_validation(accelerator, model, module, val_ds, val_tids, args.seed, null_context, null_ids, args.val_uncond)
        if is_main:
            for t, r in res.items():
                val_log.write({"step": step, "timestep": t, **r})
                for k, v in r.items():
                    if k != "n":
                        tb.add_scalar(f"val/{k}_t{int(round(t))}", v, step)
            means = {k: sum(r[k] for r in res.values()) / len(res) for k in next(iter(res.values())) if k != "n"}
            for k, v in means.items():
                tb.add_scalar(f"val/{k}_mean", v, step)
            print(f"[val] step {step}  " + "  ".join(
                f"t={t:.0f}: mse={r['mse']:.4f}" + (f" uncond={r['mse_uncond']:.4f} null={r['mse_nulltext']:.4f}" if args.val_uncond else "")
                + f" (n={r['n']})" for t, r in res.items())
                + "  | mean " + " ".join(f"{k}={v:.4f}" for k, v in means.items()) + f"  {time.time() - tv:.0f}s", flush=True)

    data_pos = {"epoch": start_epoch, "epoch_micro": skip_micro}     # 最近一个完整优化器步之后的数据位置（写进 trainer_state）

    def do_save(step):
        path = os.path.join(args.output, f"step-{step}.safetensors")
        export_dit_weights(accelerator, model, path, export_metadata(args, step))
        if args.save_state:
            save_state_rotating(accelerator, args.output, {"step": step, "arm": "gc25d", "vocab": args.vocab, "res": GEO.RES_TAG,
                                                           "init_ckpt": args.init_ckpt, **data_pos, **geometry})

    # ---- 训练循环 ----
    model.train()
    step = start_step
    # 续训：从存档记下的 (epoch, micro batch) 接着走 —— 该 epoch 先 skip_first_batches 跳过已消费的部分（resume_position）
    steps_per_epoch = epoch_steps(len(dataloader), args.grad_accum)
    epoch = start_epoch
    last_saved = start_step
    step_times, data_times = [], []
    torch.cuda.reset_peak_memory_stats(device)
    if is_main:
        print(f"[train] {len(train_ds)} 个 clip，每进程 batch {args.batch_size} × GA {args.grad_accum}，{accelerator.num_processes} 进程"
              f"（全局 batch {accelerator.num_processes * args.batch_size * args.grad_accum}），每进程每 epoch {len(dataloader)} 个 "
              f"micro batch = {steps_per_epoch} 步，从 step {start_step}（epoch {epoch}，跳过 {skip_micro} 个 micro）到 {args.max_steps}",
              flush=True)
    micro_time = data_time = 0.0
    step_clips = []
    t_prev = time.time()                                    # 上一个 micro batch 处理完的时刻：到下一个 batch 到手之间 = 等数据
    while step < args.max_steps:
        train_ds.set_epoch(epoch)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)                     # SeedableRandomSampler 的排列 = f(seed, epoch)
        epoch_loader, epoch_micro = dataloader, 0
        if skip_micro:
            # 只在续训后的第一个 epoch：在 batch sampler 层跳过（被跳过的样本不加载）；它共用同一个 sampler，epoch 照设
            epoch_loader, epoch_micro, skip_micro = accelerator.skip_first_batches(dataloader, skip_micro), skip_micro, 0
            epoch_loader.set_epoch(epoch)
        for batch in epoch_loader:
            if step >= args.max_steps:
                break
            t_step = time.time()
            data_time += t_step - t_prev
            epoch_micro += 1
            step_clips.extend(batch["clip_id"])
            batch = to_device(batch, device)
            tid = int(torch.randint(0, 1000, (1,), generator=cpu_gen))
            lrs = lr_scheduler.get_last_lr()  # 本步用的 lr（scheduler.step 之后拿到的是下一步的）
            # 梯度累积：accumulate() 决定这个 micro batch 是否到了同步/更新边界（sync_gradients）；
            # DeepSpeed 引擎按 config 自己累积并在边界 step
            with accelerator.accumulate(model):
                loss, stats = model(batch, tid)
                accelerator.backward(loss)  # DeepSpeed 下这一步已包含 engine.step()（裁剪 + 优化器更新）
                # 单进程：真的裁剪；DeepSpeed：clip 已在 engine.step 里按 config 做过，这里只取回全局范数
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.grad_clip) if accelerator.sync_gradients else None
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()  # DeepSpeed 的 scheduler 包装不认累积边界，每调一次就推进一次，只能在这里守门
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t_prev = time.time()
            micro_time += t_prev - t_step
            if not accelerator.sync_gradients:
                continue  # 还在累积，不算一步
            step += 1
            data_pos = {"epoch": epoch, "epoch_micro": epoch_micro}
            # step_time = 整步墙钟（计算 + 等数据），data_time = 其中等 DataLoader 的部分
            step_time, step_data_time, micro_time, data_time = micro_time + data_time, data_time, 0.0, 0.0
            clips, step_clips = ";".join(step_clips), []
            step_times.append(step_time)
            data_times.append(step_data_time)
            loss_mean = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            if not math.isfinite(loss_mean):
                raise SystemExit(f"step {step}: loss = {loss_mean}（非有限），停训")
            if is_main and (step % args.log_every == 0 or step == args.max_steps):
                gn = float(grad_norm) if grad_norm is not None and math.isfinite(float(grad_norm)) else float("nan")
                dr = batch["dropped"]
                row = {"step": step, "loss": loss_mean, "mse": stats["mse"].item(), "timestep": stats["timestep"],
                       "lr": lrs[0], "cond_lr": lrs[-1], "grad_norm": gn,
                       "sem_rms": stats["sem_rms"].item(), "x_rms": stats["x_rms"].item(),
                       "text_drop": float(dr["text"].float().mean()), "cond_drop": float(dr["cond"].float().mean()),
                       "mask_frac": float(dr["mask"].float().mean()), "step_time": step_time, "data_time": step_data_time,
                       "mem_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3, "epoch": epoch, "clips": clips}
                train_log.write(row)
                for k in ("loss", "mse", "lr", "cond_lr", "grad_norm", "sem_rms", "x_rms", "step_time", "data_time", "mem_gb"):
                    tb.add_scalar(f"train/{k}", row[k], step)
                tb.add_scalar("train/sem_over_x", row["sem_rms"] / max(row["x_rms"], 1e-12), step)
                print(f"step {step}/{args.max_steps}  loss {loss_mean:.4f}  (mse {row['mse']:.4f} t {row['timestep']:.0f} "
                      f"sem_rms {row['sem_rms']:.3e} x_rms {row['x_rms']:.3f} drop t{int(row['text_drop'])}c{int(row['cond_drop'])}"
                      f"m{row['mask_frac']:.2f})  lr {row['lr']:.2e}/{row['cond_lr']:.2e}  gnorm {gn:.3f}  "
                      f"{step_time:.2f}s (data {step_data_time:.2f}s)  mem {row['mem_gb']:.1f}GB", flush=True)

            if args.val_every > 0 and step % args.val_every == 0:
                do_validation(step)
            if args.save_every > 0 and step % args.save_every == 0:
                do_save(step)
                last_saved = step
            t_prev = time.time()                            # 验证 / 存盘的时间不算进下一步的等数据
        epoch += 1

    if last_saved != step:
        do_save(step)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 1024 ** 3, device=device)
    peak_max = accelerator.reduce(peak, reduction="max").item()
    if is_main:
        tail = step_times[2:] or step_times
        dtail = data_times[2:] or data_times
        s_step, s_data = sum(tail) / max(len(tail), 1), sum(dtail) / max(len(dtail), 1)
        print(f"[done] step {step}，本进程峰值显存 {peak.item():.1f} GB，各进程最大 {peak_max:.1f} GB，"
              f"s/step {s_step:.2f}（其中等数据 {s_data:.2f}）", flush=True)
        with open(os.path.join(args.output, "done.json"), "w", encoding="utf-8") as fh:
            json.dump({"step": step, "start_step": start_step, "peak_mem_gb_main": peak.item(), "peak_mem_gb_max": peak_max,
                       "s_per_step_mean": s_step, "data_s_per_step_mean": s_data, "n_steps_timed": len(tail),
                       "data_pos": data_pos, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
        train_log.close()
        val_log.close()
        tb.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
