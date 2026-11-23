#!/usr/bin/env python3
"""gc25d 推理 / 评测：首帧（风格）+ 数值语义 id 视频（条件）+ 逐 0.25 s 动作文本 ⇒ 81 帧 W×H 视频。

分辨率 W×H = geometry.DST_W × DST_H（env GC_RES，默认 832×480；下文的 832×480 / [81,480,832] / [.., 30, 52] 都是默认值的例子，
1280×704 时是 [81,704,1280] / [.., 44, 80]，此时源分辨率 = 训练分辨率、不裁不缩）。

三种输入
--------
(i)   --clip_id <id>：处理好的 clip（默认 data_gc25d/），首帧 latent = 缓存 latent 的 cell 0（因果 VAE，cell 0 只看帧 0，
      与单独编码首帧相同），sem_ids = cond/<id>.npz 的 ids，prompts = .pt 里的 21 条；GT = 原始数据集 tar 里的 81 帧 RGB
      经 geometry.resample_rgb；另把缓存 latent 直接解码一遍当"VAE 重建上限"（recon_ceiling）。
      可再给 --actions / --ids / --first_frame 覆盖对应输入（反事实测试），json 里记 overrides。
(ii)  --first_frame <png/jpg> --ids <npz/npy> --actions "<spec>"：用户自带。
      首帧：W×H 原样；1280×704（数据集源分辨率）走 geometry.resample_rgb；其它尺寸先居中裁成 W:H 再 INTER_AREA
      缩放（会打印提示）。ids：uint8 [81,480,832]（npz 取 'ids' 键或唯一数组）；也收源分辨率 [81,704,1280]，逐帧
      geometry.resample_ids。取值必须 ⊂ {0..6, 10, 11, 255}，否则报错（不静默映射成 UNKNOWN）。
      源分辨率 ids 里出现 10/11 默认报错：的条件先去掉角色剪影、在 832×480 上才画椭圆 marker，合规的源分辨率
      条件不含 10/11；原始 ids.png 的剪影是分布外输入，要硬用须加 --allow_raw_ids。
      这条是 gc25d_v8（椭圆 marker）时代的规则，只在源 ≠ 训练分辨率时能按形状判断：GC_RES=1280x704 时源 = 训练分辨率，
      [81,704,1280] 分不出原始 ids.png 与处理过的条件，不做这条检查（v10 的条件本来就是 UE 剪影：ids 词表的 v10 模型
      吃的正是 collapse_parts(cond) = 原始 ids.png；parts 词表的条件含 12..21，本 CLI 不收，用 sample_user.py）。
(iv)  --eval_val N / --clip_ids a,b,c --out_dir D：批量评测，每条 clip 写 mp4 + json，汇总写 D/metrics.json。
      加载模型前先查齐所有 clip 的 latents/cond（prep 没跑完立刻报错）；逐条记 split（--clip_ids 混进 train clip 会提示）；
      aggregate 按 clip 平均，aggregate_scene_balanced 按 scene 等权（N 不是 scene 数倍数时 --eval_val 的配额不均）；
      中途出错把已完成部分写到 metrics_partial.json 再抛出（不写 metrics.json）。

--actions 语法（每段 = 若干个 0.25 s 动作块 = 若干个 latent cell）
-------------------------------------------------------------------
    spec  := seg ("," seg)*
    seg   := keys ":" count
    keys  := tok ("+" tok)*
    tok   := "idle" | "-" | "atk" | "attack" | "j" | 1–4 个 WASD 字母（大小写不限，同一字母至多一次）
    count := 正整数 | "*"（填满剩余块数，至多一段用 *）
各段块数之和必须正好 20（cell 1..20 各一块；cell 0 = cell 1）。严格：空段（'D:10,,D:10'、结尾多逗号）报错；
一段里同一个键（WASD 字母 / atk / idle）至多一次（'W+W'、'atk+j' 报错）；idle 不能与方向键同用（'idle+W' 报错），
idle+atk 允许（= atk，原地攻击，即 viz 标题栏的 'idle atk'）。例：
    "D:8,DW:4,idle:4,atk:1,D:3"   右 2 s → 右上 1 s → 停 1 s → 原地攻击一次 → 右 0.75 s
    "W+atk:2,W:*"                 上移，前两块每块按一次攻击，之后一直上移
    "AD:4,idle:*"                 对向键照原样写进动作块（文本里互相抵消成 standing still）
atk 的含义与数据一致：该 0.25 s 块里按了一次攻击键（attack=1）。也可直接给 .npy：uint8 [20,5] (w,a,s,d,attack)，
与 actions_block_0.25s.npy 同列序。块 → 21 条 prompt 统一走 text.cell_prompts。

采样（照抄 actionrope/infer.py::sample 的 TI2V 路径）
---------------------------------------------------
FlowMatchScheduler("Wan")，默认 50 步 shift 5；噪声 = CPU 上 manual_seed(seed) 的 float32 randn 再转 bf16；
帧 0 的 latent 每步写回干净首帧 z0（并以 first_frame_cond=True 让 cell 0 的 t=0）；timestep 以 latents 的 dtype（bf16）传入，
与训练链路一致；DiT 前向 = model.gc_forward。

CFG（--cfg 文本，--sem_cfg 语义条件；的负条件）
------------------------------------------------------
    v   = f(text, sem)                        正支
    v_t = f(NULL_PROMPT × 21 cell, sem)       文本负支（训练时 text dropout p=0.1 正是这个输入）
    v_s = f(text, ids 全 255)                 语义负支（训练时 full condition dropout p=0.05 正是这个输入）
    pred = v + (cfg − 1)·(v − v_t) + (sem_cfg − 1)·(v − v_s)
设计 只说两种 CFG 各自的负条件，没说怎么合并；这里取"独立条件"的组合式（composable guidance）：
两个负支各只去掉一种条件，恰好都是训练中见过的输入（两者同时去掉的输入训练里只有 0.5% 概率）；
只开一种时退化为标准 CFG（cfg≠1 时 = v_t + cfg·(v − v_t)，与 actionrope 一致）。组合在 float32 里算再转回 bf16。
关掉的那一支不跑前向：cfg = sem_cfg = 1 时每步 1 次前向，都开时 3 次。

--sem_mode：cond（默认，用给定 ids）| unknown（ids 全 255 = 训练里 full condition dropout 的输入，即模型的"无条件"）|
none（sem_ids=None，完全不加语义项；只有零初始化的 gc_sem_cond（--base）时才等于基座模型，配训练过的 ckpt 是分布外
输入——训练从不去掉语义项，会打印提示）。--forward arope：模型代码就绪前的冒烟路径，用 actionrope.arope.arope_forward
（plain RoPE，无语义项）+ 基座 Wan2.2-TI2V-5B，只允许 --base 且 sem_mode=none。

并排视频的 ids / overlay 面板画的是**模型实际收到的条件**：cond = 给定 ids；unknown / none = 全 255 斜纹；面板名写明
sem_mode，生成面板名写 sem_mode / cfg / sem_cfg / 步数 / 种子。json 里 ids_class_frac = 模型收到的条件的类别占比
（none 时为 null），ids_supplied_class_frac = 给定 ids 的；seg_consistency（钩子）始终对照给定（GT）ids。

context_ids 用进程内全局的 串 → id 表（context_id，初值与 dataset 的 key_to_id 相同）：actionrope 的 _dedup_rows 按 id
在整个 batch 上去重，id 必须跨样本一致；sample() 开头还会核对"同 id ⇒ 同内容"，不一致直接报错。

与 设计 / actionrope 的差异（都在这里写明）
-----------------------------------------
* CFG 两项的合并方式见上（设计 未规定）。
* 必须显式给 --ckpt 或 --base：不允许"忘了给 ckpt 就静默用基座权重评测"。
* VAE 解码后转 uint8 用 float32 的 round((x+1)·127.5)（DiffSynth 的 vae_output_to_video 在 bf16 里乘再截断，
  bf16 在 128–255 的间隔是 1，会额外引入 ~0.5 灰阶的系统偏差）；首帧 / 整段编码用 fp32 VAE（数据集也是 fp32 编码后存 bf16）。
* 用户模式没有 GT ⇒ 并排视频不画 GT 面板，也不出 PSNR/SSIM。
* GT 帧来自原始 tar 的 JPEG（cv2 解码）+ geometry.resample_rgb；与 prep 编码 latent 时的解码库若不同，差异在 1 灰阶内。

用法
----
    export PYTHONPATH=$PWD/code/gc25d:$PWD/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True
    CUDA_VISIBLE_DEVICES=2 $PY gc25d/infer.py --ckpt outputs/<run>/step-5000.safetensors --clip_id <val clip> \
        --out outputs/<run>/samples/<clip>.mp4 --overlay
    CUDA_VISIBLE_DEVICES=2 $PY gc25d/infer.py --ckpt ... --first_frame f.png --ids ids.npz \
        --actions "D:8,DW:4,idle:4,atk:1,D:3" --sem_cfg 2.0 --out outputs/<run>/samples/user.mp4
    CUDA_VISIBLE_DEVICES=2 $PY gc25d/infer.py --ckpt ... --eval_val 16 --out_dir outputs/<run>/eval_step-5000
    CUDA_VISIBLE_DEVICES=2 $PY gc25d/infer.py --base --forward arope --clip_id <id> --out ...   # 基座冒烟
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
import argparse
import json
import os
import sys
import tarfile
import time
from typing import Callable, Optional

import numpy as np
import torch

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
# 临时文件只落 nvme（根盘小）
os.environ.setdefault("TMPDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tmp"))   # 独立部署包：包里的 tmp/
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
AROPE_ROOT = os.environ.get("AROPE_ROOT", _mos.path.join(MALIANG_ROOT, "code", "arope"))   # 独立部署包：run_demo.sh 指到包里的 code/arope
for _p in (AROPE_ROOT, HERE):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import cv2  # noqa: E402

import geometry as GEO  # noqa: E402
import metrics as MET  # noqa: E402
import text as TXT  # noqa: E402
import viz as VIZ  # noqa: E402

DATA_ROOT = os.path.join(MALIANG_ROOT, "data", "gc25d_v10_fxcond")
RAW_ROOT = os.path.join(MALIANG_ROOT, "data", "recordings", "data")
MODEL_DIR = os.path.join(MALIANG_ROOT, "models", "Wan2.2-TI2V-5B")
VAE_FILE = "Wan2.2_VAE.pth"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
T5_TOK = os.path.join("google", "umt5-xxl")

F_CLIP, N_CELLS, N_BLOCKS = 81, 21, 20
LAT_C, LAT_H, LAT_W = 48, GEO.LAT_H, GEO.LAT_W       # 832×480 ⇒ 30, 52（geometry / GC_RES）
H, W = GEO.DST_H, GEO.DST_W
SRC_H, SRC_W = GEO.SRC_H, GEO.SRC_W
UNKNOWN_ID = 255                       # 与 model.UNKNOWN_ID 相同；sample() 里以 model 的为准并核对
VALID_IDS = frozenset([0, 1, 2, 3, 4, 5, 6, 10, 11, UNKNOWN_ID])
SHARD = 1600                           # 原始 tar 每片 1,600 帧


# ------------------------------------------------------------------ 动作脚本 ----
def _parse_keys(keys: str) -> list[int]:
    """'D+atk' / 'DW' / 'idle' / 'atk' → [w, a, s, d, attack]。

    严格：同一个键（某个 WASD 字母、atk、idle）在整段里至多出现一次（'W+W'、'atk+j' 报错）；
    idle 不能和方向键同用（'idle+W' 自相矛盾，报错）；idle+atk 允许 = 原地攻击（与 atk 单写相同，
    也正是 viz 标题栏里 'idle atk' 的写法），不存在歧义。"""
    blk = [0, 0, 0, 0, 0]
    toks = [t.strip().lower() for t in keys.split("+")]
    if not toks or any(not t for t in toks):
        raise ValueError(f"动作键为空：{keys!r}")
    idle = False
    for t in toks:
        if t in ("idle", "-"):
            if idle:
                raise ValueError(f"idle 重复：{keys!r}")
            idle = True
            continue
        if t in ("atk", "attack", "j"):
            if blk[4]:
                raise ValueError(f"攻击键重复：{keys!r}")
            blk[4] = 1
            continue
        if not set(t) <= set("wasd") or len(set(t)) != len(t):
            raise ValueError(f"无法解析的动作键 {t!r}（可用 idle / atk / WASD 组合，例如 'DW'、'D+atk'）")
        for ch in t:
            k = "wasd".index(ch)
            if blk[k]:
                raise ValueError(f"方向键 {ch.upper()} 重复：{keys!r}")
            blk[k] = 1
    if idle and any(blk[:4]):
        raise ValueError(f"idle 不能与方向键同用：{keys!r}")
    return blk


def parse_actions(spec: str, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """动作脚本（语法见模块文档）或 .npy 路径 → uint8 [20, 5] (w, a, s, d, attack)，第 k 行 = cell k+1 的动作块。"""
    spec = spec.strip()
    if spec.endswith(".npy"):
        arr = np.load(spec, allow_pickle=False)
        if arr.shape != (n_blocks, 5):
            raise ValueError(f"{spec}：动作块应为 [{n_blocks}, 5]，得到 {arr.shape}")
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{spec}：动作块取值只能是 0/1")
        return arr.astype(np.uint8)
    if not spec:
        raise ValueError("动作脚本为空")
    segs = [s.strip() for s in spec.split(",")]
    if any(not s for s in segs):
        raise ValueError(f"动作脚本里有空段（多余的逗号？）：{spec!r}")
    parsed, star = [], None
    for i, seg in enumerate(segs):
        if seg.count(":") != 1:
            raise ValueError(f"动作段应为 keys:count，得到 {seg!r}")
        keys, cnt = (x.strip() for x in seg.split(":"))
        blk = _parse_keys(keys)
        if cnt == "*":
            if star is not None:
                raise ValueError(f"至多一段用 '*'：{spec!r}")
            star = i
            parsed.append([blk, 0])
        else:
            if not cnt.isdigit() or int(cnt) <= 0:
                raise ValueError(f"块数应为正整数或 '*'，得到 {cnt!r}（段 {seg!r}）")
            parsed.append([blk, int(cnt)])
    total = sum(n for _, n in parsed)
    if star is not None:
        if total >= n_blocks:
            raise ValueError(f"'*' 段没有剩余块可填：其余段已占 {total} / {n_blocks}")
        parsed[star][1] = n_blocks - total
        total = n_blocks
    if total != n_blocks:
        raise ValueError(f"动作块数合计应为 {n_blocks}（每块 0.25 s，cell 1..20），得到 {total}：{spec!r}")
    rows = [blk for blk, n in parsed for _ in range(n)]
    out = np.array(rows, np.uint8)
    assert out.shape == (n_blocks, 5)
    return out


def prompt_runs(prompts: list[str]) -> list[str]:
    """21 条 prompt 压成 'cell a–b: …' 的段（日志 / json 用）。"""
    runs, k0 = [], 0
    for k in range(1, len(prompts) + 1):
        if k == len(prompts) or prompts[k] != prompts[k0]:
            runs.append(f"cell {k0}-{k - 1}: {prompts[k0]}")
            k0 = k
    return runs


# ------------------------------------------------------------------ 文本 ----
class TextEncoder:
    """查 text_table_gc25d.pt；表里没有的串（或 table_path=None）现场用 UMT5 编码（与 build_text_table.py 同一路径：
    DiffSynth tokenizer + WanTextEncoder，有效长度之后清零），并计数写进 json，不静默。"""

    def __init__(self, table_path: Optional[str], device: str, model_dir: str = MODEL_DIR):
        if table_path is not None and not os.path.isfile(table_path):
            raise FileNotFoundError(f"文本表 {table_path} 不存在；要现场用 UMT5 编码请显式给 --text_table none")
        self.table = {} if table_path is None else torch.load(table_path, map_location="cpu", weights_only=False)
        self.table_path, self.device, self.model_dir = table_path, device, model_dir
        self.pipe = None
        self.n_fallback = 0

    def _t5(self, prompts: list[str]) -> list[torch.Tensor]:
        if self.pipe is None:
            print(f"   文本表里没有 {len(prompts)} 条串，加载 UMT5 …", flush=True)
            from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16, device=self.device,
                model_configs=[ModelConfig(path=os.path.join(self.model_dir, T5_FILE))],
                tokenizer_config=ModelConfig(path=os.path.join(self.model_dir, T5_TOK)), redirect_common_files=False)
            self.pipe.load_models_to_device(["text_encoder"])
        ids, mask = self.pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            emb = self.pipe.text_encoder(ids, mask)
        for j, v in enumerate(seq_lens):
            emb[j, v:] = 0
        return [e.to(torch.bfloat16).cpu().clone() for e in emb]

    def encode(self, prompts: list[str]) -> torch.Tensor:
        """list[str] → bf16 [n, 512, 4096]（CPU）。"""
        missing = sorted({p for p in prompts if p not in self.table})
        if missing:
            self.n_fallback += len(missing)
            for p, e in zip(missing, self._t5(missing)):
                self.table[p] = e
        out = torch.stack([self.table[p] for p in prompts], 0)
        assert out.shape[1:] == (512, 4096) and out.dtype == torch.bfloat16, (out.shape, out.dtype)
        return out

    def release(self):
        """采样前把 UMT5（~11 GB）从显存里放掉。"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()


# 文本串 → 全局 context_id（进程内固定，相同串 ⇔ 相同 id，跨 build_context 调用也一样）。
# 初值 = 19 个合法键排序后的下标，与 dataset.GcLatentDataset.key_to_id（文本表键排序）一致；表外串按出现顺序追加。
# 必须是全局的：actionrope.arope._dedup_rows 在整个 batch 的 (b·21) 行上按 id 去重，若 id 只在单次调用内有效，
# 把两次 build_context 的结果拼成 b>1 时，不同的串会共用一个 id，后面的样本静默拿到前面样本的文本。
_KEY_TO_ID: dict[str, int] = {k: i for i, k in enumerate(sorted(TXT.all_prompts() + [TXT.NULL_PROMPT]))}


def context_id(prompt: str) -> int:
    return _KEY_TO_ID.setdefault(prompt, len(_KEY_TO_ID))


def build_context(enc: TextEncoder, prompts: list[str], device: str):
    """21 条 cell prompt → (context bf16 [1,21,512,4096], context_ids int64 [1,21])，与训练的逐 cell 形式一致；
    context_ids = 全局 id（context_id，只用于 _embed_context 去重）。多条结果可直接在 dim 0 上拼成 batch。"""
    assert len(prompts) == N_CELLS, f"应有 {N_CELLS} 条 cell prompt，得到 {len(prompts)}"
    uniq = sorted(set(prompts))
    emb = enc.encode(uniq)
    pos = torch.tensor([uniq.index(p) for p in prompts], dtype=torch.int64)
    ids = torch.tensor([context_id(p) for p in prompts], dtype=torch.int64)
    return emb[pos].unsqueeze(0).to(device), ids.unsqueeze(0).to(device)


def check_context_ids(context: torch.Tensor, context_ids, name: str = "context") -> None:
    """相同 context_id 的行内容必须逐位相同（_dedup_rows 只看 id，不看内容）；逐 cell [b,21,512,4096] 才检查。
    每次 sample() 只做一次（≤ b·21 次 torch.equal），防止外部调用者传入只在单条样本内有效的 id。"""
    if context_ids is None or context.dim() != 4:
        return
    rows = context.reshape(-1, *context.shape[-2:])
    ids = torch.as_tensor(context_ids).reshape(-1).tolist()
    assert len(ids) == rows.shape[0], f"{name}_ids 应有 {rows.shape[0]} 个，得到 {len(ids)}"
    first: dict[int, int] = {}
    for i, k in enumerate(ids):
        j = first.setdefault(int(k), i)
        if j != i:
            assert torch.equal(rows[i], rows[j]), \
                f"{name}_ids 里 id {k} 对应了不同的内容（行 {j} vs {i}）：batch 内的 id 必须全局一致（见 infer.context_id）"


# ------------------------------------------------------------------ 模型 ----
def _load_base_dit(model_dir: str, device: str):
    """基座 DiT（与 actionrope/train.py::load_dit 同一路径），只给 --forward arope 冒烟用。"""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    files = [os.path.join(model_dir, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device,
                                            model_configs=[ModelConfig(path=files)],
                                            tokenizer_config=None, redirect_common_files=False)
    return pipe.dit


def load_vae(device: str, model_dir: str = MODEL_DIR):
    """只加载 Wan2.2 VAE（WanVideoVAE38，bf16），返回 pipe（用 pipe.vae / pipe.device / pipe.torch_dtype）。"""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device,
                                            model_configs=[ModelConfig(path=os.path.join(model_dir, VAE_FILE))],
                                            tokenizer_config=None, redirect_common_files=False)
    assert pipe.vae is not None, "VAE 没加载上"
    return pipe


def load_models(ckpt: Optional[str], base: bool, device: str, forward: str = "gc", model_dir: str = MODEL_DIR,
                need_vae: bool = True) -> dict:
    """→ {"dit", "vae_pipe", "forward": callable, "info"}。

    forward="gc"：model.load_dit + install_sem_cond（零初始化）+ load_gc_weights(ckpt)；--base 时不加载 ckpt
                  （新装的 gc_sem_cond 输出为零 ⇒ 与基座完全相同）。
    forward="arope"：基座 DiT + actionrope.arope.arope_forward（无语义项），只允许 base。"""
    if bool(ckpt) == bool(base):
        raise ValueError("必须二选一：--ckpt <safetensors>（训练好的权重）或 --base（基座权重，冒烟 / 对照）")
    info = {"ckpt": ckpt, "base": bool(base), "forward": forward, "model_dir": model_dir, "res": GEO.RES_TAG}
    if ckpt:
        GEO.check_ckpt_res(ckpt, "--ckpt")                  # ckpt metadata 的 res 必须 = GC_RES（缺失 ⇒ 832x480）
    t0 = time.time()
    if forward == "gc":
        import model as GM
        dit = GM.load_dit(model_dir, device=device)
        GM.install_sem_cond(dit)
        if ckpt:
            info["load_info"] = GM.load_gc_weights(dit, ckpt)
        dit = dit.to(device)
        sem = getattr(dit, GM.SEM_COND_NAME)
        info["sem_cond_params"] = int(sum(p.numel() for p in sem.parameters()))
        info["sem_cond_dtype"] = str(next(sem.parameters()).dtype)
        assert GM.UNKNOWN_ID == UNKNOWN_ID, (GM.UNKNOWN_ID, UNKNOWN_ID)
        fwd = GM.gc_forward
    elif forward == "arope":
        if ckpt:
            raise ValueError("--forward arope 只用于基座冒烟（--base），gc 的 ckpt 请用默认 --forward gc")
        dit = _load_base_dit(model_dir, device)
        fwd = arope_forward_nosem
    else:
        raise ValueError(f"未知 forward {forward!r}")
    dit = dit.eval().requires_grad_(False)
    info["dit_load_sec"] = time.time() - t0
    t0 = time.time()
    vae_pipe = load_vae(device, model_dir) if need_vae else None
    info["vae_load_sec"] = time.time() - t0
    return {"dit": dit, "vae_pipe": vae_pipe, "forward": fwd, "info": info}


def arope_forward_nosem(dit, latents, timestep, context, sem_ids, *, first_frame_cond=True, context_ids=None, **_):
    """gc_forward 同签名的基座前向（plain RoPE，无语义项）。sem_ids 必须为 None —— 不能假装用了条件。"""
    assert sem_ids is None, "arope 前向没有语义条件，sem_ids 必须为 None（--sem_mode none）"
    from actionrope.arope import arope_forward
    return arope_forward(dit, latents, timestep, context, first_frame_cond=first_frame_cond, context_ids=context_ids)


# ------------------------------------------------------------------ VAE ----
class _vae_fp32:
    """临时把 VAE 切到 fp32 编码（数据集 latent 是 fp32 编码后存 bf16），用完切回 bf16。"""

    def __init__(self, pipe, enabled: bool):
        self.pipe, self.enabled = pipe, enabled

    def __enter__(self):
        if self.enabled:
            self.pipe.vae.model.float()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.pipe.vae.model.to(self.pipe.torch_dtype)


def encode_video(vae_pipe, frames: np.ndarray, fp32: bool = True) -> torch.Tensor:
    """uint8 [F,H,W,3]（F = 1 或 4k+1）→ latent bf16 [1, 48, (F−1)/4+1, H/16, W/16]（non-tiled，帧映射到 [-1, 1]）。"""
    assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[1:] == (H, W, 3), (frames.dtype, frames.shape)
    assert (frames.shape[0] - 1) % 4 == 0, f"帧数应为 4k+1，得到 {frames.shape[0]}"
    dtype = torch.float32 if fp32 else vae_pipe.torch_dtype
    x = torch.from_numpy(frames).permute(3, 0, 1, 2).to(dtype).div_(127.5).sub_(1.0)       # [3, F, H, W]
    with torch.no_grad(), _vae_fp32(vae_pipe, fp32):
        z = vae_pipe.vae.encode([x], device=vae_pipe.device, tiled=False)
    z = z.to(dtype=torch.bfloat16, device=vae_pipe.device)
    assert z.shape == (1, LAT_C, (frames.shape[0] - 1) // 4 + 1, LAT_H, LAT_W), z.shape
    return z


def encode_first_frame(vae_pipe, img: np.ndarray, fp32: bool = True) -> torch.Tensor:
    """uint8 [H,W,3] → z0 bf16 [1, 48, 1, H/16, W/16]。"""
    return encode_video(vae_pipe, img[None], fp32=fp32)


def decode_video(vae_pipe, latents: torch.Tensor) -> np.ndarray:
    """latent [1,48,21,H/16,W/16] → uint8 [81,H,W,3]（non-tiled；float32 里 round((x+1)·127.5)）。"""
    assert latents.dim() == 5 and latents.shape[0] == 1 and latents.shape[1] == LAT_C, latents.shape
    with torch.no_grad():
        video = vae_pipe.vae.decode(latents.to(vae_pipe.torch_dtype), device=vae_pipe.device, tiled=False)  # [1,3,F,H,W]，已 clamp 到 [-1,1]
    v = video[0].float().add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    frames = v.permute(1, 2, 3, 0).cpu().numpy()
    assert frames.shape == ((latents.shape[2] - 1) * 4 + 1, H, W, 3), frames.shape
    return frames


# ------------------------------------------------------------------ 采样 ----
@torch.no_grad()
def sample(dit, z0: torch.Tensor, context, context_ids, sem_ids, *, steps: int, shift: float, cfg: float,
           sem_cfg: float, seed: int, neg_context=None, neg_context_ids=None,
           forward: Optional[Callable] = None, log=print) -> torch.Tensor:
    """flow-matching 采样 → latents [b, 48, 21, H/16, W/16]（帧 0 = z0）。

    z0 [b,48,1,H/16,W/16]；context [b,21,512,4096]（逐 cell）或 [b,512,4096]；sem_ids uint8 [b,81,H,W] 或 None
    （彩色白模模型 cond_key=wb_rgb：[b,81,H,W,3]）；
    cfg ≠ 1 需要 neg_context（NULL prompt）；sem_cfg ≠ 1 需要 sem_ids（负支 = 同形状的全 255）。
    forward：默认 model.gc_forward；测试 / 基座冒烟可传同签名的函数。合并公式见模块文档。"""
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    if forward is None:
        import model as GM
        forward = GM.gc_forward
    b = z0.shape[0]
    assert z0.shape == (b, LAT_C, 1, LAT_H, LAT_W), f"z0 应为 [b,48,1,{LAT_H},{LAT_W}]（{GEO.RES_ENV}={GEO.RES_TAG}），得到 {tuple(z0.shape)}"
    assert context.shape[0] == b and context.shape[-2:] == (512, 4096) and context.dim() in (3, 4), tuple(context.shape)
    if context.dim() == 4:
        assert context.shape[1] == N_CELLS, f"逐 cell context 应有 {N_CELLS} 个 cell，得到 {tuple(context.shape)}"
    if sem_ids is not None:
        assert sem_ids.dtype == torch.uint8 and tuple(sem_ids.shape) in ((b, F_CLIP, H, W), (b, F_CLIP, H, W, 3)), \
            f"sem_ids 应为 uint8 [b,81,{H},{W}]（彩色白模 [b,81,{H},{W},3]），得到 {sem_ids.dtype} {tuple(sem_ids.shape)}"
        sem_ids = sem_ids.to(z0.device)
    check_context_ids(context, context_ids, "context")
    if cfg != 1.0:
        assert neg_context is not None and neg_context.shape == context.shape, "cfg≠1 需要与 context 同形状的 neg_context"
        check_context_ids(neg_context, neg_context_ids, "neg_context")
    if sem_cfg != 1.0:
        assert sem_ids is not None, "sem_cfg≠1 需要 sem_ids（负支 = 全 UNKNOWN）"
    sem_neg = torch.full_like(sem_ids, UNKNOWN_ID) if sem_cfg != 1.0 else None

    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(steps, shift=shift)
    shape = (b, LAT_C, N_CELLS, LAT_H, LAT_W)
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype=z0.dtype, device=z0.device)
    latents[:, :, 0:1] = z0
    t0 = time.time()
    for i, t in enumerate(sched.timesteps):
        timestep = t.unsqueeze(0).to(dtype=latents.dtype, device=latents.device)
        v = forward(dit, latents, timestep, context, sem_ids, first_frame_cond=True, context_ids=context_ids)
        assert v.shape == latents.shape, (v.shape, latents.shape)
        if cfg == 1.0 and sem_cfg == 1.0:
            pred = v
        else:
            pred = v.float()
            if cfg != 1.0:
                v_t = forward(dit, latents, timestep, neg_context, sem_ids, first_frame_cond=True, context_ids=neg_context_ids)
                pred = pred + (cfg - 1.0) * (v.float() - v_t.float())
            if sem_cfg != 1.0:
                v_s = forward(dit, latents, timestep, context, sem_neg, first_frame_cond=True, context_ids=context_ids)
                pred = pred + (sem_cfg - 1.0) * (v.float() - v_s.float())
            pred = pred.to(latents.dtype)
        latents = sched.step(pred, sched.timesteps[i], latents)
        latents[:, :, 0:1] = z0
        if (i + 1) % 10 == 0 or i + 1 == steps:
            log(f"   step {i + 1}/{steps} · {time.time() - t0:.1f}s")
    return latents


# ------------------------------------------------------------------ 数据 ----
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_clip(data_root: str, clip_id: str) -> dict:
    """处理好的 clip → {latent bf16 [48,21,H/16,W/16], prompts [21], blocks uint8 [21,5], ids uint8 [81,H,W], meta}。"""
    pt_path, npz_path = clip_files(data_root, clip_id)
    for p in (pt_path, npz_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{clip_id}：缺 {p}")
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    lat = d["latent"]
    assert lat.dtype == torch.bfloat16 and tuple(lat.shape) == (LAT_C, N_CELLS, LAT_H, LAT_W), \
        f"{pt_path}：latent 应为 bf16 [48,21,{LAT_H},{LAT_W}]（{GEO.RES_ENV}={GEO.RES_TAG}），得到 {lat.dtype} {tuple(lat.shape)}"
    prompts = list(d["prompts"])
    assert len(prompts) == N_CELLS and all(isinstance(p, str) for p in prompts), f"{pt_path}：prompts 应为 21 条 str"
    blocks = np.asarray(d["actions_block"].numpy() if torch.is_tensor(d["actions_block"]) else d["actions_block"])
    assert blocks.shape == (N_CELLS, 5), f"{pt_path}：actions_block 应为 [21,5]，得到 {blocks.shape}"
    with np.load(npz_path, allow_pickle=False) as z:
        ids = z["ids"]
    assert ids.dtype == np.uint8 and ids.shape == (F_CLIP, H, W), f"{npz_path}：ids 应为 uint8 [81,{H},{W}]，得到 {ids.dtype} {ids.shape}"
    meta = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in d.items()
            if k not in ("latent", "prompts", "actions_block") and isinstance(v, (str, int, float, bool, np.integer, np.floating))}
    return {"latent": lat, "prompts": prompts, "blocks": blocks.astype(np.uint8), "ids": ids, "meta": meta}


def parse_clip_id(clip_id: str) -> tuple[str, str, int]:
    """'05_Japanese_Spring_Sanctuary__def_b__000160' → (scene, run, start)。"""
    parts = clip_id.split("__")
    if len(parts) != 3 or not parts[2].isdigit():
        raise ValueError(f"clip_id 应为 scene__run__start(6 位)，得到 {clip_id!r}")
    return parts[0], parts[1], int(parts[2])


def read_raw_frames(scene: str, run: str, start: int, n: int = F_CLIP, kind: str = "jpg",
                    raw_root: str = RAW_ROOT) -> np.ndarray:
    """原始 tar 里帧 [start, start+n) 的 RGB（kind="jpg"，uint8 [n,704,1280,3]）或 ids（kind="ids.png"，[n,704,1280]）。
    只把成员字节读进内存解码（不解包到磁盘；原始数据当不可信输入）。"""
    run_dir = os.path.join(raw_root, scene, run)
    want = set(range(start, start + n))
    got: dict[int, np.ndarray] = {}
    for shard in range(start // SHARD * SHARD, start + n, SHARD):
        path = os.path.join(run_dir, f"frames-{shard:06d}.tar")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        last = min(shard + SHARD, start + n) - 1
        with tarfile.open(path, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key, _, ext = m.name.rpartition("/")[2].partition(".")
                if not key.isdigit():
                    continue
                fi = int(key)
                if fi in want and ext == kind:
                    buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
                    if kind == "jpg":
                        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        assert img is not None, f"{path}:{m.name} 解码失败"
                        got[fi] = img[:, :, ::-1].copy()
                    else:
                        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                        assert img is not None and img.ndim == 2 and img.dtype == np.uint8, f"{path}:{m.name} 应为单通道 uint8"
                        got[fi] = img
                if fi > last:
                    break                               # tar 内按帧号有序（README）：过了要的最后一帧就不再往后扫；缺帧下面会报错
    missing = sorted(want - set(got))
    if missing:
        raise RuntimeError(f"{scene}/{run}：缺帧 {missing[:5]}…（共 {len(missing)}）")
    out = np.stack([got[f] for f in range(start, start + n)], 0)
    assert out.shape[1:3] == (SRC_H, SRC_W), out.shape
    return out


def gt_frames_for_clip(clip_id: str, raw_root: str = RAW_ROOT) -> np.ndarray:
    """val clip 的 GT：原始 81 帧 → geometry.resample_rgb→ uint8 [81,H,W,3]。"""
    scene, run, start = parse_clip_id(clip_id)
    raw = read_raw_frames(scene, run, start, F_CLIP, "jpg", raw_root)
    out = np.stack([GEO.resample_rgb(f) for f in raw], 0)
    assert out.shape == (F_CLIP, H, W, 3) and out.dtype == np.uint8, out.shape
    return out


def load_image_rgb(path: str) -> np.ndarray:
    """png/jpg → uint8 [H,W,3]（规则见模块文档）。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不了图像 {path}")
    img = img[:, :, ::-1].copy()
    h, w = img.shape[:2]
    if (h, w) == (H, W):
        return img
    if (h, w) == (SRC_H, SRC_W):
        return GEO.resample_rgb(img)
    th, tw = (h, round(h * W / H)) if w * H >= h * W else (round(w * H / W), w)
    y0, x0 = (h - th) // 2, (w - tw) // 2
    print(f"   注意：首帧 {w}×{h} 不是 {W}×{H} / 1280×704，居中裁成 {tw}×{th} 再缩放到 {W}×{H}")
    return cv2.resize(img[y0:y0 + th, x0:x0 + tw], (W, H), interpolation=cv2.INTER_AREA)


def load_ids(path: str, allow_raw_ids: bool = False) -> np.ndarray:
    """npz/npy → uint8 [81,H,W]（规则见模块文档）。

    源分辨率 [81,704,1280] 输入里若有 10/11 ⇒ 默认报错：的条件是先在源分辨率上去掉角色剪影（atlas 填充）、
    重采样到 832×480 **之后**才画椭圆 marker，所以合规的源分辨率条件里不会有 10/11；有就是原始 ids.png 的角色剪影，
    模型训练时从没见过。allow_raw_ids=True（--allow_raw_ids）时照样重采样（分布外，打印提示）。
    1280×704（源 = 训练分辨率）时形状分不出原始与处理过的 ids ⇒ 不查 10/11（见模块文档 (ii)）。"""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            if "ids" in keys:
                arr = z["ids"]
            elif len(keys) == 1:
                arr = z[keys[0]]
            else:
                raise ValueError(f"{path}：npz 里没有 'ids' 键且不止一个数组：{keys}")
    elif path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"--ids 应为 .npz / .npy，得到 {path}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{path}：ids 应为 uint8，得到 {arr.dtype}")
    if arr.shape == (F_CLIP, SRC_H, SRC_W) and (SRC_H, SRC_W) != (H, W):     # 1280×704 时源 = 训练分辨率，按训练分辨率收
        print(f"   ids 是源分辨率 1280×704，逐帧 geometry.resample_ids → {W}×{H}")
        chars = sorted({10, 11} & set(np.unique(arr).tolist()))
        if chars:
            msg = (f"{path}：源分辨率 ids 含角色类 {chars}（原始 ids.png 的角色剪影）。的条件先去掉剪影、在 832×480 上"
                   f"再画椭圆 marker，训练里没有剪影形状的 10/11；请用 prep 的流程生成 832×480 条件，"
                   f"或加 --allow_raw_ids 原样重采样（分布外）")
            if not allow_raw_ids:
                raise ValueError(msg)
            print(f"   注意（--allow_raw_ids）：{msg}")
        arr = np.stack([GEO.resample_ids(a) for a in arr], 0)
    if arr.shape != (F_CLIP, H, W):
        raise ValueError(f"{path}：ids 应为 [81,{H},{W}]（或源分辨率 [81,704,1280]），得到 {arr.shape}")
    bad = sorted(set(np.unique(arr).tolist()) - VALID_IDS)
    if bad:
        raise ValueError(f"{path}：ids 含非法取值 {bad}（允许 {sorted(VALID_IDS)}；未知区域请写 255）")
    return np.ascontiguousarray(arr)


def select_val_clips(data_root: str, n: int) -> list[str]:
    """从 index/clips.jsonl 的 val 里确定性地挑 n 条：按 scene 平均分配，scene 内等距取。"""
    rows = [r for r in read_jsonl(os.path.join(data_root, "index", "clips.jsonl")) if r["split"] == "val"]
    if not rows:
        raise RuntimeError(f"{data_root}/index/clips.jsonl 里没有 val clip")
    by_scene: dict[str, list[str]] = {}
    for r in rows:
        by_scene.setdefault(r["scene"], []).append(r["clip_id"])
    scenes = sorted(by_scene)
    for s in scenes:
        by_scene[s].sort()
    if n >= len(rows):
        print(f"   要 {n} 条，val 只有 {len(rows)} 条，全取")
        return sorted(r["clip_id"] for r in rows)
    quota = {s: 0 for s in scenes}
    left, k = n, 0
    while left > 0:                                   # 轮流给每个还有余量的 scene 分 1 条
        s = scenes[k % len(scenes)]
        if quota[s] < len(by_scene[s]):
            quota[s] += 1
            left -= 1
        k += 1
    if len(set(quota.values())) > 1:
        print(f"   注意：{n} 条在 {len(scenes)} 个 scene 间分不匀（{ {s: quota[s] for s in scenes} }）；"
              f"aggregate 是按 clip 平均（偏向条数多的 scene），scene 等权请看 metrics.json 的 aggregate_scene_balanced")
    out = []
    for s in scenes:
        L, m = by_scene[s], quota[s]
        out += [L[int((j + 0.5) * len(L) / m)] for j in range(m)]
    assert len(out) == n and len(set(out)) == n
    return out


def clip_index(data_root: str) -> dict[str, dict]:
    """index/clips.jsonl → {clip_id: row}；没有索引文件 ⇒ {}（调用方会提示 split 未知）。"""
    path = os.path.join(data_root, "index", "clips.jsonl")
    if not os.path.isfile(path):
        print(f"   注意：{path} 不存在，clip 的 split 记为 null")
        return {}
    return {r["clip_id"]: r for r in read_jsonl(path)}


def clip_files(data_root: str, clip_id: str) -> list[str]:
    return [os.path.join(data_root, "latents", clip_id + ".pt"), os.path.join(data_root, "cond", clip_id + ".npz")]


# ------------------------------------------------------------------ 一条样本 ----
def prepare_item(args, models: dict, enc: TextEncoder, clip_id: Optional[str]) -> dict:
    """把 CLI 输入整理成 {z0, ids, prompts, blocks, gt, latent_cached, title, overrides, meta}。"""
    vae_pipe = models["vae_pipe"]
    dev = args.device
    item: dict = {"clip_id": clip_id, "overrides": []}
    if clip_id is not None:
        c = load_clip(args.data_root, clip_id)
        item.update(prompts=c["prompts"], blocks=c["blocks"], ids=c["ids"], meta=c["meta"], latent_cached=c["latent"])
        item["z0"] = c["latent"][:, 0:1].unsqueeze(0).to(dev)
        item["gt"] = gt_frames_for_clip(clip_id, args.raw_root)
        item["title"] = clip_id
    else:
        missing = [k for k in ("first_frame", "ids", "actions") if not getattr(args, k)]
        if missing:
            raise ValueError(f"用户模式缺参数：{['--' + m for m in missing]}（或改用 --clip_id）")
        item.update(gt=None, latent_cached=None, meta={}, title=os.path.basename(args.first_frame))
    if args.first_frame:
        img = load_image_rgb(args.first_frame)
        item["first_frame"] = img
        item["z0"] = encode_first_frame(vae_pipe, img)
        if clip_id is not None:
            item["overrides"].append("first_frame")
    if args.ids:
        item["ids"] = load_ids(args.ids, allow_raw_ids=args.allow_raw_ids)
        if clip_id is not None:
            item["overrides"].append("ids")
    if args.actions:
        blocks20 = parse_actions(args.actions)
        item["blocks"] = TXT.cell_blocks(blocks20)                 # cell 顺序 [21,5]，cell 0 = cell 1
        item["prompts"] = TXT.cell_prompts(blocks20)
        if clip_id is not None:
            item["overrides"].append("actions")
    assert len(item["prompts"]) == N_CELLS, len(item["prompts"])
    return item


SEM_INPUT = {"cond": "supplied ids", "unknown": "all 255 (UNKNOWN)", "none": "none (no sem term)"}
_IDS_PANEL = {"cond": "ids = model input (sem_mode=cond)",
              "unknown": "model input = all 255 UNKNOWN (sem_mode=unknown)",
              "none": "NO sem term (sem_mode=none): no ids given"}


def _class_frac(ids: np.ndarray) -> dict:
    cls, cnt = np.unique(ids, return_counts=True)
    return {int(k): float(c) / ids.size for k, c in zip(cls, cnt)}


def panel_names(args) -> dict:
    """并排视频的面板名：生成面板写 sem_mode / cfg / sem_cfg / 步数 / 种子（/ 前向），ids 面板写它画的是不是模型输入。"""
    fwd = "" if args.forward == "gc" else f"  fwd={args.forward}"
    gen = (f"generated  [sem={args.sem_mode}  cfg={args.cfg:g}  sem_cfg={args.sem_cfg:g}  "
           f"{args.steps} steps  seed={args.seed}{fwd}]")
    ids = _IDS_PANEL[args.sem_mode]
    return {"gen_name": gen, "ids_name": ids,
            "overlay_name": "overlay: generated x model-input ids" if args.sem_mode == "cond" else "overlay (no ids given)"}


def run_item(args, models: dict, enc: TextEncoder, item: dict, out_mp4: str) -> dict:
    """采样 + 解码 + 写视频 / json；返回 info（含指标）。"""
    dev = args.device
    dit, vae_pipe, fwd = models["dit"], models["vae_pipe"], models["forward"]
    info = {k: v for k, v in models["info"].items()}
    info.update(clip_id=item["clip_id"], steps=args.steps, shift=args.shift, cfg=args.cfg, sem_cfg=args.sem_cfg,
                seed=args.seed, sem_mode=args.sem_mode, overrides=item["overrides"], meta=item["meta"],
                prompt_runs=prompt_runs(item["prompts"]), actions_block=np.asarray(item["blocks"]).tolist())
    print(f"[{item['title']}] 文本：\n   " + "\n   ".join(info["prompt_runs"]))

    context, context_ids = build_context(enc, item["prompts"], dev)
    neg_context = neg_ids = None
    if args.cfg != 1.0:
        neg_context, neg_ids = build_context(enc, [TXT.NULL_PROMPT] * N_CELLS, dev)
        info["neg_prompt"] = TXT.NULL_PROMPT
    enc.release()
    info["text_fallback_t5"] = enc.n_fallback

    # ids_np = 给定的 ids（clip 模式即 GT 条件）；ids_vis = 模型实际收到的条件（并排视频的 ids / overlay 面板画它）。
    # sem_mode none（或 --forward arope）时模型根本没有语义项，面板画全 255 的斜纹并在面板名里写明"无语义项"。
    ids_np = item["ids"]
    if args.sem_mode == "cond":
        sem = torch.from_numpy(ids_np).unsqueeze(0).to(dev)
        ids_vis = ids_np
    elif args.sem_mode == "unknown":
        sem = torch.full((1, F_CLIP, H, W), UNKNOWN_ID, dtype=torch.uint8, device=dev)
        ids_vis = np.full_like(ids_np, UNKNOWN_ID)
    elif args.sem_mode == "none":
        sem = None
        ids_vis = np.full_like(ids_np, UNKNOWN_ID)
    else:
        raise ValueError(f"未知 sem_mode {args.sem_mode!r}")
    info["sem_input"] = SEM_INPUT[args.sem_mode]
    info["ids_class_frac"] = _class_frac(ids_vis) if sem is not None else None        # 模型收到的条件
    info["ids_supplied_class_frac"] = _class_frac(ids_np)                              # 给定的 ids（未必给了模型）

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    latents = sample(dit, item["z0"], context, context_ids, sem, steps=args.steps, shift=args.shift, cfg=args.cfg,
                     sem_cfg=args.sem_cfg, seed=args.seed, neg_context=neg_context, neg_context_ids=neg_ids, forward=fwd)
    torch.cuda.synchronize()
    info["sample_sec"] = time.time() - t0
    t0 = time.time()
    frames = decode_video(vae_pipe, latents)
    info["decode_sec"] = time.time() - t0
    info["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"   采样 {info['sample_sec']:.1f}s（{args.steps} 步，cfg={args.cfg}，sem_cfg={args.sem_cfg}）· 解码 {info['decode_sec']:.1f}s"
          f" · 峰值显存 {info['peak_mem_gb']:.1f} GB")

    # ---- 指标 ----
    gt = item["gt"]
    if gt is not None:
        info["metrics"] = MET.clip_metrics(frames, gt)
        if item["latent_cached"] is not None and args.recon_ceiling:
            recon = decode_video(vae_pipe, item["latent_cached"].unsqueeze(0).to(dev))
            rc = MET.clip_metrics(recon, gt)
            info["recon_ceiling"] = {k: v for k, v in rc.items() if k not in ("psnr", "ssim")}
        # HOOK，现状 None。对照的是给定（GT）条件 ids，不随 sem_mode 变：衡量"生成是否贴着真实布局"
        info["seg_consistency"] = MET.segmentation_consistency(frames, ids_np, segmenter=None)
        m = info["metrics"]
        print(f"   vs GT：PSNR 帧0 {m['psnr_frame0']:.2f} · 帧1–80 均值 {m['psnr_mean']:.2f} dB · SSIM 均值 {m['ssim_mean']:.4f}"
              + (f" ·（VAE 重建上限 PSNR {info['recon_ceiling']['psnr_mean']:.2f} / SSIM {info['recon_ceiling']['ssim_mean']:.4f}）"
                 if "recon_ceiling" in info else ""))

    # ---- 写文件 ----
    stem = os.path.splitext(out_mp4)[0]
    t0 = time.time()
    title = item["title"] + (f"  ckpt={os.path.basename(args.ckpt)}" if args.ckpt else "  base")
    names = panel_names(args)
    sbs = VIZ.write_side_by_side(out_mp4, frames, ids_vis, gt=gt, overlay=args.overlay, prompts=item["prompts"],
                                 blocks=item["blocks"], title=title, **names)
    info["panels"] = list(sbs.names)
    VIZ.write_video(stem + "_gen.mp4", frames)
    VIZ.contact_sheet(stem + "_sheet.jpg", sbs)
    if args.save_latents:
        torch.save({"latents": latents.cpu(), "prompts": item["prompts"], "info": dict(info)}, stem + "_latents.pt")
    info["write_sec"] = time.time() - t0
    info.update(out=out_mp4, out_gen=stem + "_gen.mp4", out_sheet=stem + "_sheet.jpg", frames=int(frames.shape[0]),
                resolution=[int(frames.shape[2]), int(frames.shape[1])])
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, default=_json_default)
    print(f"   写出 {out_mp4}（{VIZ.FPS} fps，{sbs.width}×{sbs.height}）")
    return info


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------ CLI ----
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="gc25d 推理 / 评测")
    w = ap.add_argument_group("权重")
    w.add_argument("--ckpt", default=None, help="训练导出的 safetensors（含 gc_sem_cond.*）")
    w.add_argument("--base", action="store_true", help="用基座 Wan2.2-TI2V-5B（gc_sem_cond 零初始化 ⇒ 与基座相同）")
    w.add_argument("--forward", choices=("gc", "arope"), default="gc",
                   help="gc = model.gc_forward（默认）；arope = 基座冒烟（无语义项，只配 --base）")
    w.add_argument("--model_dir", default=MODEL_DIR)
    i = ap.add_argument_group("输入")
    i.add_argument("--clip_id", default=None, help="处理好的 clip（data_root/latents + cond）")
    i.add_argument("--first_frame", default=None, help=f"png/jpg（{W}×{H} 或 1280×704）")
    i.add_argument("--ids", default=None, help=f"npz/npy uint8 [81,{H},{W}]（或源分辨率 [81,704,1280]，不得含 10/11）")
    i.add_argument("--allow_raw_ids", action="store_true",
                   help="允许源分辨率 ids 含角色剪影 10/11（原始 ids.png；条件里没有，分布外）")
    i.add_argument("--actions", default=None, help='如 "D:8,DW:4,idle:4,atk:1,D:3"（语法见模块文档）或 [20,5] 的 .npy')
    i.add_argument("--data_root", default=DATA_ROOT)
    i.add_argument("--raw_root", default=RAW_ROOT)
    i.add_argument("--text_table", default=None, help="默认 <data_root>/text_table_gc25d.pt；'none' = 全部现场 UMT5 编码")
    e = ap.add_argument_group("批量评测")
    e.add_argument("--eval_val", type=int, default=0, help="评 N 条 val clip（按 scene 均分、确定性挑选）")
    e.add_argument("--clip_ids", default=None, help="逗号分隔的 clip 列表，或 @文件（每行一个）")
    e.add_argument("--out_dir", default=None, help="批量评测输出目录（每条 mp4/json + metrics.json）")
    s = ap.add_argument_group("采样")
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--shift", type=float, default=5.0)
    s.add_argument("--cfg", type=float, default=1.0, help="文本 CFG（负条件 NULL prompt）")
    s.add_argument("--sem_cfg", type=float, default=1.0, help="语义条件 CFG（负条件 ids 全 255）")
    s.add_argument("--sem_mode", choices=("cond", "unknown", "none"), default="cond")
    s.add_argument("--seed", type=int, default=0)
    o = ap.add_argument_group("输出")
    o.add_argument("--out", default=None, help="单条模式的输出 mp4（同名 _gen.mp4 / _sheet.jpg / .json）")
    o.add_argument("--overlay", action="store_true", help="多一个 生成帧 × ids 色图 的叠加面板")
    o.add_argument("--save_latents", action="store_true")
    o.add_argument("--no_recon_ceiling", dest="recon_ceiling", action="store_false",
                   help="不解码缓存 latent 算 VAE 重建上限（省 ~5 s/clip）")
    ap.add_argument("--device", default="cuda")
    return ap


def _resolve_clip_list(args) -> Optional[list[str]]:
    if args.eval_val and args.clip_ids:
        raise ValueError("--eval_val 与 --clip_ids 二选一")
    if args.clip_ids:
        if args.clip_ids.startswith("@"):
            with open(args.clip_ids[1:], encoding="utf-8") as fh:
                return [ln.strip() for ln in fh if ln.strip()]
        return [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    if args.eval_val:
        return select_val_clips(args.data_root, args.eval_val)
    return None


def main(argv=None, models: Optional[dict] = None) -> dict:
    """models：可传入已加载的 load_models(...) 结果（测试里复用，省一次加载）。"""
    ap = build_parser()
    args = ap.parse_args(argv)
    t_all = time.time()
    if args.forward == "arope":
        if args.sem_mode != "none":
            print("   --forward arope 没有语义项：sem_mode 置为 none")
            args.sem_mode = "none"
        if args.sem_cfg != 1.0:
            ap.error("--forward arope 没有语义项，不能用 --sem_cfg")
    if args.sem_mode != "cond" and args.sem_cfg != 1.0:
        ap.error("--sem_cfg 只在 --sem_mode cond 下有意义")
    if args.sem_mode == "none" and args.ckpt:
        print("   注意：--sem_mode none 配训练过的 ckpt 是分布外输入（训练从不去掉语义项；"
              "'无条件'在训练里是 ids 全 255，即 --sem_mode unknown）")
    clip_list = _resolve_clip_list(args)
    if clip_list is not None:
        if not args.out_dir:
            ap.error("批量评测需要 --out_dir")
        if args.first_frame or args.ids or args.actions or args.clip_id:
            ap.error("批量评测不接受 --clip_id / --first_frame / --ids / --actions")
        if len(set(clip_list)) != len(clip_list):
            ap.error("clip 列表里有重复")
        # 加载模型（~15 s）和逐条采样（~30 s/条）之前先查齐文件：prep 没跑完时立刻报错，而不是跑到一半才停
        missing = [cid for cid in clip_list if not all(os.path.isfile(p) for p in clip_files(args.data_root, cid))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} / {len(clip_list)} 条 clip 缺 latents/cond 文件（prep 未完成？）："
                                    f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
    else:
        if not args.out:
            ap.error("单条模式需要 --out")
    if clip_list is not None or args.clip_id:
        GEO.check_root_res(args.data_root, "--data_root")   # 数据根的 res 必须 = GC_RES
    text_table = (None if args.text_table == "none"
                  else args.text_table or os.path.join(args.data_root, "text_table_gc25d.pt"))

    if models is None:
        models = load_models(args.ckpt, args.base, args.device, args.forward, args.model_dir)
    else:
        assert models["info"]["ckpt"] == args.ckpt and models["info"]["base"] == bool(args.base) \
            and models["info"]["forward"] == args.forward, ("传入的 models 与参数不符", models["info"], args.ckpt, args.base, args.forward)
    enc = TextEncoder(text_table, args.device, args.model_dir)

    if clip_list is None:
        item = prepare_item(args, models, enc, args.clip_id)
        info = run_item(args, models, enc, item, args.out)
        if args.clip_id is not None:
            info["split"] = clip_index(args.data_root).get(args.clip_id, {}).get("split")
        info["total_sec"] = time.time() - t_all
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=1, default=_json_default)
        print(f"DONE {args.out} · {info['total_sec']:.0f}s")
        return info

    # ---- 批量评测 ----
    os.makedirs(args.out_dir, exist_ok=True)
    index = clip_index(args.data_root)
    splits = {cid: index.get(cid, {}).get("split") for cid in clip_list}
    not_val = [cid for cid in clip_list if splits[cid] != "val"]
    if not_val:
        print(f"   注意：{len(not_val)} / {len(clip_list)} 条 clip 不是 val（split = "
              f"{ {cid: splits[cid] for cid in not_val[:5]} }{'…' if len(not_val) > 5 else ''}）；metrics.json 里逐条记了 split")
    print(f"评测 {len(clip_list)} 条 clip → {args.out_dir}")
    per_clip, ceil = [], []
    try:
        for n, cid in enumerate(clip_list):
            print(f"[{n + 1}/{len(clip_list)}] {cid}")
            item = prepare_item(args, models, enc, cid)
            info = run_item(args, models, enc, item, os.path.join(args.out_dir, cid + ".mp4"))
            per_clip.append(dict(clip_id=cid, scene=info["meta"].get("scene"), split=splits[cid], **info["metrics"],
                                 sample_sec=info["sample_sec"], seg_consistency=info["seg_consistency"]))
            if "recon_ceiling" in info:
                ceil.append(info["recon_ceiling"])
    except BaseException as e:
        # 已跑完的 clip 写进 metrics_partial.json（逐条 json 本来就在），再原样抛出 —— 不写 metrics.json，不假装完整
        if per_clip:
            part = _summarize(args, clip_list, per_clip, ceil, splits, t_all)
            part["error"] = f"{type(e).__name__}: {e}（第 {len(per_clip) + 1} 条 {clip_list[len(per_clip)]}）"
            with open(os.path.join(args.out_dir, "metrics_partial.json"), "w", encoding="utf-8") as fh:
                json.dump(part, fh, ensure_ascii=False, indent=1, default=_json_default)
            print(f"   中断：已完成 {len(per_clip)} 条的汇总写到 {args.out_dir}/metrics_partial.json")
        raise
    summary = _summarize(args, clip_list, per_clip, ceil, splits, t_all)
    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=_json_default)
    agg, bal = summary["aggregate"], summary["aggregate_scene_balanced"]
    print(f"DONE 评测 {len(clip_list)} 条：PSNR {agg.get('psnr_mean', float('nan')):.2f} dB · SSIM {agg.get('ssim_mean', float('nan')):.4f}"
          f"（scene 等权 {_fmt(bal['psnr_mean'], '.2f')} / {_fmt(bal['ssim_mean'], '.4f')}，{bal['n_scenes']} 个 scene）"
          + (f"（VAE 上限 PSNR {summary['recon_ceiling']['psnr_mean']:.2f}）" if ceil else "") + f" → {out_json}")
    return summary


def _fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def _summarize(args, clip_list: list[str], per_clip: list[dict], ceil: list[dict], splits: dict, t_all: float) -> dict:
    """批量评测汇总。aggregate = 按 clip 平均；aggregate_scene_balanced = 先每个 scene 内平均、再对 scene 等权平均
    （--eval_val N 不是 scene 数的倍数时两者不同；scene 未知的 clip 各自算一组）。"""
    agg = MET.aggregate(per_clip)
    by_scene = {}
    for s in sorted({c["scene"] or c["clip_id"] for c in per_clip}):
        sub_ = [c for c in per_clip if (c["scene"] or c["clip_id"]) == s]
        a = MET.aggregate(sub_)
        by_scene[s] = {k: a[k] for k in ("n_clips", "psnr_mean", "ssim_mean") if k in a}
    bal = {"n_scenes": len(by_scene)}
    for k in ("psnr_mean", "ssim_mean"):
        vals = [v[k] for v in by_scene.values() if k in v]
        bal[k] = float(np.mean(vals)) if vals else None
    split_counts: dict = {}
    for c in per_clip:
        split_counts[str(c["split"])] = split_counts.get(str(c["split"]), 0) + 1
    return {
        "config": {k: getattr(args, k) for k in ("ckpt", "base", "forward", "steps", "shift", "cfg", "sem_cfg",
                                                  "sem_mode", "seed", "data_root", "eval_val")},
        "clip_ids": clip_list,
        "splits": {cid: splits[cid] for cid in clip_list},
        "split_counts": split_counts,
        "aggregate": agg,
        "aggregate_scene_balanced": bal,
        "by_scene": by_scene,
        "recon_ceiling": MET.aggregate(ceil) if ceil else None,
        "seg_consistency": None,     # HOOK：metrics.segmentation_consistency 接上分割器后在这里汇总 mIoU
        "per_clip": per_clip,
        "total_sec": time.time() - t_all,
    }


if __name__ == "__main__":
    main()
