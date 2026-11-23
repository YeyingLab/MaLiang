#!/usr/bin/env python3
"""三阶段蒸馏共用的 run 目录 / 日志 / checkpoint 约定。

    from distill import runlib
    run_dir = runlib.resolve_run_dir(args.output)
    log = runlib.TextTee(os.path.join(run_dir, "train.log"), enabled=accel.is_main_process)
    runlib.save_dit(accel, student, runlib.dit_path(run_dir, step), {"step": step})
    runlib.rotate(runlib.dit_glob(run_dir), args.keep_last_dits)

目录布局（三个 stage 共用，run_dir 默认落在 <repo>/outputs/distill/<name>）：

    <run_dir>/
        config.json                            本次 run 的完整参数快照
        progress.json                          {"step": N, ...}，续训只认它
        train.log                              TextTee 的人读日志（与 stdout 同步）
        train_log.csv                          CsvLog，一步一行，固定表头
        train_log.jsonl                        JsonlLogger，一步一行 JSON（可被别的机器 tail）
        dits/dit_step{N}.safetensors           学生原始权重，只为续训 —— 不要拿去部署
        ema/ema_step{N}.safetensors            EMA 影子权重 = stage2 / stage3 的部署权重
        critic/critic_step{N}.safetensors      stage3 的 fake-score critic，只为续训
        state_A/ state_B/ + state -> 软链      accelerate 完整分片存档（只有 stage1 用）

为什么 dits / ema / critic 都要 rotate：单个文件 10.0 GB。an earlier internal project 只 rotate 了
state_step*（infra/run.py:rotate_states 的注释明说不动 dits/），它的 v4 stage-1 run
光 dits/ 就吃掉 746 GB。这里 rotate() 对三种前缀一视同仁。

checkpoint 格式沿用 ARoPE 的（不是 an earlier internal project 的）：单个 safetensors，827 个 bf16
张量，DiT 键去掉 `dit.` 前缀（臂键保留 `arm.`），metadata 全是字符串，chmod 0644。
这样 tests/check_ckpt.py、actionrope/infer.py、eval_bg50.py 不用改一行就能吃蒸馏产物。

stage2-flowmap 的产物是这个格式的**唯一一处分叉**：`AropeDiT.install_flowmap()` 多挂 5 个
张量（`delta_embedding.{0,2}.{weight,bias}` + `delta_gate`），827 → **832**（AROPE_DIT_KEYS_FLOWMAP，
+20.5 MB）。格式本身一个字没变（单文件 / 全 bf16 / 字符串 metadata / 0644 / .tmp+rename），
只是键数变了 —— 于是上面那三个下游会**拒收**它。那是对的：flow map 要第二个时间步 r，拿旧的
renoise 采样器去采它不会报错，只会安静地产出另一个模型的结果。用 is_flowmap_ckpt() 分流。

★ save_dit 显式 cast 到 bf16 并断言导出 dtype。stage3 在 prepare 前把每个模型 .float()
（FSDP + MixedPrecision(bf16) 下不这么做就没有 fp32 主权重），而 accelerator.get_state_dict
在 FSDP 分支返回的是 FULL_STATE_DICT(offload_to_cpu=True, rank0_only=True) —— rank0 CPU 上的
**fp32**。不 cast 的话每个文件 20 GB（10 次保存 × 3 个文件 = 600 GB 而不是 300 GB），
而且 827-key / 10.0 GB 的格式承诺当场作废。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shutil
import struct
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "distill")

DIT_PREFIX = "dit."          # AropeDiT 里 WanModel 的属性名；导出时剥掉
ARM_PREFIX = "arm."          # baseline 动作臂；蒸馏不用臂，保留只为格式兼容
MASK_EMBEDDING_NAME = "arope_mask_embedding"
MASK_KEYS = (f"{MASK_EMBEDDING_NAME}.weight", f"{MASK_EMBEDDING_NAME}.bias")
AROPE_DIT_KEYS = 827         # 与 outputs/arope_10k_ga2/step-10000.safetensors 逐键一致
REF_CKPT = os.path.join(REPO_ROOT, "outputs", "arope_10k_ga2", "step-10000.safetensors")

# ---- flow map（stage2-flowmap）：第二时间步 r 的 delta 分支 ----------------------------------
# AropeDiT.install_flowmap() 把 delta_embedding（= time_embedding 的 deepcopy，读出层零初始化）和
# delta_gate（persistent buffer）挂在 **dit 上**，于是 strip_dit_prefix 剥掉 `dit.` 之后就是下面这
# 5 个平铺键 —— 格式一个字没变（单 safetensors / 全 bf16 / 字符串 metadata / 0644 / .tmp+rename），
# 只是键数 827 → 832（+20.5 MB，delta_embedding 是 10,229,760 个参数 = 5B 的 0.20 %）。
#
# ★ 没有"精确 827 键"的导出：residual 形式下 u(x,t,t) = head(E(t) + g·D(P(t)))，两条
#   Sequential(Linear, SiLU, Linear) 的和折不回一条。所以 actionrope/infer.py、tests/check_ckpt.py、
#   eval_bg50.py 会拒收 flow-map 权重 —— 这是 **对的**：拿旧的 renoise 采样器、不给 r 去采一个 flow map，
#   出来的东西看着像样但是另一个模型。导出时另外写 metadata flowmap=1 / flowmap_form / delta_gate，
#   下游按 metadata 选采样器，而不是按命令行开关。
DELTA_EMBEDDING_NAME = "delta_embedding"
DELTA_GATE_NAME = "delta_gate"
FLOWMAP_KEYS = (
    f"{DELTA_EMBEDDING_NAME}.0.weight", f"{DELTA_EMBEDDING_NAME}.0.bias",
    f"{DELTA_EMBEDDING_NAME}.2.weight", f"{DELTA_EMBEDDING_NAME}.2.bias",
    DELTA_GATE_NAME,
)
AROPE_DIT_KEYS_FLOWMAP = AROPE_DIT_KEYS + len(FLOWMAP_KEYS)   # 832


# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------

def resolve_run_dir(out: str) -> str:
    """`out` 可以是绝对路径、相对仓库根的路径、或一个裸名字（落在 outputs/distill/ 下）。"""
    if not out or not str(out).strip():
        raise ValueError("run dir 不能为空")
    out = os.path.expanduser(str(out))
    if os.path.isabs(out):
        return os.path.normpath(out)
    if os.sep in out:
        return os.path.normpath(os.path.join(REPO_ROOT, out))
    return os.path.join(OUTPUT_ROOT, out)


def dits_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "dits")


def ema_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "ema")


def critic_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "critic")


def dit_path(run_dir: str, step: int) -> str:
    """学生原始权重：只为续训。部署一律用 ema_path()。"""
    return os.path.join(dits_dir(run_dir), f"dit_step{int(step)}.safetensors")


def ema_path(run_dir: str, step: int) -> str:
    """EMA 影子权重 = stage2 / stage3 的部署权重。"""
    return os.path.join(ema_dir(run_dir), f"ema_step{int(step)}.safetensors")


def critic_path(run_dir: str, step: int) -> str:
    return os.path.join(critic_dir(run_dir), f"critic_step{int(step)}.safetensors")


def dit_glob(run_dir: str) -> str:
    return os.path.join(dits_dir(run_dir), "dit_step*.safetensors")


def ema_glob(run_dir: str) -> str:
    return os.path.join(ema_dir(run_dir), "ema_step*.safetensors")


def critic_glob(run_dir: str) -> str:
    return os.path.join(critic_dir(run_dir), "critic_step*.safetensors")


_STEP_RE = re.compile(r"step(\d+)")


def _step_of(path: str) -> int | None:
    """从 dit_step500.safetensors / ema_step500.safetensors / state_step500 里取出 500。"""
    m = _STEP_RE.search(os.path.basename(os.path.normpath(path)))
    return int(m.group(1)) if m else None


def list_ckpts(pattern: str) -> list[tuple[int, str]]:
    """[(step, path), ...] 按 step 升序。名字里解析不出 step 的条目直接忽略（也就永远不会被 rotate 删）。"""
    out = [(s, p) for p in glob.glob(pattern) if (s := _step_of(p)) is not None]
    return sorted(out)


def latest_ckpt(pattern: str) -> tuple[int, str] | None:
    c = list_ckpts(pattern)
    return c[-1] if c else None


def rotate(pattern: str, keep: int) -> list[str]:
    """按 step 排序，删掉最旧的，保留最新 keep 个。dits / emas / critics / state_step* 通用。

    keep <= 0 ⇒ 什么都不删（不是"全删"，别把唯一的续训点抹了）。解析不出 step 的路径不碰。
    返回被删掉的路径。目录用 rmtree，文件用 remove。"""
    keep = int(keep)
    if keep <= 0:
        return []
    removed = []
    for _, p in list_ckpts(pattern)[:-keep]:
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            removed.append(p)
        except OSError:
            pass
    return removed


def rotate_run(run_dir: str, keep_dits: int = 0, keep_emas: int = 0, keep_critics: int = 0) -> dict:
    """一次把三种权重都 rotate 掉。默认 0 = 不动（显式传才删）。"""
    return {"dits": rotate(dit_glob(run_dir), keep_dits),
            "emas": rotate(ema_glob(run_dir), keep_emas),
            "critics": rotate(critic_glob(run_dir), keep_critics)}


def disk_free_gb(path: str = REPO_ROOT) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("nan")


def warn_if_low_disk(path: str = REPO_ROOT, min_gb: float = 200.0) -> str | None:
    free = disk_free_gb(path)
    if free == free and free < min_gb:  # 非 NaN 且不够
        return (f"磁盘告急：{path} 只剩 {free:.0f} GB（< {min_gb:.0f} GB）。一个 dit/ema/critic 是 10.0 GB，"
                f"一个 accelerate state 槽约 60 GB；调小 --save_every 或 --keep_last_*。")
    return None


# ----------------------------------------------------------------------------
# progress.json / config.json
# ----------------------------------------------------------------------------

def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def progress_path(run_dir: str) -> str:
    return os.path.join(run_dir, "progress.json")


def write_progress(run_dir: str, step: int, **extra) -> str:
    """续训只认这个文件。原子写：保存大权重时崩掉也不会留下半个 progress.json。"""
    row = {"step": int(step), "wall": time.time(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    p = progress_path(run_dir)
    _atomic_write_text(p, json.dumps(row, ensure_ascii=False, default=str))
    return p


def read_progress(run_dir: str) -> dict | None:
    p = progress_path(run_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def progress_step(run_dir: str, default: int = 0) -> int:
    pr = read_progress(run_dir)
    return int(pr["step"]) if pr and "step" in pr else int(default)


def save_config(run_dir: str, cfg: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    p = os.path.join(run_dir, "config.json")
    _atomic_write_text(p, json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
    return p


# ----------------------------------------------------------------------------
# checkpoint：导出
# ----------------------------------------------------------------------------

def _unwrap(model):
    """剥掉 DDP / DeepSpeedEngine / EmaShadow 之类的外壳，拿到裸 module。"""
    seen = 0
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module) and seen < 8:
        model = model.module
        seen += 1
    return model


def _gather_state_dict(accel, model) -> dict:
    """accel=None ⇒ 单进程，直接 state_dict()。否则走 accelerator.get_state_dict（集合操作，
    每个 rank 都必须进来；FSDP 分支是 rank0_only，非主进程拿到空 dict）。

    非 nn.Module（distill.ema.EmaShadow）走本地 state_dict：它没进过 prepare，权重在每个 rank 上
    都是整份的，get_state_dict 的 FSDP 分支反而会拿它当 FSDP 模块去遍历。"""
    if accel is None or not isinstance(model, nn.Module):
        return _unwrap(model).state_dict()
    return accel.get_state_dict(model)


def strip_dit_prefix(state_dict: dict) -> tuple[dict, list[str]]:
    """AropeDiT 包装的 state_dict → ARoPE 的导出键集。

    有 `dit.` 键 ⇒ 剥前缀，`arm.` 键原样保留，其余（包装层自己的 buffer 之类）丢掉并回报；
    没有 `dit.` 键 ⇒ 传进来的本来就是裸 WanModel，原样返回。"""
    if not any(k.startswith(DIT_PREFIX) for k in state_dict):
        return dict(state_dict), []
    out, dropped = {}, []
    for k, v in state_dict.items():
        if k.startswith(DIT_PREFIX):
            out[k[len(DIT_PREFIX):]] = v
        elif k.startswith(ARM_PREFIX):
            out[k] = v
        else:
            dropped.append(k)
    return out, dropped


def save_dit(accel, model, path: str, metadata: dict | None = None, *,
             expect_keys: int | None = AROPE_DIT_KEYS, require_mask: bool = True,
             dtype: torch.dtype = torch.bfloat16, log=None, post=None) -> int:
    """把（可能被 ZeRO-2 / FSDP 包着的）模型导出成 ARoPE 格式的 safetensors，返回字节数。

    每个 rank 都要调用（get_state_dict 是集合操作），只有主进程落盘。

    ★ 显式 cast 到 bf16 并断言导出 dtype。FSDP 下 accelerator.get_state_dict 返回的是 rank0 CPU 上的
    fp32 全量权重（stage3 在 prepare 前 .float() 过每个模型），不 cast 文件就是 20 GB，827-key /
    10.0 GB 的格式承诺当场作废，infer.py / check_ckpt.py 的 dtype 断言也会跟着变。

    expect_keys / require_mask 是格式闸门：默认按 827 键 + arope_mask_embedding.{weight,bias} 校，
    对不上直接炸（少两个 mask 键 = 模型是拿 mask_channel=False 建的，装回 infer.py 会静默走错分支）。
    小模型自测传 expect_keys=<n>, require_mask=False。"""
    if accel is not None:
        accel.wait_for_everyone()
    sd = _gather_state_dict(accel, model)
    # post：gather 之后、格式闸门之前的一次改写。目前只有 LoRA 用它把 B@A 折回 base.weight——
    # 不折的话 state_dict 里是 `...base.weight` + `lora_A` + `lora_B`，expect_keys 当场拦下。
    # 每个 rank 都会走到这里，但只有主进程的 sd 非空（见 _gather_state_dict）。
    if post is not None and sd:
        sd = post(sd)
    is_main = True if accel is None else bool(accel.is_main_process)
    size = 0
    if is_main and sd:
        out, dropped = strip_dit_prefix(sd)
        out = {k: v.detach().to(device="cpu", dtype=dtype).contiguous() for k, v in out.items()}
        bad = sorted({str(v.dtype) for v in out.values()} - {str(dtype)})
        if bad:
            raise RuntimeError(f"导出 dtype 不是 {dtype}：{bad}")
        if expect_keys is not None and len(out) != expect_keys:
            raise RuntimeError(f"{path}: 导出 {len(out)} 个张量，期望 {expect_keys} 个"
                               f"（丢弃的非 DiT 键 {dropped[:5]}…{len(dropped)}）")
        if require_mask:
            miss = [k for k in MASK_KEYS if k not in out]
            if miss:
                raise RuntimeError(f"{path}: 缺 {miss} —— 模型是拿 mask_channel=False 建的？")
        meta = {"arm": "arope", "mask_channel": "True"}
        meta.update({k: str(v) for k, v in (metadata or {}).items()})
        meta.update({"n_tensors": str(len(out)), "dtype": str(dtype).replace("torch.", ""),
                     "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp"                       # 先写临时文件再 rename：写到一半崩掉不会留下半个 ckpt
        save_file(out, tmp, metadata=meta)
        os.chmod(tmp, 0o644)                      # safetensors 按 umask 写成 0600，推理可能由别的账号跑
        os.replace(tmp, path)
        size = os.path.getsize(path)
        msg = f"[save] {path}  {len(out)} 个张量  {size / 1e9:.2f} GB"
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)
        del out
    del sd
    if accel is not None:
        accel.wait_for_everyone()
    return size


# ----------------------------------------------------------------------------
# checkpoint：装载 / 体检
# ----------------------------------------------------------------------------

def load_dit_ckpt(model, path: str, *, assign: bool = False, allow_missing_mask: bool = True,
                  allow_missing: tuple | list = ()) -> dict:
    """把 ARoPE 格式的 safetensors 装进 AropeDiT（键带 `dit.`）或裸 WanModel（键不带）。

    strict —— 默认唯一放行的缺失是 arope_mask_embedding.{weight,bias}（install_arope 已经零初始化，
    从没装过 mask 通道的 ckpt 起步时就缺这两个键）。an earlier internal project 的 load_dit_ckpt 对任何 missing 都抛，
    照抄会拒收每一个 ARoPE checkpoint。

    allow_missing：额外放行的键（去掉 `dit.` 前缀的名字，也接受带前缀的写法）。**唯一** 的正当用法是
    "先 install_flowmap() 再从 827 键的 stage1b 装载"这一个调用点，传 `allow_missing=FLOWMAP_KEYS`；
    别把它设成默认，否则任何一个真的漏键都会静默通过。推荐的顺序反过来：先 strict 装 827、再 install
    （见 AropeDiT.install_flowmap 的文档），那样一个例外都不需要。

    assign=False（默认）：copy_ 进现有参数，保持模型自己的 dtype —— stage3 在 .float() 之后装 bf16
    ckpt 就靠这个，assign=True 会把参数直接换成 bf16 张量，fp32 主权重当场消失。"""
    sd = load_file(path)
    tgt = _unwrap(model)
    tgt_keys = set(tgt.state_dict().keys())
    prefix = DIT_PREFIX if any(k.startswith(DIT_PREFIX) for k in tgt_keys) else ""
    if prefix:
        sd = {(k if k.startswith(prefix) else prefix + k): v for k, v in sd.items()}
    missing, unexpected = tgt.load_state_dict(sd, strict=False, assign=assign)
    allowed = {prefix + k for k in MASK_KEYS} if allow_missing_mask else set()
    allowed |= {(k if prefix and k.startswith(prefix) else prefix + k) for k in allow_missing}
    bad = [k for k in missing if k not in allowed]
    if bad or unexpected:
        raise RuntimeError(f"{path} 与模型键集不符：缺 {bad[:5]}…({len(bad)})，多 {list(unexpected)[:5]}…({len(unexpected)})")
    return {"path": path, "n_tensors": len(sd), "prefix": prefix,
            "missing": list(missing), "unexpected": list(unexpected)}


def read_ckpt_header(path: str) -> tuple[dict, dict]:
    """只读 safetensors 头（不碰权重数据）→ ({key: {dtype, shape, data_offsets}}, metadata)。"""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    meta = hdr.pop("__metadata__", {}) or {}
    return hdr, meta


def is_flowmap_ckpt(path: str) -> bool:
    """这份 ckpt 是不是 flow map（832 键）。**看张量键，不看 metadata** —— metadata 是人写的字符串，
    张量在不在是事实。下游（rollout_eval）应当据此选采样器：拿 renoise 采样器去采 flow map 不会报错，
    只会安静地产出另一个模型的结果。"""
    hdr, _ = read_ckpt_header(path)
    return all(k in hdr for k in FLOWMAP_KEYS)


def check_ckpt_format(path: str, expect_keys: int | None = AROPE_DIT_KEYS,
                      dtype: str = "BF16", require_mask: bool = True) -> dict:
    """导出后的秒级体检（不加载权重）：张量数 / dtype / mask 键 / metadata 全是字符串。

    正品参照：outputs/arope_10k_ga2/step-10000.safetensors —— 827 个 BF16 张量，9,999,690,792 B。"""
    hdr, meta = read_ckpt_header(path)
    dts = sorted({v["dtype"] for v in hdr.values()})
    res = {"path": path, "n_tensors": len(hdr), "dtypes": dts, "size": os.path.getsize(path),
           "metadata": meta, "mask_keys": sorted(k for k in hdr if k.startswith(MASK_EMBEDDING_NAME))}
    if expect_keys is not None and len(hdr) != expect_keys:
        raise RuntimeError(f"{path}: {len(hdr)} 个张量，期望 {expect_keys}")
    if dtype is not None and dts != [dtype]:
        raise RuntimeError(f"{path}: dtype {dts}，期望 ['{dtype}']")
    if require_mask and set(res["mask_keys"]) != set(MASK_KEYS):
        raise RuntimeError(f"{path}: mask 键 {res['mask_keys']}，期望 {sorted(MASK_KEYS)}")
    bad = {k: type(v).__name__ for k, v in meta.items() if not isinstance(v, str)}
    if bad:
        raise RuntimeError(f"{path}: metadata 必须全是字符串，{bad}")
    return res


# ----------------------------------------------------------------------------
# accelerate 完整存档（只有 stage1 用）
# ----------------------------------------------------------------------------

STATE_SLOTS = ("state_A", "state_B")


def save_state_rotating(accel, run_dir: str, trainer_state: dict) -> str:
    """accelerator.save_state 到两个槽之一，<run_dir>/state 软链指向最新完整的那个。

    原地覆盖唯一的 state/ 目录时，保存中途崩溃会把唯一的续训点也毁掉；这里总是写到「不是当前
    state 指向」的那个槽，写完再 rename 换软链，任一时刻 state 都指向一份完整存档。不删任何目录，
    磁盘占两份（4 卡 ZeRO-2 约 2×60 GB）。--resume <run_dir>/state 照常。"""
    link = os.path.join(run_dir, "state")
    cur = os.path.basename(os.path.realpath(link)) if os.path.islink(link) else None
    slot = STATE_SLOTS[1] if cur == STATE_SLOTS[0] else STATE_SLOTS[0]
    state_dir = os.path.join(run_dir, slot)
    accel.save_state(state_dir)
    if accel.is_main_process:
        with open(os.path.join(state_dir, "trainer_state.json"), "w", encoding="utf-8") as fh:
            json.dump(trainer_state, fh, ensure_ascii=False, default=str)
        tmp = link + ".tmp"
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(slot, tmp)
        os.replace(tmp, link)                     # 只替换软链本身，不动任何目录
        print(f"[save] accelerator.save_state → {state_dir}（{link} → {slot}）", flush=True)
    accel.wait_for_everyone()
    return state_dir


# ----------------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------------

class TextTee:
    """打到 stdout，同时带时间戳追加进 <run_dir>/train.log。

    enabled=False（非主进程）⇒ 完全静默，也不开文件句柄。"""

    def __init__(self, path: str, enabled: bool = True):
        self.path, self.enabled, self._f = path, bool(enabled), None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, *args):
        if not self.enabled:
            return
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self._f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None


class CsvLog:
    """固定表头的追加式 CSV（actionrope/train.py:328 的写法）。已存在且非空就不再写表头。"""

    def __init__(self, path: str, fields: list[str], enabled: bool = True):
        self.enabled, self.f, self.w = bool(enabled), None, None
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        self.f = open(path, "a", encoding="utf-8", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        if not exists:
            self.w.writeheader()
            self.f.flush()

    def write(self, row: dict):
        if not self.enabled:
            return
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        if self.f is not None:
            try:
                self.f.close()
            finally:
                self.f = None


class JsonlLogger:
    """一行一个 JSON 的追加式指标日志，每行都 flush，别的机器可以直接 tail。"""

    def __init__(self, path: str, enabled: bool = True):
        self.path, self.enabled, self._f = path, bool(enabled), None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, row: dict):
        if not self.enabled:
            return
        row = dict(row)
        row.setdefault("wall", time.time())
        self._f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None
