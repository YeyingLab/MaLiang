"""蒸馏用的 gc25d DiT：`GcDiT` = ARoPE `distill.arope_dit.AropeDiT` + gc25d 的语义图条件。

为什么能直接继承 AropeDiT
------------------------
gc25d 的前向（model.gc_forward）= `arope_forward` 的普通路径（plain RoPE、无 offset、无掩码通道）+ patch embedding 后
加一项 `dit.gc_sem_cond(sem_ids)`。AropeDiT 的三种前向（并行双向 / 并行块因果 + TF 双流 / 流式单 cell 滑窗 KV）里，
patch embedding 之后恰好有一个加性入口 `_mask_embed(mask_input, ref)`（ARoPE 的 known 掩码通道），而且：
  * TF 双流时 clean 半流与 noisy 半流加的是**同一份**（clean cell j 与 noisy cell j 对应同一组帧 ⇒ 同一份语义图）；
  * 流式路径的 `mask_in_k` 只给当前 cell，提交趟也用同一份。
所以把这个入口换成语义条件，其余一行不改：mask_channel=False（不装 arope_mask_embedding），`mask_input` /
`mask_in_k` 这两个参数改装语义 id。offset_px / offset_win 恒为 None ⇒ 普通查表 RoPE（gc25d 没有世界坐标 RoPE）。

语义 id 的时间切片（Wan2.2 VAE 的 4× 时间压缩：cell 0 ← 帧 0，cell j ← 帧 4j−3 … 4j）
------------------------------------------------------------------------------------------
gc_sem_cond 先把帧按 VAE 的方式分组（帧 0 重复 4 次补成一组），卷积在时间方向的核都是 1 ⇒ cell j 的语义项只依赖
cell j 自己那几帧（tests/test_causal_cond.py 已经验过；这正是能蒸成因果模型的前提）。这里接受两种帧数：
  * T = 4f−3：一段以"首帧"开头的片子（帧 0 单独成组、重复 4 次）—— 并行路径、打分尾巴（锚点帧 + 后面 4(f−1) 帧）、
    以及流式路径的 cell 0（f=1, T=1）；
  * T = 4f：不带首帧的整组帧 —— 流式路径的 cell k≥1（f=1, T=4，帧 4k−3 … 4k）。
f=1 时两者是 1 帧 vs 4 帧，不会混。`cell_frames` / `window_frames` 给出切片。

v2 模块（adapter / 语义注意力）不支持：它们走 gc_block_forward 的另一条 block 路径，蒸馏只做 v1 配方（gc25d_v3v4）。

白模条件（cond_type="whitebox"）
--------------------------------------------------------------
`GcDiT(..., cond_type="whitebox")` 装 `dit.gc_wb_cond`（model.WhiteboxCondEncoder）而不是 gc_sem_cond，`mask_input` /
`mask_in_k` 里放白模 uint8 [b, T, H, W]（灰度，255 = UNKNOWN），其余一行不变：特征逐帧算、按同一套 `_group` 分组
（对通道数通用），所以 `cell_frames` / `window_frames` / 因果性 / KV 流式完全照旧。键数仍是 831（825 + 6 个
gc_wb_cond 键）；`build_gcdit` 不给 cond_type 时按 ckpt 的键名自动判断。默认 "ids"：行为与原来逐位相同。

彩色白模（cond_key="wb_rgb"-v3 / gc25d_pk3）
--------------------------------------------------------------------------
条件张量多一个末尾通道维：[b, T, H, W, 3]（三通道都是 255 = UNKNOWN，model.wb_pooled_features 直接吃）。时间维仍是 dim 1，
所以 `cell_frames` / `window_frames` / score_tail 的切片与拼接不变（cell_frames 的形状断言放宽到 4 维或末维 3 的 5 维）。
gc_wb_cond 的参数与灰度版同形 ⇒ ckpt 本身分不出灰度 / 彩色：npz 读哪个键**只**看谱系（gcdistill/lineage.py：ckpt meta 的
cond_key，旧 ckpt 没写 ⇒ "wb"），`cond_spec(cond_type, cond_key)` 不给 cond_key 时仍是旧默认（白模 "wb"）。
白模导出（save_gc，非 dry run）必须带 meta cond_key，否则报错 —— 下游 infer_pk / 下一阶段都按它读 npz。
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
import os
import sys

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

import torch
import torch.nn.functional as F
from einops import rearrange

AROPE_ROOT = os.environ.get("AROPE_ROOT", _mos.path.join(MALIANG_ROOT, "code", "arope"))   # maliang：vendored ARoPE
GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (AROPE_ROOT, GC_ROOT):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.append(_p)

import geometry as GEO  # noqa: E402  gc25d/geometry.py（GC_RES）
import model as GM  # noqa: E402  gc25d/model.py
from distill import runlib  # noqa: E402
from distill.arope_dit import MODEL_DIR, AropeDiT, unwrap  # noqa: E402

__all__ = ["GcDiT", "GC_DIT_KEYS", "SEM_KEYS", "WB_KEYS", "COND_KEYS", "cell_frames", "window_frames", "sem_embed",
           "wb_embed", "cond_embed", "load_gc_ckpt", "save_gc", "check_gc_ckpt", "build_gcdit", "unwrap", "MODEL_DIR",
           "CELL_FRAMES", "ckpt_cond_type", "ckpt_vocab", "cond_spec", "WB_COND_KEYS"]

CELL_FRAMES = 4                              # Wan2.2 VAE：一个 latent cell = 4 帧（cell 0 = 帧 0 单独）
SEM_KEYS = tuple(f"{GM.SEM_COND_NAME}.{m}.{p}" for m in ("proj_in", "conv_mid", "proj_out") for p in ("weight", "bias"))
WB_KEYS = tuple(f"{GM.WB_COND_NAME}.{m}.{p}" for m in ("proj_in", "conv_mid", "proj_out") for p in ("weight", "bias"))
COND_KEYS = {"ids": SEM_KEYS, "whitebox": WB_KEYS}
GC_DIT_KEYS = 831                            # 825 个 Wan DiT 键 + 6 个 gc_sem_cond 键（= outputs/gc25d_v3v4/step-*.safetensors）
                                             # 白模模型同样是 825 + 6 个 gc_wb_cond 键
GC_META = {"arm": "gc25d", "mask_channel": "False", "format": "gc25d", "sem_cond": GM.SEM_COND_NAME}


# ----------------------------------------------------------------------------
# 语义 id 的时间切片
# ----------------------------------------------------------------------------

def cell_frames(sem_ids: torch.Tensor, k: int) -> torch.Tensor:
    """流式路径 cell k 的语义帧：k=0 ⇒ 帧 0（[b,1,H,W]，GcDiT 里补成 4 份）；k≥1 ⇒ 帧 4k−3 … 4k（[b,4,H,W]）。
    彩色白模 [b,T,H,W,3] 同样按 dim 1 切（[b,1|4,H,W,3]）。"""
    assert sem_ids.dim() == 4 or (sem_ids.dim() == 5 and sem_ids.shape[-1] == 3), tuple(sem_ids.shape)
    if k == 0:
        out = sem_ids[:, :1]
    else:
        out = sem_ids[:, CELL_FRAMES * k - 3: CELL_FRAMES * k + 1]
    want = 1 if k == 0 else CELL_FRAMES
    assert out.shape[1] == want, f"cell {k} 要 {want} 帧语义图，sem_ids 只有 {sem_ids.shape[1]} 帧"
    return out


def window_frames(sem_ids: torch.Tensor, a: int, n_cells: int) -> torch.Tensor:
    """以绝对 cell a 为干净锚点（新的 cell 0）、共 n_cells 个 cell 的一段整片输入：帧 4a … 4(a+n_cells−1)。

    锚点 cell 用 4a 这一帧（cell a 的最后一帧，重复 4 次 —— 与训练时"cell 0 = 首帧重复 4 次"同构），
    窗口的 cell j≥1 用帧 4(a+j)−3 … 4(a+j)，与这些 cell 在整条 rollout 里的帧逐一对应。a=0 时就是 sem_ids[:, :4(n−1)+1]。"""
    lo, hi = CELL_FRAMES * a, CELL_FRAMES * (a + n_cells - 1) + 1
    assert hi <= sem_ids.shape[1], f"窗口 [cell {a}, +{n_cells}) 要到帧 {hi - 1}，sem_ids 只有 {sem_ids.shape[1]} 帧"
    return sem_ids[:, lo:hi]


def _group(pooled: torch.Tensor, f: int) -> torch.Tensor:
    """pooled one-hot [b,K,T,H/4,W/4] → 编码器输入 [b,640,f,H/16,W/16]。T=4f−3（首帧补 4 份）或 T=4f（整组）。"""
    T = pooled.shape[2]
    if T == CELL_FRAMES * f - 3:
        return GM.group_and_unshuffle(pooled)
    if T == CELL_FRAMES * f:
        return rearrange(pooled, "b k (f t) (h p) (w q) -> b (k t p q) f h w",
                         t=GM.T_GROUP, p=GM.UNSHUFFLE, q=GM.UNSHUFFLE)
    raise AssertionError(f"语义帧数 {T} 与 {f} 个 cell 对不上（应为 4f−3={4 * f - 3} 或 4f={4 * f}）")


def sem_embed(dit, sem_ids: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """语义项 [b, dim, f, h, w]（dtype = ref 的 dtype），f 取自 ref（patch embedding 的**输出**）。

    数值与 model.gc_forward 里的 `enc.forward_pooled(pooled)[0]` 逐位相同（同一组层、同一顺序；T=4f−3 时连分组函数都是同一个）。
    必须在根模块 forward 里调（FSDP 只在那里 all-gather 根上的 gc_sem_cond 参数）—— GcDiT._mask_embed 就是。"""
    enc = getattr(dit, GM.SEM_COND_NAME, None)
    if enc is None:
        raise ValueError(f"给了语义 id 但 dit 没有 {GM.SEM_COND_NAME}")
    if sem_ids.dtype.is_floating_point or sem_ids.dtype == torch.bool:
        raise TypeError(f"语义 id 必须是整数（uint8），得到 {sem_ids.dtype} —— 是不是把 ARoPE 的 known 掩码传进来了？")
    b, f, h, w = ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[4]
    assert sem_ids.shape[0] == b, (tuple(sem_ids.shape), tuple(ref.shape))
    pooled = GM.pooled_onehot(sem_ids.to(enc.proj_in.weight.device), torch.float32, enc.vocab)
    x = _group(pooled, f).to(enc.proj_in.weight.dtype)
    x = F.silu(enc.proj_in(x))
    x = F.silu(enc.conv_mid(x))
    s = enc.proj_out(x)
    assert tuple(s.shape) == tuple(ref.shape), f"语义项 {tuple(s.shape)} ≠ patch embedding {tuple(ref.shape)}"
    return s.to(ref.dtype)


def wb_embed(dit, wb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """白模项 [b, dim, f, h, w]（dtype = ref 的 dtype）。与 model.gc_forward 白模分支的 `enc.forward_pooled(...)[0]`
    同一组层、同一顺序（T=4f−3 时连分组函数都是同一个）。同 sem_embed，只能在根模块 forward 里调。"""
    enc = getattr(dit, GM.WB_COND_NAME, None)
    if enc is None:
        raise ValueError(f"给了白模但 dit 没有 {GM.WB_COND_NAME}")
    if wb.dtype != torch.uint8:
        raise TypeError(f"白模必须是 uint8 像素值，得到 {wb.dtype} —— 是不是把 ARoPE 的 known 掩码传进来了？")
    b, f = ref.shape[0], ref.shape[2]
    assert wb.shape[0] == b, (tuple(wb.shape), tuple(ref.shape))
    pooled = GM.wb_pooled_features(wb.to(enc.proj_in.weight.device), torch.float32)
    x = _group(pooled, f).to(enc.proj_in.weight.dtype)
    x = F.silu(enc.proj_in(x))
    x = F.silu(enc.conv_mid(x))
    s = enc.proj_out(x)
    assert tuple(s.shape) == tuple(ref.shape), f"白模项 {tuple(s.shape)} ≠ patch embedding {tuple(ref.shape)}"
    return s.to(ref.dtype)


def cond_embed(dit, cond: torch.Tensor, ref: torch.Tensor, cond_type: str = "ids") -> torch.Tensor:
    """按 cond_type 分发：ids ⇒ sem_embed，whitebox ⇒ wb_embed。"""
    if cond_type == "whitebox":
        return wb_embed(dit, cond, ref)
    if cond_type != "ids":
        raise ValueError(f"cond_type 只能是 {GM.COND_TYPES}，得到 {cond_type!r}")
    return sem_embed(dit, cond, ref)


WB_COND_KEYS = ("wb", "wb_rgb")              # 白模 npz 的两个键：灰度（跑酷数据的/ pk3 的旧定义）/ 彩色（pk3）


def cond_spec(cond_type: str, cond_key: str | None = None) -> dict:
    """蒸馏里每种条件的数据约定（与 gc25d/train.py 按 cond_type 取的默认值一致）：
    cond_key = cond/*.npz 读哪个键（白模数据里 "ids" 键是 UE 语义图，也是 uint8 —— 读错键不会报错，所以一律从这里取）；
    null_prompt = 文本 dropout / 无条件支的空串；mask_p = SFT 的 UNKNOWN 遮挡概率（第一人称没有世界平移 ⇒ 白模 0）；
    prefix = 条件模块的参数名前缀（GcDiT 包了一层 ⇒ 以 dit. 开头），给 train.sem_param_groups 分 cond_lr。
    cond_key=None ⇒ 旧默认（白模 "wb"）；白模可选 "wb_rgb"（彩色，pk3）。蒸馏 trainer 不直接用默认值，而是由
    gcdistill/lineage.py 按 ckpt 谱系定下 cond_key 再传进来。"""
    if cond_type == "whitebox":
        from text_pk import NULL_PROMPT as null
        key = "wb" if cond_key is None else cond_key
        if key not in WB_COND_KEYS:
            raise ValueError(f"白模的 cond_key 只能是 {WB_COND_KEYS}，得到 {cond_key!r}")
        return {"cond_key": key, "null_prompt": null, "mask_p": 0.0, "prefix": f"dit.{GM.WB_COND_NAME}."}
    if cond_type == "ids":
        from text import NULL_PROMPT as null
        if cond_key not in (None, "ids"):
            raise ValueError(f"语义 id 条件的 cond_key 只能是 'ids'，得到 {cond_key!r}")
        return {"cond_key": "ids", "null_prompt": null, "mask_p": 0.5, "prefix": f"dit.{GM.SEM_COND_NAME}."}
    raise ValueError(f"cond_type 只能是 {GM.COND_TYPES}，得到 {cond_type!r}")


def ckpt_cond_type(path: str) -> str:
    """只读 safetensors 头判断 ckpt 是哪种条件（有 gc_wb_cond.* ⇒ whitebox，否则 ids）。"""
    hdr, _ = runlib.read_ckpt_header(path)
    return GM.cond_type_from_keys(hdr.keys())


# ----------------------------------------------------------------------------
# GcDiT
# ----------------------------------------------------------------------------

def ckpt_vocab(path: str) -> str | None:
    """ckpt 头里 gc_sem_cond.proj_in 的形状 → 词表（ids / parts）；没有 gc_sem_cond ⇒ None。"""
    hdr, _ = runlib.read_ckpt_header(path)
    return GM.vocab_from_state(hdr)


class GcDiT(AropeDiT):
    """AropeDiT（mask_channel=False）+ gc_sem_cond。语义 id 走 `mask_input`（并行）/ `mask_in_k`（流式）。

    causal=False：双向教师 / critic（与 gc_forward 逐位相同，tests/test_gcdit.py::test_bitparity）；
    causal=True：因果学生（块因果掩码、TF 双流、滑窗 KV）。"""

    def __init__(self, dit=None, causal: bool = False, model_dir: str = MODEL_DIR, cond_type: str = "ids",
                 vocab: str | None = None, **kw):
        if cond_type not in GM.COND_TYPES:
            raise ValueError(f"cond_type 只能是 {GM.COND_TYPES}，得到 {cond_type!r}")
        if dit is None:
            dit = GM.load_dit(model_dir, device="cpu")
        if cond_type == "whitebox":
            GM.install_wb_cond(dit)         # 幂等；dit 上已有 gc_sem_cond 时报错
        else:
            if vocab is None:               # 已装 ⇒ 沿用它的词表；未装 ⇒ 默认 ids
                vocab = GM.sem_vocab(dit) if hasattr(dit, GM.SEM_COND_NAME) else GM.DEFAULT_VOCAB
            GM.install_sem_cond(dit, vocab)  # 幂等：已装（从 gc ckpt 加载过）就原样保留；词表不符报错
            enc = getattr(dit, GM.SEM_COND_NAME)
            if hasattr(enc, GM.ADAPTER_NAME) or hasattr(enc, GM.ATTN_GAIN_NAME):
                raise ValueError("GcDiT 只支持 v1 配方（patch embedding 上的语义项）：v2 的 adapter / 语义注意力走另一条 block 路径")
        super().__init__(dit=dit, mask_channel=False, causal=causal, **kw)
        self.cond_type = cond_type

    def _mask_embed(self, mask_input, ref: torch.Tensor):
        if mask_input is None:
            return None
        if getattr(self, "cond_type", "ids") == "whitebox":
            return wb_embed(self.dit, mask_input, ref)
        return sem_embed(self.dit, mask_input, ref)


def build_gcdit(ckpt: str | None, causal: bool, model_dir: str = MODEL_DIR, log=print, cond_type: str | None = None,
                **kw) -> GcDiT:
    """建 GcDiT 并（可选）装 gc25d 格式的 ckpt（831 键，strict）。ckpt=None ⇒ 基座 + 零初始化 gc_sem_cond。
    cond_type=None ⇒ 有 ckpt 时按它的键名判断（ckpt_cond_type），没有 ckpt 时 "ids"。"""
    if cond_type is None:
        cond_type = ckpt_cond_type(ckpt) if ckpt else "ids"
    if cond_type == "ids" and ckpt and "vocab" not in kw:
        kw["vocab"] = ckpt_vocab(ckpt) or GM.DEFAULT_VOCAB
    m = GcDiT(causal=causal, model_dir=model_dir, cond_type=cond_type, **kw)
    if ckpt:
        info = load_gc_ckpt(m, ckpt)
        log(f"[model] {'因果' if causal else '双向'} GcDiT ← {ckpt}（{info['n_tensors']} 个张量）")
    return m


def load_gc_ckpt(model, path: str) -> dict:
    """装 gc25d 格式的 safetensors（键不带 `dit.`），strict：缺任何一个键都报错（包括 gc_sem_cond.*）。"""
    info = runlib.load_dit_ckpt(model, path, allow_missing_mask=False)
    if info["n_tensors"] != GC_DIT_KEYS:
        raise RuntimeError(f"{path}: {info['n_tensors']} 个张量，gc25d v1 格式应为 {GC_DIT_KEYS}")
    return info


def save_gc(accel, model, path: str, metadata: dict | None = None, *, dry_run: bool = False, log=None) -> int:
    """runlib.save_dit 的 gc25d 版：831 键、不要求 ARoPE 掩码键、metadata 标成 gc25d 格式（gc25d/infer.py 直接能吃）。
    metadata 恒带 res（训练分辨率 = geometry.RES_TAG / GC_RES；调用方显式给了就用调用方的）：下游加载时按它拒收分辨率不符的 ckpt。
    白模（非 dry run）的 metadata 必须带 cond_key（wb / wb_rgb）：ckpt 本身分不出灰度 / 彩色，下游只能按它读 npz。"""
    meta = dict(GC_META, res=GEO.RES_TAG)
    if getattr(unwrap(model), "cond_type", "ids") == "whitebox":
        meta.update(sem_cond=GM.WB_COND_NAME, cond_type="whitebox")
        ck = (metadata or {}).get("cond_key")
        if not dry_run and ck not in WB_COND_KEYS:
            raise ValueError(f"{path}: 白模导出的 metadata 缺 / 错 cond_key（{ck!r}），应为 {WB_COND_KEYS} 之一（lineage.lineage_meta）")
    meta.update(metadata or {})
    return runlib.save_dit(accel, model, path, meta, expect_keys=(None if dry_run else GC_DIT_KEYS),
                           require_mask=False, log=log)


def check_gc_ckpt(path: str) -> dict:
    """导出后的秒级体检：831 个 BF16 张量、6 个 gc_sem_cond 键都在。"""
    res = runlib.check_ckpt_format(path, expect_keys=GC_DIT_KEYS, require_mask=False)
    hdr, _ = runlib.read_ckpt_header(path)
    keys = COND_KEYS[GM.cond_type_from_keys(hdr.keys())]
    miss = [k for k in keys if k not in hdr]
    if miss:
        raise RuntimeError(f"{path}: 缺条件键 {miss}")
    return res


# ----------------------------------------------------------------------------
# 不读基础权重的 DiT 骨架（maliang 蒸馏 / 部署用；纯新增，不改上面任何函数）
# ----------------------------------------------------------------------------

# Wan2.2-TI2V-5B 的 DiT 构造参数（= DiffSynth configs/model_configs.py 里 hash 1f5ab770… 那一条；同 codeworld/live.py）
WAN22_5B_DIT = dict(has_image_input=False, patch_size=[1, 2, 2], in_dim=48, dim=3072, ffn_dim=14336, freq_dim=256,
                    text_dim=4096, out_dim=48, num_heads=24, num_layers=30, eps=1e-06, seperated_timestep=True,
                    require_clip_embedding=False, require_vae_embedding=False, fuse_vae_embedding_in_latents=True)


def empty_wan_dit():
    """codeworld/live.py::empty_wan_dit 的同一份：参数只分配不初始化（bf16），之后必须用 load_gc_ckpt **严格**装满
    831 键 —— 蒸馏的四个模型都从完整的 gc25d ckpt 起步（SFT / stage1 / stage2 EMA），基础权重反正被整个覆盖，
    省掉每个模型一次 19 GB 的 fp32 分片读盘。与 GM.load_dit 起步再装同一份 ckpt 逐位相同（冒烟里核过）。"""
    import contextlib
    import torch.nn as nn
    from diffsynth.models.wan_video_dit import WanModel

    @contextlib.contextmanager
    def no_init():
        mods = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LayerNorm, nn.Embedding]
        saved = {m: m.reset_parameters for m in mods}
        for m in mods:
            m.reset_parameters = lambda self: None
        try:
            yield
        finally:
            for m, f in saved.items():
                m.reset_parameters = f
    with no_init():
        dit = WanModel(**WAN22_5B_DIT)
    return dit.to(torch.bfloat16).eval()
