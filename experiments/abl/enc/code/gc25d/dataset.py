"""gc25d 训练数据集：预计算 latent + 逐像素 parts id 条件 + 文本表查表 + 三种 dropout。

照 ARoPE `actionrope/dataset.py`（AropeLatentDataset / arope_collate）改：没有 sidecar / 偏移 / 动作臂，
多一路 `sem_ids`；不 import actionrope.dataset（它会拖进 baseline）。

一个样本 = 一个 clip（81 帧 ⇒ 21 个 latent cell）：

    input_latents  bf16 [48, 21, H/16, W/16]   latents/<clip_id>.pt 的 "latent"（832×480：[48,21,30,52]）
    context        bf16 [21, 512, 4096]        逐 cell 的 UMT5 文本嵌入（查 text_table_gc25d.pt，不挂 T5）
    context_ids    int64 [21]                  文本串的全局 id = 在 sorted(text_table) 里的下标（= infer._KEY_TO_ID）
    sem_ids        uint8 [81, H, W]            cond/<clip_id>.npz 的 "ids"（parts 词表 raw id）；vocab="ids" ⇒ model.collapse_parts
（H × W = geometry 的 DST_H × DST_W，由 env GC_RES 定；数据根的 index/prep_args.json "res" 必须与之相同，否则构造时报错）
    clip_id        str
    dropped        {"text": bool, "cond": bool, "mask": float}   本样本的抽签结果；mask = 被 UNKNOWN 遮挡改成 255 的像素占比

三种 dropout
---------------------------------------------------------------------
* text  p=0.1：21 个 cell 全换成 NULL_PROMPT（= infer 的文本 CFG 负支）。
* cond  p=0.05：ids 全 255（= infer 的语义 CFG 负支 / "无条件"），与 text 独立（同时去掉 0.5 %）。训练从不去掉语义项本身。
* mask  p=0.5（只在 cond 没被 drop 时）：世界锚定的 UNKNOWN 遮挡，见下。
val split 一律不 dropout（三个概率强制为 0）。

随机数按 (seed, epoch, index) 定：`random.Random(f"{seed}:{epoch}:{index}")`，同一样本同一 epoch 每次抽到的完全一样，
与 worker 数 / rank 无关；epoch 由 set_epoch 设（训练脚本每个 epoch 开头显式调用 —— 用 SeedableRandomSampler 时
accelerate 的 DataLoaderShard.set_epoch 只转给 sampler、不转给 dataset；worker 不能是 persistent，
否则 fork 出去的副本看不到新 epoch —— train.py 里 persistent_workers=False）。

世界锚定的 UNKNOWN 遮挡
-----------------------------------------------------------------------------------
模仿代码世界里"没看过的画布 = 255、随相机滑动"：
  1. 1 个（p .7，q ~ U(0.05, 0.40)）或 2 个（p .3，各 q ~ U(0.05, 0.25)）随机方向 θ ~ U[0, 2π)（世界 XY 平面）；
  2. 地面 = 水平面 Z = meta.ground_z[0]（帧 0 玩家胶囊最低点）；像素的地面点 = 该像素射线与地面的交点（capsules.ground_points），
     在 4×4 池化格上算（格 (i, j) 取 dst 像素 (4j+2, 4i+2)：ground_points 只收整数 dst 像素，几何中心 (4j+1.5, 4i+1.5)
     落在像素缝上，取右下那个，偏半个 dst 像素 < 1 个源像素），再最近邻 ×4 上采样到 H×W；
  3. 每个方向的阈值 τ = max_t（帧 t 静态格点地面点在 θ 上投影的 (1−q) 分位数），t 取标定帧（每 4 帧 + 末帧）
     ⇒ 任一帧的静态像素至多 ≈ q 在"界外"，覆盖最大的那帧恰 ≈ q。最初只按帧 0 标定，相机跟着玩家走向界外时后段帧
     会几乎全遮（review：帧 80 覆盖 p90 = 1.0），远超部署时画布 8–40 % 没见过的量级，故改；
  4. 每一帧：静态像素的地面点（用该帧自己的相机）落在任一半平面界外（投影 > τ）⇒ 255；地面点 NaN（射线不交地面）⇒ 也 255。
     角色像素（10..21）永不遮挡。
半平面固定在世界里 ⇒ 同一世界点在各帧里同遮或同不遮（相机走 ⇒ 遮挡区在画面上滑动）；判定只用该帧自己的相机与 ids
⇒ 逐帧（cell-local），因果蒸馏也能原样用。
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import capsules as CAP
import geometry as GEO
import model as GM
from text import NULL_PROMPT

# 的形状契约（分辨率取自 geometry / GC_RES），__getitem__ 里逐项核对
LATENT_SHAPE = (48, 21, GEO.LAT_H, GEO.LAT_W)        # 832×480 ⇒ (48, 21, 30, 52)
CELLS = 21
CONTEXT_SHAPE = (512, 4096)
N_FRAMES = 81
IDS_SHAPE = (N_FRAMES, GEO.DST_H, GEO.DST_W)          # 832×480 ⇒ (81, 480, 832)
UNKNOWN_ID = GM.UNKNOWN_ID
STATIC_MAX = 6                      # raw 0..6 = 静态语义；10..21 = 角色部位；255 = UNKNOWN
SPLITS = ("train", "val")
TEXT_TABLE_NAME = "text_table_gc25d.pt"

# UNKNOWN 遮挡
MASK_PLANES_P2 = 0.3                # 2 个半平面的概率（否则 1 个）
MASK_Q_RANGE = (0.05, 0.40)         # 1 个半平面时，覆盖最大的那帧遮住的静态像素占比
MASK_Q_RANGE_2 = (0.05, 0.25)       # 2 个半平面时，各自的 q
MASK_CAL_STRIDE = 4                 # 标定帧步长（另加末帧）
GRID = GM.POOL                      # 4×4 池化格
GRID_H, GRID_W = IDS_SHAPE[1] // GRID, IDS_SHAPE[2] // GRID      # = GEO.POOL_H, POOL_W（832×480 ⇒ 120, 208）
_GY, _GX = np.meshgrid(np.arange(GRID_H) * GRID + GRID // 2, np.arange(GRID_W) * GRID + GRID // 2, indexing="ij")


def load_text_table(text_table) -> dict[str, torch.Tensor]:
    if isinstance(text_table, dict):
        return text_table
    return torch.load(text_table, map_location="cpu", weights_only=False)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ----------------------------------------------------------------------------
# 世界锚定的 UNKNOWN 遮挡
# ----------------------------------------------------------------------------

def grid_ground_points(world_to_clip, world_to_view, z_ground: float) -> np.ndarray:
    """逐帧相机 [T,16] → 4×4 池化格中心像素的地面点（世界 XY）float64 [T, H/4, W/4, 2]；不交地面 ⇒ NaN。"""
    w2c = np.asarray(world_to_clip, np.float64).reshape(-1, 16)
    w2v = np.asarray(world_to_view, np.float64).reshape(-1, 16)
    assert len(w2c) == len(w2v), (w2c.shape, w2v.shape)
    xs, ys = _GX.ravel(), _GY.ravel()
    out = np.empty((len(w2c), GRID_H, GRID_W, 2), np.float64)
    for t in range(len(w2c)):
        out[t] = CAP.ground_points(w2c[t], w2v[t], xs, ys, float(z_ground)).reshape(GRID_H, GRID_W, 2)
    return out


def upsample_grid(a: np.ndarray) -> np.ndarray:
    """[..., H/4, W/4] → [..., H, W]，最近邻 ×4（每格 16 个像素同值）。"""
    return np.repeat(np.repeat(a, GRID, axis=-2), GRID, axis=-1)


def calib_frames(T: int) -> list[int]:
    return sorted(set(range(0, T, MASK_CAL_STRIDE)) | {T - 1})


def sample_planes(rng: random.Random, g: np.ndarray, static_grid: np.ndarray) -> list[dict]:
    """抽 1 / 2 个世界 XY 半平面 {theta, q, d, tau}：投影 g·d > tau 为"界外"；tau = 各标定帧 (1−q) 分位数的最大值
    ⇒ 每帧静态像素至多 ≈ q 在界外，最大的那帧恰 ≈ q。

    g [T,H/4,W/4,2]（逐帧格地面点），static_grid bool [T,H/4,W/4]（格中心像素是静态类）。没有可用的标定帧 ⇒ []。"""
    n = 2 if rng.random() < MASK_PLANES_P2 else 1
    qr = MASK_Q_RANGE if n == 1 else MASK_Q_RANGE_2
    planes = []
    for _ in range(n):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        q = rng.uniform(*qr)
        d = np.array([math.cos(theta), math.sin(theta)])
        taus = []
        for t in calib_frames(len(g)):
            s = (g[t] @ d)[static_grid[t]]
            s = s[np.isfinite(s)]
            if s.size:
                taus.append(float(np.quantile(s, 1.0 - q)))
        if not taus:
            return []
        planes.append({"theta": theta, "q": q, "d": d, "tau": max(taus)})
    return planes


def planes_mask(ids: np.ndarray, g: np.ndarray, planes: list[dict]) -> np.ndarray:
    """ids uint8 [T,H,W] + 逐帧格地面点 g [T,H/4,W/4,2] → 要改成 255 的像素 bool [T,H,W]：
    静态像素（raw ≤ 6）且（地面点 NaN 或落在任一半平面界外）。"""
    beyond = np.isnan(g).any(-1)                               # [T,H/4,W/4]
    for p in planes:
        with np.errstate(invalid="ignore"):
            beyond |= (g @ p["d"]) > p["tau"]
    return upsample_grid(beyond) & (ids <= STATIC_MAX)


def apply_unknown_mask(ids: np.ndarray, meta, rng: random.Random) -> dict:
    """原地把 ids [81,H,W] 的世界锚定遮挡区改成 255，返回 {frac, planes, z_ground}（frac = 全部像素里被改的占比）。

    meta：带 world_to_clip / world_to_view [F,16]、ground_z [F] 的 npz / dict（F ≥ 81，取前 81 帧）。"""
    T = ids.shape[0]
    gz = np.asarray(meta["ground_z"], np.float64)[:T]
    z = float(gz[0]) if np.isfinite(gz[0]) else (float(np.nanmedian(gz)) if np.isfinite(gz).any() else float("nan"))
    if not math.isfinite(z):                                    # 整段没有玩家胶囊（不该发生）：不遮
        return {"frac": 0.0, "planes": [], "z_ground": z}
    g = grid_ground_points(np.asarray(meta["world_to_clip"])[:T], np.asarray(meta["world_to_view"])[:T], z)
    planes = sample_planes(rng, g, ids[:, _GY, _GX] <= STATIC_MAX)
    if not planes:
        return {"frac": 0.0, "planes": [], "z_ground": z}
    m = planes_mask(ids, g, planes)
    ids[m] = UNKNOWN_ID
    return {"frac": float(m.mean()), "planes": planes, "z_ground": z}


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------

class GcLatentDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        text_table=None,
        vocab: str = "parts",
        text_dropout: float = 0.1,
        cond_dropout: float = 0.05,
        mask_p: float = 0.5,
        limit: int | None = None,
        seed: int = 0,
        include_dropped: bool = False,
        verbose: bool = False,
    ):
        if split not in SPLITS:
            raise ValueError(f"split 只能是 {SPLITS}，得到 {split!r}")
        GM.vocab_spec(vocab)
        self.res = GEO.check_root_res(root)                    # index/prep_args.json 的 res 必须 = GC_RES（缺失 ⇒ 832x480）
        self.root, self.split, self.vocab = root, split, vocab
        if split == "val":                                      # 评测口径：不 dropout、不遮挡
            text_dropout = cond_dropout = mask_p = 0.0
        self.text_dropout, self.cond_dropout, self.mask_p = float(text_dropout), float(cond_dropout), float(mask_p)
        self.seed, self.epoch = int(seed), 0
        self.text_table = load_text_table(os.path.join(root, TEXT_TABLE_NAME) if text_table is None else text_table)
        if NULL_PROMPT not in self.text_table:
            raise KeyError(f"文本表里没有 NULL_PROMPT {NULL_PROMPT!r}")
        # 文本串的全局 id：按键排序后的下标，跨样本、跨 batch 都稳定，相同串 ⇒ 相同 id（与 infer._KEY_TO_ID 一致）
        self.key_to_id = {k: i for i, k in enumerate(sorted(self.text_table))}

        rows = read_jsonl(os.path.join(root, "index", "clips.jsonl"))
        in_split = [r for r in rows if r["split"] == split]
        self.n_total = len(in_split)
        self.n_dropped = sum(1 for r in in_split if r.get("drop"))
        kept = [r for r in in_split if include_dropped or not r.get("drop")]
        if limit is not None and limit < len(kept):
            # smoke / val 子集：按 seed 打乱后取前 limit 个再排序（同 AropeLatentDataset）
            rng = random.Random(seed)
            rng.shuffle(kept)
            kept = sorted(kept[:limit], key=lambda r: r["clip_id"])
        self.rows = {r["clip_id"]: r for r in kept}
        self.items: list[str] = [r["clip_id"] for r in kept]
        if not self.items:
            raise RuntimeError(f"{root}: split={split} 没有可用 clip（共 {self.n_total}，drop {self.n_dropped}）")
        if verbose:
            print(f"[GcLatentDataset] {root} split={split}: {self.n_total} 条，drop {self.n_dropped}"
                  f"{'（保留）' if include_dropped else '（剔除）'}，可用 {len(self.items)}"
                  + (f"（limit={limit}）" if limit is not None else "")
                  + f"；res={self.res}，vocab={vocab}，文本表 {len(self.text_table)} 键，dropout text={self.text_dropout} "
                    f"cond={self.cond_dropout} mask={self.mask_p}", flush=True)

    def __len__(self):
        return len(self.items)

    def clip_ids(self) -> list[str]:
        return list(self.items)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def paths(self, cid: str) -> dict[str, str]:
        return {k: os.path.join(self.root, k, cid + ext) for k, ext in (("latents", ".pt"), ("cond", ".npz"), ("meta", ".npz"))}

    # ------------------------------------------------------------------
    def rng(self, idx: int) -> random.Random:
        """(seed, epoch, index) → 本样本的 python RNG（str 种子走 sha512，不受 PYTHONHASHSEED 影响）。"""
        return random.Random(f"gc25d:{self.seed}:{self.epoch}:{int(idx)}")

    def draw(self, rng: random.Random) -> dict:
        """三个 dropout 的抽签（顺序固定：text、cond、mask 各一次 random()，不管概率是否为 0）。"""
        u_text, u_cond, u_mask = rng.random(), rng.random(), rng.random()
        cond = u_cond < self.cond_dropout
        return {"text": u_text < self.text_dropout, "cond": cond, "mask": (not cond) and u_mask < self.mask_p}

    def context_for(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            context = torch.stack([self.text_table[k] for k in prompts], 0)
        except KeyError as e:
            raise KeyError(f"文本表里没有 {e.args[0]!r}") from None
        return context.to(torch.bfloat16), torch.tensor([self.key_to_id[k] for k in prompts], dtype=torch.int64)

    def __getitem__(self, idx: int) -> dict:
        cid = self.items[idx]
        p = self.paths(cid)
        rng = self.rng(idx)
        dr = self.draw(rng)
        d = torch.load(p["latents"], map_location="cpu", weights_only=False)
        latent = d["latent"]
        if tuple(latent.shape) != LATENT_SHAPE:
            raise ValueError(f"{cid}: latent 形状 {tuple(latent.shape)} ≠ {LATENT_SHAPE}")
        prompts = [NULL_PROMPT] * CELLS if dr["text"] else list(d["prompts"])
        if len(prompts) != CELLS:
            raise ValueError(f"{cid}: prompts 有 {len(prompts)} 条，应为 {CELLS}")
        context, context_ids = self.context_for(prompts)

        mask_frac = 0.0
        if dr["cond"]:
            ids = np.full(IDS_SHAPE, UNKNOWN_ID, np.uint8)       # 整段条件 dropout：不用读 npz
        else:
            with np.load(p["cond"], allow_pickle=False) as z:
                ids = z["ids"]
            if ids.dtype != np.uint8 or ids.shape != IDS_SHAPE:
                raise ValueError(f"{cid}: ids 应为 uint8 {IDS_SHAPE}，得到 {ids.dtype} {ids.shape}")
            if dr["mask"]:
                with np.load(p["meta"], allow_pickle=False) as meta:
                    mask_frac = apply_unknown_mask(ids, meta, rng)["frac"]
        if self.vocab == "ids":
            ids = GM.collapse_parts(ids)
        return {
            "input_latents": latent.to(torch.bfloat16).contiguous(),
            "context": context,
            "context_ids": context_ids,
            "sem_ids": torch.from_numpy(np.ascontiguousarray(ids)),
            "clip_id": cid,
            "dropped": {"text": bool(dr["text"]), "cond": bool(dr["cond"]), "mask": float(mask_frac)},
        }


def gc_collate(batch: list[dict]) -> dict:
    """张量按第 0 维堆叠，字符串收成 list，dict（dropped）逐键收成 [b] 张量。batch_size=1 时也带 batch 维。"""
    out = {}
    for k in batch[0]:
        v = batch[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], 0)
        elif isinstance(v, dict):
            out[k] = {kk: torch.tensor([b[k][kk] for b in batch]) for kk in v}
        else:
            out[k] = [b[k] for b in batch]
    return out
