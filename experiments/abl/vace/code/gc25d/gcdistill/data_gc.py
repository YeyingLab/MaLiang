"""gc25d 三阶段蒸馏的数据与条件切片。

分辨率 = geometry（env GC_RES，默认 832x480）：下面的 [.., 30, 52] / [.., 480, 832] 是 832×480 的例子
（1280×704：[.., 44, 80] / [.., 704, 1280]）；数据根 / ckpt 的 "res" 必须等于 GC_RES（check_res）。

stage1 / stage2：与 SFT 同一份 `dataset.GcLatentDataset`（惰性 import —— dataset.py 由 SFT 组件维护），
    返回 {input_latents bf16 [48,21,30,52], context bf16 [21,512,4096], context_ids int64 [21],
          sem_ids uint8 [81,480,832], clip_id}；dropout / UNKNOWN 遮挡在它的 worker 里做。
stage3：`GcRolloutDataset`（本文件）。DMD 的 rollout 只用真实的 cell 0 latent（z0），其余 cell 全是学生自己
    生成的；但每个生成 cell 都要**真实**的逐 cell 条件 —— gc 的语义图和动作文本每个 cell 都在变，不能像 ARoPE
    那样"保持最后一条"（distill_port §6.6：hold = 对着移动的语义图训"站着不动"）。所以只取 has_ext 的 clip：
        sem_ids     = cat(cond ids [81], cond_ext ids [28]) → uint8 [109,480,832]   帧 s … s+108
        context/ids  = 21 条 clip prompt + 7 条 ext prompt   → [28,512,4096] / [28]   cell 0 … 27
    rollout 抽到 N（21..27）后切前 1+N 个 cell / 前 4N+1 帧。
    延长段的来源：v10_real 在帧 s+81 换 look（布景重散），原始 cond_ext 在 cell 21 整体跳变
    ⇒ ext_look_change 的行默认改读 cond_ext_canvas/（ext_canvas.py：地面画布 + 代码画胶囊 + 没见过 = 255，同键）；
    v9 没有 look 切换，照用 cond_ext/。ext_source="raw" 恢复旧行为（A/B 用）。

条件切片（Wan2.2 VAE 4× 时间压缩：cell 0 = 帧 0，cell j = 帧 4j−3 … 4j）
----------------------------------------------------------------------
流式单 cell 与连续窗口直接用 gc_dit.cell_frames / window_frames。这里补上 gc25d_v8 stage3 真正用的
**frankenstein 尾巴**：cat([cell 0, cells a..N])，a = total − K。语义帧 = cat(sem[:, :1], sem[:, 4a−3 : 4N+1])
（T = 1 + 4K = 4(1+K)−3，GcDiT._group 把帧 0 补成 4 份，与"首帧 + 后续整组"的训练片同构），
文本 = cat(ctx[:, :1], ctx[:, a:])。
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

from gcdistill.gc_dit import CELL_FRAMES, cell_frames, window_frames  # noqa: F401  （先 import：它把 code/arope 补进 sys.path）
import geometry as GEO  # noqa: E402  gc25d/geometry.py（分辨率的唯一出处）
import model as GM  # noqa: E402  gc25d/model.py
import text as TXT  # noqa: E402

M_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_ROOT = os.path.join(M_ROOT, "data", "gc25d_v10_fxcond")
TEXT_TABLE = os.path.join(DATA_ROOT, "text_table_gc25d.pt")
REF_TEXT_TABLE = os.path.join(M_ROOT, "ref", "gc25d-live-demo", "weights", "text_table_gc25d.pt")   # 同一张表

CLIP_CELLS = 21                  # 一条 clip 的 latent cell 数
CLIP_FRAMES = 81
EXT_CELLS = 7                    # cond_ext：cells 21..27 = 帧 s+81 … s+108
ROLLOUT_CELLS = CLIP_CELLS + EXT_CELLS          # 28 = 1 + rollout_max(27)
ROLLOUT_FRAMES = CELL_FRAMES * (ROLLOUT_CELLS - 1) + 1   # 109


LAT_HW = (GEO.LAT_H, GEO.LAT_W)                 # 全尺寸 latent 网格（832×480 ⇒ 30×52）


def frames_of(n_cells: int) -> int:
    """n 个 cell（以首帧开头）的语义帧数 4n−3。"""
    return CELL_FRAMES * n_cells - 3


def check_res(*, ckpts=(), root: str | None = None, run_dir: str | None = None) -> None:
    """蒸馏 runner / 评测的分辨率闸（加载 5B 之前调）：每个 (名字, ckpt) 的 metadata res、数据根的 res、已有 run 目录的
    config.json res 都必须 = GC_RES（缺失 ⇒ 832x480；run_dir 没有 config.json ⇒ 不查）。不符 ⇒ SystemExit（消息里点名 GC_RES）。"""
    try:
        for what, p in ckpts:
            if p:
                GEO.check_ckpt_res(p, what)
        if root is not None:
            GEO.check_root_res(root)
        if run_dir is not None:
            GEO.check_run_dir_res(run_dir)
    except RuntimeError as e:
        raise SystemExit(f"ABORT: {e}") from None


# ----------------------------------------------------------------------------
# 文本表
# ----------------------------------------------------------------------------

def load_text_table(path_or_table) -> dict:
    """dict[str → bf16 [512,4096]]。给 dict 就原样返回。"""
    if isinstance(path_or_table, dict):
        return path_or_table
    tab = torch.load(path_or_table, map_location="cpu", weights_only=False)
    assert isinstance(tab, dict) and tab, f"{path_or_table} 不是非空的文本表"
    return tab


def key_to_id(table: dict) -> dict:
    """文本串 → context_id = 排序后的下标（= dataset.GcLatentDataset / infer._KEY_TO_ID）。"""
    return {k: i for i, k in enumerate(sorted(table))}


def cell_context(prompts, table: dict, kid: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 cell prompt → (context bf16 [n,512,4096], context_ids int64 [n])。表里没有的串直接 KeyError（不静默）。"""
    ctx = torch.stack([table[p] for p in prompts], 0).to(torch.bfloat16)
    ids = torch.tensor([kid[p] for p in prompts], dtype=torch.int64)
    return ctx, ids


# ----------------------------------------------------------------------------
# 索引 / 词表
# ----------------------------------------------------------------------------

def read_index(root: str, split: str | None, *, need_ext: bool = False, keep_drop: bool = False) -> list[dict]:
    """index/clips.jsonl → 行列表（按 clip_id 排序，跨 rank 稳定）。默认去掉 drop 行；need_ext ⇒ 只留 has_ext。"""
    path = os.path.join(root, "index", "clips.jsonl")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    out = [r for r in rows
           if (split is None or r.get("split") == split)
           and (keep_drop or not r.get("drop", False))
           and (not need_ext or r.get("has_ext", False))]
    return sorted(out, key=lambda r: r["clip_id"])


def _valid_lut(vocab: str) -> np.ndarray:
    lut = np.zeros(256, dtype=bool)
    lut[sorted(GM.valid_raw_ids(vocab))] = True
    return lut


def to_vocab(ids: np.ndarray, vocab: str) -> np.ndarray:
    """磁盘上恒为 parts 词表的 raw id→ 模型词表；越界值直接报错（模型会把它静默当成 "other"）。"""
    disk = "parts_fx" if vocab == "parts_fx" else "parts"
    if not _valid_lut(disk)[ids].all():
        bad = sorted(set(np.unique(ids).tolist()) - set(GM.valid_raw_ids(disk)))
        raise ValueError(f"语义图里有 {disk} 词表之外的值 {bad[:8]}")
    if vocab in ("parts", "parts_fx"):
        return ids
    if vocab == "ids":
        return GM.collapse_parts(ids)
    raise ValueError(f"vocab 只能是 parts / parts_fx / ids，得到 {vocab!r}")


# ----------------------------------------------------------------------------
# stage1/2：SFT 的数据集（惰性 import）
# ----------------------------------------------------------------------------

def latent_dataset(root: str, split: str, text_table, *, vocab: str, text_dropout: float, cond_dropout: float,
                   mask_p: float, limit: int | None = None, seed: int = 0, verbose: bool = False):
    """dataset.GcLatentDataset（SFT 组件维护）+ 它的 collate。stage1 用 SFT 的 dropout，stage2 全 0。
    它的 dropout 按 (seed, epoch, index) 抽、epoch 由 set_epoch 设 ⇒ DataLoader 不能用 persistent worker
    （fork 出去的副本看不到新 epoch，同 train.py），runner 每个 epoch 开头调 set_epoch（见 loader_kwargs）。"""
    from dataset import GcLatentDataset, gc_collate     # 惰性：dataset.py 与本包并行开发，dry run / 测试不需要它
    ds = GcLatentDataset(root, split, text_table, vocab=vocab, text_dropout=text_dropout,
                         cond_dropout=cond_dropout, mask_p=mask_p, limit=limit, seed=seed, verbose=verbose)
    return ds, gc_collate


def loader_kwargs(ds, num_workers: int) -> dict:
    """DataLoader 的 worker 设置：有 set_epoch 的数据集（GcLatentDataset）不用 persistent worker。"""
    return dict(num_workers=num_workers, pin_memory=True,
                persistent_workers=num_workers > 0 and not hasattr(ds, "set_epoch"))


def gc_collate_basic(samples: list[dict]) -> dict:
    """张量按 dim 0 堆叠，其余（clip_id 之类）收成 list。dataset.gc_collate 不可用时（合成数据 / stage3）用。"""
    out = {}
    for k in samples[0]:
        v = [s[k] for s in samples]
        out[k] = torch.stack(v, 0) if isinstance(v[0], torch.Tensor) else v
    return out


# ----------------------------------------------------------------------------
# stage3：GcRolloutDataset
# ----------------------------------------------------------------------------

class GcRolloutDataset(torch.utils.data.Dataset):
    """stage3 DMD 的样本：真实 z0 + 1+N_max 个 cell 的真实语义图 / 动作文本（只取 has_ext 且未 drop 的 clip）。

    返回 {z0 bf16 [48,1,H/16,W/16], sem_ids uint8 [4·(n_cells−1)+1, H, W], context bf16 [n_cells,512,4096],
          context_ids int64 [n_cells], clip_id}。n_cells 默认 28（rollout_max 27）。dropout 一律 0（DMD 按真实条件走）。"""

    def __init__(self, root: str = DATA_ROOT, split: str = "train", text_table=TEXT_TABLE, *, vocab: str = "parts",
                 n_cells: int = ROLLOUT_CELLS, limit: int | None = None, seed: int = 0, verbose: bool = False,
                 ext_source: str = "auto"):
        assert CLIP_CELLS < n_cells <= ROLLOUT_CELLS or n_cells <= CLIP_CELLS, n_cells
        assert ext_source in ("auto", "raw"), ext_source
        self.res = GEO.check_root_res(root)             # index/prep_args.json 的 res 必须 = GC_RES
        self.root, self.split, self.vocab, self.n_cells = root, split, vocab, int(n_cells)
        self.ext_source = ext_source
        self.text_table = load_text_table(text_table)
        self.kid = key_to_id(self.text_table)
        rows = read_index(root, split, need_ext=self.n_cells > CLIP_CELLS)
        self.n_total = len(rows)
        if limit is not None:
            rows = random.Random(seed).sample(rows, min(int(limit), len(rows))) if limit < len(rows) else rows
            rows = sorted(rows, key=lambda r: r["clip_id"])
        self.rows = rows
        self.canvas_ext = {r["clip_id"] for r in rows if ext_source == "auto" and r.get("ext_look_change")}
        miss = [c for c in self.canvas_ext if not os.path.exists(os.path.join(root, "cond_ext_canvas", f"{c}.npz"))]
        if miss:
            raise RuntimeError(f"{len(miss)} 个 ext_look_change 的 clip 没有 cond_ext_canvas（先跑 ext_canvas.py），例如 {miss[:3]}")
        if not rows:
            raise RuntimeError(f"{root} split={split} 没有可用的 clip（stage3 要 has_ext 且未 drop）")
        if verbose:
            print(f"[data] GcRolloutDataset {root} split={split}: {len(rows)}/{self.n_total} clip，"
                  f"{self.n_cells} cell / {frames_of(self.n_cells)} 帧，vocab={vocab}，"
                  f"延长段 canvas {len(self.canvas_ext)} / raw {len(rows) - len(self.canvas_ext)}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def clip_ids(self) -> list[str]:
        return [r["clip_id"] for r in self.rows]

    def load(self, clip_id: str) -> dict:
        lat = torch.load(os.path.join(self.root, "latents", f"{clip_id}.pt"), map_location="cpu", weights_only=False)
        x = lat["latent"]
        assert tuple(x.shape) == (48, CLIP_CELLS) + LAT_HW, (clip_id, tuple(x.shape), GEO.RES_TAG)
        prompts = list(lat["prompts"])
        assert len(prompts) == CLIP_CELLS, (clip_id, len(prompts))
        ids = np.load(os.path.join(self.root, "cond", f"{clip_id}.npz"))["ids"]
        assert ids.shape[0] == CLIP_FRAMES and ids.dtype == np.uint8, (clip_id, ids.shape, ids.dtype)
        n_ext = self.n_cells - CLIP_CELLS
        if n_ext > 0:
            sub = "cond_ext_canvas" if clip_id in self.canvas_ext else "cond_ext"
            z = np.load(os.path.join(self.root, sub, f"{clip_id}.npz"))
            ext = z["ids"]
            assert ext.shape[0] >= CELL_FRAMES * n_ext and ext.dtype == np.uint8, (clip_id, ext.shape)
            if "prompts" in z.files:
                p_ext = [str(p) for p in z["prompts"]]
            else:
                p_ext = [TXT.block_to_prompt(*map(int, b)) for b in z["blocks"]]
            ids = np.concatenate([ids, ext[:CELL_FRAMES * n_ext]], 0)
            prompts = prompts + p_ext[:n_ext]
        else:
            ids = ids[:frames_of(self.n_cells)]
            prompts = prompts[:self.n_cells]
        assert ids.shape[0] == frames_of(self.n_cells) and len(prompts) == self.n_cells, (ids.shape, len(prompts))
        ctx, cid = cell_context(prompts, self.text_table, self.kid)
        return {"z0": x[:, :1].to(torch.bfloat16).contiguous(),
                "sem_ids": torch.from_numpy(np.ascontiguousarray(to_vocab(ids, self.vocab))),
                "context": ctx, "context_ids": cid, "clip_id": clip_id, "prompts": prompts}

    def __getitem__(self, i: int) -> dict:
        s = self.load(self.rows[i]["clip_id"])
        s.pop("prompts")
        return s


def rollout_collate(samples: list[dict]) -> dict:
    return gc_collate_basic(samples)


def assert_rollout_aligned_gc(sem_ext: torch.Tensor, ctx_ext: torch.Tensor, ids_ext: torch.Tensor, N: int) -> None:
    """rollout 前的对齐闸（data.assert_rollout_aligned 的 gc 版：offset 换成语义帧）：1+N 个 cell、4N+1 帧。"""
    b, want = ctx_ext.shape[0], N + 1
    if tuple(ctx_ext.shape[:2]) != (b, want):
        raise AssertionError(f"context 形状 {tuple(ctx_ext.shape)} 前两维 != {(b, want)}")
    if tuple(ids_ext.shape) != (b, want):
        raise AssertionError(f"context_ids 形状 {tuple(ids_ext.shape)} != {(b, want)}")
    if sem_ext.dim() != 4 or tuple(sem_ext.shape[:2]) != (b, frames_of(want)):
        raise AssertionError(f"sem_ids 形状 {tuple(sem_ext.shape)} 前两维 != {(b, frames_of(want))}")
    if sem_ext.dtype != torch.uint8:
        raise AssertionError(f"sem_ids 必须是 uint8，得到 {sem_ext.dtype}")


def slice_rollout(batch: dict, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GcRolloutDataset 的 batch → 前 1+N 个 cell 的 (sem_ext [b,4N+1,H,W], ctx_ext [b,1+N,L,C], ids_ext [b,1+N])。"""
    have = batch["context"].shape[1]
    if N + 1 > have:
        raise ValueError(f"rollout 要 {N + 1} 个 cell 的条件，样本只有 {have} 个（数据集 n_cells / --rollout_max 对不上）")
    sem = batch["sem_ids"][:, :frames_of(N + 1)]
    ctx, ids = batch["context"][:, :N + 1], batch["context_ids"][:, :N + 1]
    assert_rollout_aligned_gc(sem, ctx, ids, N)
    return sem, ctx, ids


# ----------------------------------------------------------------------------
# 打分尾巴的切片
# ----------------------------------------------------------------------------

def frankenstein_frames(sem_ids: torch.Tensor, a: int, total: int) -> torch.Tensor:
    """cat([cell 0, cells a..total−1]) 的语义帧：cat(sem[:, :1], sem[:, 4a−3 : 4(total−1)+1])，T = 4(total−a)+1。
    a=1 ⇒ 整片 sem[:, :4(total−1)+1]。"""
    assert 1 <= a < total, (a, total)
    lo, hi = CELL_FRAMES * a - 3, CELL_FRAMES * (total - 1) + 1
    assert hi <= sem_ids.shape[1], f"尾巴要到帧 {hi - 1}，sem_ids 只有 {sem_ids.shape[1]} 帧"
    out = torch.cat([sem_ids[:, :1], sem_ids[:, lo:hi]], dim=1)
    assert out.shape[1] == frames_of(1 + total - a), (tuple(out.shape), a, total)
    return out


def frankenstein_cells(x: torch.Tensor, a: int, dim: int = 1) -> torch.Tensor:
    """逐 cell 张量（文本 [b,F,…] / id [b,F] 用 dim=1，latent [b,C,F,…] 用 dim=2）的 cat([cell 0, cells a..])。"""
    return torch.cat([x.narrow(dim, 0, 1), x.narrow(dim, a, x.shape[dim] - a)], dim=dim)


def tail_slices(total: int, Keff: int | None, frankenstein: bool) -> tuple[int, list[int]]:
    """打分尾巴的 (a, tail_to_frame)。三种形态：
      整片（Keff 为 None 或 ≥ N）：a=0，[0..total)；
      frankenstein：a = total−Keff，[0] + [a..total)；
      连续：a = total−1−Keff，[a..total)（锚点 = 生成的 cell a）。"""
    N = total - 1
    if Keff is None or Keff >= N:
        return 0, list(range(total))
    if frankenstein:
        a = total - Keff
        return a, [0] + list(range(a, total))
    a = total - 1 - Keff
    return a, list(range(a, total))


# ----------------------------------------------------------------------------
# 合成数据（dry run / 测试 / 真数据到位之前的真权重冒烟）
# ----------------------------------------------------------------------------

def syn_sem_ids(T: int, H: int, W: int, vocab: str, g: torch.Generator, block: int = 16) -> torch.Tensor:
    """块状随机语义图（每 block×block 一个合法 raw id），uint8 [T,H,W]；帧间缓慢变化（每帧重抽 1/8 的块）。"""
    vals = torch.tensor(sorted(GM.valid_raw_ids(vocab)), dtype=torch.uint8)
    hb, wb = max(1, H // block), max(1, W // block)
    cur = vals[torch.randint(0, len(vals), (hb, wb), generator=g)]
    out = []
    for _ in range(T):
        flip = torch.rand(hb, wb, generator=g) < 0.125
        cur = torch.where(flip, vals[torch.randint(0, len(vals), (hb, wb), generator=g)], cur)
        out.append(cur)
    ids = torch.stack(out, 0).repeat_interleave(H // hb, 1).repeat_interleave(W // wb, 2)
    return ids[:, :H, :W].contiguous()


class SynGcDataset(torch.utils.data.Dataset):
    """形状与 GcLatentDataset 一致的合成样本。tiny（dry run：cells/lat 缩小，文本随机 L=8）或全尺寸
    （真权重冒烟：cells=21, lat = LAT_HW（GC_RES 的 latent 网格，832×480 ⇒ 30×52），文本取自真文本表）。"""

    def __init__(self, n: int = 64, cells: int = 5, lat_hw=(8, 10), *, vocab: str = "parts", text_table=None,
                 txt_len: int = 8, seed: int = 0):
        self.n, self.cells, self.lat_hw, self.vocab, self.seed, self.txt_len = n, cells, lat_hw, vocab, seed, txt_len
        self.table = load_text_table(text_table) if text_table is not None else None
        self.kid = key_to_id(self.table) if self.table is not None else None

    def __len__(self):
        return self.n

    def _text(self, g, n):
        if self.table is None:
            return (torch.randn(n, self.txt_len, 4096, generator=g).to(torch.bfloat16),
                    torch.zeros(n, dtype=torch.int64))
        keys = [k for k in TXT.all_prompts() if k in self.table] or sorted(self.table)
        prompts = [keys[int(torch.randint(0, len(keys), (1,), generator=g))] for _ in range(n - 1)]
        return cell_context([prompts[0]] + prompts, self.table, self.kid)     # cell 0 = cell 1

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        h, w = self.lat_hw
        ctx, cid = self._text(g, self.cells)
        return {"input_latents": torch.randn(48, self.cells, h, w, generator=g).to(torch.bfloat16),
                "context": ctx, "context_ids": cid,
                "sem_ids": syn_sem_ids(frames_of(self.cells), 16 * h, 16 * w, self.vocab, g),
                "clip_id": f"syn_{i:04d}"}


class SynGcRolloutDataset(SynGcDataset):
    """GcRolloutDataset 的合成版：z0 + n_cells 个 cell 的条件。"""

    def __getitem__(self, i: int) -> dict:
        s = super().__getitem__(i)
        return {"z0": s["input_latents"][:, :1].contiguous(), "sem_ids": s["sem_ids"], "context": s["context"],
                "context_ids": s["context_ids"], "clip_id": s["clip_id"]}


def wait_ready(root: str) -> bool:
    """prep 写完会落 <root>/READY。"""
    return os.path.isfile(os.path.join(root, "READY"))


# ----------------------------------------------------------------------------
# dry run / 测试的 tiny GcDiT（与上游 stage 的 TINY 同几何：dim 128 / 4 头 / 2 层，latent 8×10 ⇒ 语义图 128×160）
# ----------------------------------------------------------------------------

TINY = dict(dim=128, in_dim=48, ffn_dim=256, out_dim=48, text_dim=4096, freq_dim=256, eps=1e-6,
            patch_size=(1, 2, 2), num_heads=4, num_layers=2, has_image_input=False)
TINY_SHAPE = (48, 5, 8, 10)          # [C, cells, lat_h, lat_w]


def tiny_gcdit(causal: bool, seed: int = 0, *, vocab: str = "parts", dtype=torch.bfloat16, device="cpu",
               randomize_cond: bool = True, tiny: dict | None = None, **kw):
    """固定种子 ⇒ 同 seed 两次调用逐位相同（P1：teacher / student / ema 同初值）。
    randomize_cond：gc_sem_cond.proj_out 是零初始化（= 条件没接上），随机化之后语义图才真的参与前向。"""
    from diffsynth.models.wan_video_dit import WanModel
    from gcdistill.gc_dit import GcDiT
    torch.manual_seed(seed)
    dit = WanModel(**(tiny or TINY))
    GM.install_sem_cond(dit, vocab)
    if randomize_cond:
        g = torch.Generator().manual_seed(seed + 77)
        enc = getattr(dit, GM.SEM_COND_NAME)
        with torch.no_grad():
            enc.proj_out.weight.copy_(torch.randn(enc.proj_out.weight.shape, generator=g) * 0.02)
            enc.proj_out.bias.copy_(torch.randn(enc.proj_out.bias.shape, generator=g) * 0.02)
    dit = dit.to(device=device, dtype=dtype)
    dit.freqs = tuple(t.to(device) for t in dit.freqs)
    return GcDiT(dit=dit, causal=causal, **kw)
