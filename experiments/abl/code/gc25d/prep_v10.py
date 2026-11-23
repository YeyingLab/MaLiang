#!/usr/bin/env python3
"""gc25d v10 数据预处理：recordings tar → data/gc25d_v10/。

    python prep_v10.py --src $M/data/recordings/data --out $M/data/gc25d_v10 --gpus 0,1,...,7
    GC_RES=1280x704 python prep_v10.py --out $M/data/gc25d_v10_704 --gpus 0,1,...,7     # 原生分辨率（不裁不缩）

分辨率 = geometry（env GC_RES，默认 832x480）：RGB = resample_rgb(帧)，ids = resample_ids(ids.png)，label_parts 在同一网格上做；
1280×704 时三者都是原始分辨率（恒等）。"res" 写进 index/prep_args.json 与每一行 index；--out 里已有别的分辨率的产物 ⇒ 拒绝。

进程布局（data_v10.md §8.3）
--------------------------
* 主进程：列 run / clip → 去掉已完成的（resume）→ 按 (subset, scene, run, start) 排序后切成 G 段连续块（tar 局部性）
  → 每张卡 spawn 一个 rank 进程 → 等全部结束 → merge 出 index/clips.jsonl、index/runs.json。
* rank 进程：torch DataLoader（num_workers = --workers_per_gpu，batch_size=None）的 worker 做 CPU 活，主线程只跑 VAE。
  - worker（OMP_NUM_THREADS=1、cv2.setNumThreads(1)、每 run 缓存一个 RunTars）：读 81（+28 扩展）帧的 json / jpg / ids.png
    → resample_rgb / resample_ids → capsules.label_parts（parts 词表）→ 动作 / prompt → meta → 原子写 cond/ cond_ext/ meta/，
    把 RGB uint8 [81,H,W,3]（共享内存）交给主线程。
  - 主线程：Wan2.2 VAE（fp32 权重、PyTorch 默认 TF32、non-tiled）编码 → bf16 → 原子写 latents/<clip>.pt
    → 往 index/rows_rank<r>.jsonl 追加一行（行在 latent 之后写 ⇒ 有行 = 该 clip 全部输出都在）。
* resume：有行且输出文件都在的 clip 直接跳过；输出都在但没行（行文件丢了）⇒ worker 只读 json / meta 补行，不解码。
* 中断：kill 主进程（SIGTERM / Ctrl-C）⇒ 主进程收掉全部 rank；rank / worker 设了 PR_SET_PDEATHSIG，主进程被 SIGKILL 也跟着退出，
  不会留下孤儿进程和下一次 resume 抢同一个输出目录。

时间：clip = clips_81.json 的 [start, start+81)；blocks20 = actions_block[s//4 : s//4+20]；
rollout 扩展 = 帧 s+81…s+108（cell 21..27）+ blocks s//4+20 … s//4+26，仅当帧 s…s+108 的逐帧 json 的 episode 与
camera_variant 全相同（且帧 / 块都没越过 run 末尾）时有效。
注意：v10_real 每个 clip 边界（帧 s+81）换 look（biome + 光照），语义图里植被 / 岩石在 cell 21 整体重排（静态类直方图 L1
跳变 ≈ 30× 正常帧间）；has_ext 按 设计 不看 look，这类 clip 在行里标 ext_look_change=True，由下游决定是否过滤。
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

# worker 必须单线程（data_v10.md §5(e)：不设时 64 个 numpy 进程把 load 打到 2300）；要在 import numpy 之前设。
# 本脚本里没有吃 CPU 线程的大算子（VAE 在 GPU），所以整个进程树一律 1。
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import collections
import ctypes
import functools
import glob
import io
import json
import shutil
import signal
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

import capsules as CAP
import geometry as GEO
import model as GM
import text as TX

cv2.setNumThreads(1)

M_ROOT = MALIANG_ROOT
SRC_DEFAULT = os.path.join(M_ROOT, "data", "recordings", "data")
VAE_PATH = os.path.join(M_ROOT, "models", "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
TEXT_TABLE_SRC = os.path.join(M_ROOT, "ref", "gc25d-live-demo", "weights", "text_table_gc25d.pt")
BURIED_LIST = os.path.join(M_ROOT, "docs", "understand", "data_v10_tools", "v10_real_player_buried_clips.txt")
SUBSETS = ("v10_real", "v9_stylized")

F_CLIP, F_EXT = 81, 28                    # 81 帧 = 21 cell；扩展 28 帧 = cell 21..27
F_SPAN = F_CLIP + F_EXT                   # 109：扩展有效性要看帧 s … s+108
N_CELLS, N_BLOCKS, N_EXT_CELLS = TX.CELLS, TX.BLOCKS_PER_CLIP, F_EXT // 4
LAT_SHAPE = (48, 21, GEO.LAT_H, GEO.LAT_W)        # 832×480 ⇒ (48, 21, 30, 52)
DST_H, DST_W = GEO.DST_H, GEO.DST_W

# run 级 hold-out，10 个 val run（500 clip），其余全 train
VAL_RUNS = frozenset({
    "R10_s1512", "R10_s3512", "R10_s5512", "R10_s7511", "R10_s9511",                         # v10_real 01 03 05 07 09
    "P9_golden_s2909", "P9_autumn_s4908", "P9C_golden_s6946", "P9C_default_s8946", "P9C_snow_s10945",  # v9 02 04 06 08 10
})
SPLIT_RULE = ("run-level hold-out, 10 val runs = v10_real R10_s1512 R10_s3512 R10_s5512 R10_s7511 R10_s9511 + "
              "v9_stylized P9_golden_s2909 P9_autumn_s4908 P9C_golden_s6946 P9C_default_s8946 P9C_snow_s10945; everything else train")
DROP_REASON = "buried_player"

# parts 词表里允许出现的值（数据里不该有 UNKNOWN=255）
_VALID_PARTS = np.zeros(256, bool)
_VALID_PARTS[sorted(GM.valid_raw_ids("parts") - {GM.UNKNOWN_ID})] = True
# 源 ids.png 允许的值 {0..6,10,11}（data_v10.md §8.2 第 3 步）；label_parts 只改写 10/11，源里混进 12..21 会被当成部位 id 放过去
_VALID_RAW = np.zeros(256, bool)
_VALID_RAW[list(GEO.RAW_IDS)] = True
_PLAYER_IDS = list(GM.PLAYER_PART_IDS)
_BOSS_IDS = list(GM.BOSS_PART_IDS)


# ----------------------------------------------------------------------------
# tar 随机访问
# ----------------------------------------------------------------------------
class RunTars:
    """random access to frames of a run: member offsets indexed once per shard（首次用到该分片时索引，≈0.2 s/分片）"""

    def __init__(self, run):
        self.run = run
        self.shards = sorted((int(os.path.basename(p)[7:13]), p) for p in glob.glob(os.path.join(run, "frames-*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"{run}：没有 frames-*.tar")
        self.idx, self.done, self.fh = {}, set(), {}

    def _index(self, p):
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                k, ext = m.name.split(".", 1)
                self.idx[(int(k), ext)] = (p, m.offset_data, m.size)
        self.done.add(p)

    def get(self, f, ext):
        if (f, ext) not in self.idx:
            p = [p for s, p in self.shards if s <= f][-1]
            if p not in self.done:
                self._index(p)
        p, off, sz = self.idx[(f, ext)]
        fh = self.fh.get(p) or open(p, "rb")
        self.fh[p] = fh
        fh.seek(off)
        return fh.read(sz)

    def json(self, f):
        return json.loads(self.get(f, "json"))

    def close(self):
        for fh in self.fh.values():
            fh.close()
        self.fh = {}


_TARS: "collections.OrderedDict[str, RunTars]" = collections.OrderedDict()
_BLOCKS: dict = {}


def _tars(run_dir: str, keep: int = 2) -> RunTars:
    """每个 worker 每 run 一个 RunTars（LRU，最多 keep 个，控制打开的文件数）。"""
    if run_dir in _TARS:
        _TARS.move_to_end(run_dir)
        return _TARS[run_dir]
    while len(_TARS) >= keep:
        _TARS.popitem(last=False)[1].close()
    _TARS[run_dir] = RunTars(run_dir)
    return _TARS[run_dir]


def _blocks(run_dir: str) -> np.ndarray:
    if run_dir not in _BLOCKS:
        b = np.load(os.path.join(run_dir, "index", "actions_block_0.25s.npy"))
        assert b.ndim == 2 and b.shape[1] == 5 and b.dtype == np.uint8, (run_dir, b.shape, b.dtype)
        _BLOCKS[run_dir] = b
    return _BLOCKS[run_dir]


# ----------------------------------------------------------------------------
# 纯函数（tests/test_prep.py 覆盖）
# ----------------------------------------------------------------------------
def clip_id_of(scene: str, run: str, start: int) -> str:
    return f"{scene}__{run}__{start:06d}"


def split_of(run: str) -> str:
    return "val" if run in VAL_RUNS else "train"


def clip_blocks(AB: np.ndarray, s: int) -> np.ndarray:
    """blocks20 = actions_block[s//4 : s//4+20]（cell j ← block s//4+j−1）。"""
    assert s % 4 == 0, s
    b = AB[s // 4: s // 4 + N_BLOCKS]
    assert b.shape == (N_BLOCKS, 5), (s, b.shape)
    return b


def ext_blocks(AB: np.ndarray, s: int) -> np.ndarray:
    """扩展 cell 21..27 的块 = actions_block[s//4+20 : s//4+27]（[7,5]）。"""
    b = AB[s // 4 + N_BLOCKS: s // 4 + N_BLOCKS + N_EXT_CELLS]
    assert b.shape == (N_EXT_CELLS, 5), (s, b.shape)
    return b


def ext_valid(episodes, cameras, s: int, n_blocks: int) -> bool:
    """rollout 扩展是否有效。episodes / cameras = 帧 s, s+1, … 的逐帧值（能取到多少给多少，最多 109 个）。

    有效 ⇔ 帧 s … s+108 都存在、episode 与 camera_variant 全相同，且块 s//4+26 存在。"""
    ep = np.asarray(episodes)[:F_SPAN]
    cv = np.asarray(cameras)[:F_SPAN]
    if len(ep) < F_SPAN or len(cv) < F_SPAN or s // 4 + N_BLOCKS + N_EXT_CELLS > n_blocks:
        return False
    return bool((ep == ep[0]).all() and (cv == cv[0]).all())


def clip_paths(out: str, clip_id: str) -> dict:
    return {"latent": os.path.join(out, "latents", clip_id + ".pt"),
            "cond": os.path.join(out, "cond", clip_id + ".npz"),
            "cond_ext": os.path.join(out, "cond_ext", clip_id + ".npz"),
            "meta": os.path.join(out, "meta", clip_id + ".npz")}


def outputs_exist(paths: dict, has_ext: bool, vae: bool) -> bool:
    need = ["cond", "meta"] + (["cond_ext"] if has_ext else []) + (["latent"] if vae else [])
    return all(os.path.isfile(paths[k]) for k in need)


def _tmp(path: str) -> str:
    return f"{path}.tmp{os.getpid()}"


def atomic_savez(path: str, **arrays) -> None:
    """np.savez_compressed 到 <path>.tmp<pid> 再 os.replace（给文件对象，免得 numpy 自己补 .npz 后缀）。"""
    tmp = _tmp(path)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def atomic_torch_save(path: str, obj) -> None:
    tmp = _tmp(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path: str, s: str) -> None:
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(s)
    os.replace(tmp, path)


def check_raw_ids(raw: np.ndarray, where: str) -> None:
    """源 ids.png（uint8 [704,1280]）的值必须 ⊆ {0..6,10,11}，否则 AssertionError。"""
    cnt = np.bincount(raw.ravel(), minlength=256)
    if cnt[~_VALID_RAW].any():
        raise AssertionError(f"{where}：ids.png 有 {{0..6,10,11}} 以外的值 {np.nonzero(cnt * ~_VALID_RAW)[0].tolist()}")


def open_rows(path: str):
    """追加打开 rows_rank<r>.jsonl。上次被杀留下的半行（文件不以 '\\n' 结尾）先补 '\\n' 封口，
    否则这次的第一行会粘在半行后面、被 read_rows 一起丢掉（merge 时报缺行）。"""
    torn = False
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            torn = fh.read(1) != b"\n"
    fh = open(path, "a", encoding="utf-8")
    if torn:
        fh.write("\n")
        fh.flush()
    return fh


def die_with_parent(parent_pid: int, sig: int = signal.SIGTERM) -> None:
    """父进程退出（包括被 SIGKILL）时本进程收到 sig（Linux prctl PR_SET_PDEATHSIG）。
    fork 之后、prctl 之前父进程就已经没了 ⇒ 直接退出。非 Linux / 没有 libc ⇒ 不做。"""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return
    if libc.prctl(1, int(sig), 0, 0, 0) != 0:                  # 1 = PR_SET_PDEATHSIG
        return
    if os.getppid() != parent_pid:
        os._exit(1)


def ground_z(caps: np.ndarray) -> np.ndarray:
    """capsules f32 [F,2,12,7] → 每帧玩家胶囊端点 z − r 的最小值 f32 [F]（玩家没胶囊 ⇒ NaN）。"""
    p = caps[:, 0]
    z = np.minimum(p[..., 2], p[..., 5]) - p[..., 6]                  # [F,12]，缺槽 = NaN
    z = np.where(np.isnan(z), np.inf, z).min(1)
    return np.where(np.isfinite(z), z, np.nan).astype(np.float32)


# ----------------------------------------------------------------------------
# 规划
# ----------------------------------------------------------------------------
def list_runs(src: str, runs=None, max_runs=None) -> list[tuple[str, str, str, str]]:
    """→ [(subset, scene, run, run_dir)]，按 (subset 顺序 v10_real, v9_stylized; scene; run) 排序。"""
    out = []
    for si, sub in enumerate(SUBSETS):
        for d in sorted(glob.glob(os.path.join(src, sub, "*", "*"))):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "index", "clips_81.json")):
                out.append((sub, os.path.basename(os.path.dirname(d)), os.path.basename(d), d))
    if runs:
        want = list(dict.fromkeys(runs))
        have = {r[2]: r for r in out}
        miss = [r for r in want if r not in have]
        if miss:
            raise SystemExit(f"--runs 里这些 run 在 {src} 下找不到：{miss}")
        out = [r for r in out if r[2] in set(want)]
    if max_runs:
        out = out[:max_runs]
    return out


def plan_tasks(runs, clips_per_run=None, buried=frozenset()) -> list[dict]:
    tasks = []
    for sub, scene, run, d in runs:
        clips = json.load(open(os.path.join(d, "index", "clips_81.json")))
        n_frames = int(np.load(os.path.join(d, "index", "episodes.npy"), mmap_mode="r").shape[0])
        if clips_per_run and clips_per_run < len(clips):       # 冒烟：在 run 里均匀取
            pick = sorted(set(np.linspace(0, len(clips) - 1, clips_per_run).round().astype(int).tolist()))
            clips = [clips[i] for i in pick]
        for c in clips:
            s, e = int(c["start"]), int(c["end"])
            assert e - s == F_CLIP and s % 96 == 0, (d, c)
            cid = clip_id_of(scene, run, s)
            tasks.append({
                "clip_id": cid, "subset": sub, "scene": scene, "run": run, "run_dir": d, "start": s, "end": e,
                "n_frames": n_frames, "split": split_of(run), "drop": cid in buried,
                "drop_reason": DROP_REASON if cid in buried else None,
                "episode": int(c["episode"]), "camera_variant": int(c["camera_variant"]), "look": int(c["look"]),
                "look_name": c["look_name"], "centered": bool(c["centered"]),
                "max_frame_offset_cm": float(c["max_frame_offset_cm"]), "n_events": len(c["events"]),
            })
    return tasks


def read_rows(out: str) -> dict:
    """index/rows_rank*.jsonl → {clip_id: row}（重复的 clip 后读到的覆盖先读到的；内容与 rank 无关）。"""
    rows = {}
    for p in sorted(glob.glob(os.path.join(out, "index", "rows_rank*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:            # 被杀时写了半行
                        continue
                    rows[r["clip_id"]] = r
    return rows


def done_clips(out: str, tasks, vae: bool) -> set:
    rows = read_rows(out)
    return {t["clip_id"] for t in tasks
            if t["clip_id"] in rows and outputs_exist(clip_paths(out, t["clip_id"]), rows[t["clip_id"]]["has_ext"], vae)}


# ----------------------------------------------------------------------------
# worker：一个 clip 的 CPU 部分
# ----------------------------------------------------------------------------
def _row(t: dict, *, has_ext, ext_look_change, ortho, mirrored, dead, player_px, boss_px) -> dict:
    keys = ("clip_id", "subset", "scene", "run", "start", "end", "split", "drop", "drop_reason", "episode",
            "camera_variant", "look", "look_name", "centered", "max_frame_offset_cm", "n_events")
    row = {k: t[k] for k in keys}
    row["res"] = GEO.RES_TAG                                       # 本行产物的分辨率（GC_RES）
    dead = np.asarray(dead, bool)
    pp, bp = np.asarray(player_px[:F_CLIP]), np.asarray(boss_px[:F_CLIP])
    row.update({
        "dead_frac": round(float(dead.mean()), 4), "has_ext": bool(has_ext), "ext_look_change": bool(ext_look_change),
        "ortho": bool(ortho), "mirrored_world": bool(mirrored),
        "player_px_mean": round(float(pp.mean()), 1), "boss_visible_frac": round(float((bp > 0).mean()), 4),
        "player_hidden_frac": round(float((pp[~dead] == 0).mean()), 4) if (~dead).any() else None,
    })
    return row


def prep_clip(t: dict, out: str, vae: bool = True) -> dict:
    """一个 clip：json → 解码 / 重采样 → label_parts → 写 cond / cond_ext / meta；返回 RGB（给 VAE）+ index 行。"""
    T0 = time.time()
    cid, s = t["clip_id"], t["start"]
    paths = clip_paths(out, cid)
    R, AB = _tars(t["run_dir"]), _blocks(t["run_dir"])

    # 逐帧 json（index/frames.jsonl 没有 world_to_view / projection / v9 胶囊，必须读 tar 里的）
    n_win = min(F_SPAN, t["n_frames"] - s)
    Js = [R.json(s + i) for i in range(n_win)]
    eps = [J["episode"] for J in Js]
    cams = [J["camera_variant"] for J in Js]
    if set(eps[:F_CLIP]) != {t["episode"]} or set(cams[:F_CLIP]) != {t["camera_variant"]}:
        raise AssertionError(f"{cid}：clip 内 episode / camera_variant 与 clips_81.json 不一致 {set(eps[:F_CLIP])} {set(cams[:F_CLIP])}")
    has_ext = ext_valid(eps, cams, s, len(AB))
    F = F_SPAN if has_ext else F_CLIP
    Js = Js[:F]
    ortho = {bool(J["orthographic"]) for J in Js}
    assert len(ortho) == 1, f"{cid}：orthographic 在 clip 内变化"
    ortho = ortho.pop()
    looks = {J["v9"]["look"] for J in Js}
    rowkw = dict(has_ext=has_ext, ext_look_change=has_ext and len(looks) > 1, ortho=ortho,
                 mirrored=bool(Js[0]["mirrored_world"]), dead=[J["action"]["bot_mode"] == "dead" for J in Js[:F_CLIP]])
    blocks20 = clip_blocks(AB, s)
    T1 = time.time()

    if outputs_exist(paths, has_ext, vae):            # 输出都在、只缺 index 行：补行，不解码
        with np.load(paths["meta"], allow_pickle=False) as z:
            pp, bp = z["player_px"], z["boss_px"]
        return {"skip": True, "clip_id": cid, "row": _row(t, player_px=pp, boss_px=bp, **rowkw)}

    # 解码 + 重采样 + 部位标注
    rgb_t = torch.empty((F_CLIP, DST_H, DST_W, 3), dtype=torch.uint8).share_memory_() if vae else None
    rgb = rgb_t.numpy() if vae else None
    parts = np.empty((F, DST_H, DST_W), np.uint8)
    caps = np.empty((F, 2, len(CAP.CAPSULE_SLOTS), 7), np.float32)
    player_px = np.empty(F, np.int32)
    boss_px = np.empty(F, np.int32)
    t_dec = t_lab = 0.0
    for i in range(F):
        a = time.time()
        if vae and i < F_CLIP:
            bgr = cv2.imdecode(np.frombuffer(R.get(s + i, "jpg"), np.uint8), cv2.IMREAD_COLOR)
            rgb[i] = GEO.resample_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        raw = cv2.imdecode(np.frombuffer(R.get(s + i, "ids.png"), np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint8 or raw.shape != (GEO.SRC_H, GEO.SRC_W):
            raise AssertionError(f"{cid} 帧 {s + i}：ids.png 应为 uint8 [704,1280]，得到 {None if raw is None else (raw.dtype, raw.shape)}")
        check_raw_ids(raw, f"{cid} 帧 {s + i}")
        ids = GEO.resample_ids(raw)
        b = time.time()
        J = Js[i]
        cl = CAP.capsules_from_json(J)
        with np.errstate(invalid="ignore"):            # cap_hit 里没命中射线的 inf·0，只影响告警不影响结果
            parts[i] = CAP.label_parts(ids, J["world_to_clip"], J["world_to_view"], cl)
        caps[i] = CAP.capsules_to_array(cl)
        cnt = np.bincount(parts[i].ravel(), minlength=256)
        if cnt[~_VALID_PARTS].any():
            raise AssertionError(f"{cid} 帧 {s + i}：parts 里有词表外的值 {np.nonzero(cnt * ~_VALID_PARTS)[0].tolist()}")
        player_px[i], boss_px[i] = cnt[_PLAYER_IDS].sum(), cnt[_BOSS_IDS].sum()
        t_dec += b - a
        t_lab += time.time() - b
    T2 = time.time()

    # meta
    def actor(J, role):
        a = [x for x in J["actors"] if x["role"] == role]
        return a[0] if a else None

    def vec(J, key):
        return np.asarray(J[key], np.float64).reshape(16)

    pos = np.full((F, 2, 3), np.nan, np.float32)
    state = {"player": [], "boss": []}
    for i, J in enumerate(Js):
        for k, role in enumerate(("player", "boss")):
            a = actor(J, role)
            if a is not None:
                pos[i, k] = a["position_cm"]
            state[role].append(a["action"] if a is not None else "")
    meta = {
        "world_to_clip": np.stack([vec(J, "world_to_clip") for J in Js]),
        "world_to_view": np.stack([vec(J, "world_to_view") for J in Js]),
        "projection": np.stack([vec(J, "projection") for J in Js]),
        "orthographic": np.array(ortho),
        "fov_x_deg": np.array([J["fov_x_deg"] for J in Js], np.float32),
        "ortho_width_cm": np.array([J["ortho_width_cm"] for J in Js], np.float32),
        "frame_offset_cm": np.array([J["v9"]["frame_offset_cm"] for J in Js], np.float32),
        "capsules": caps,
        "actor_pos": pos,
        "ground_z": ground_z(caps),
        "player_state": np.array(state["player"], dtype=str),
        "boss_state": np.array(state["boss"], dtype=str),
        "bot_mode": np.array([J["action"]["bot_mode"] for J in Js], dtype=str),
        "frames": (s + np.arange(F)).astype(np.int32),
        "player_px": player_px,
        "boss_px": boss_px,
    }
    if has_ext:
        eb = ext_blocks(AB, s)
        atomic_savez(paths["cond_ext"], ids=parts[F_CLIP:], blocks=eb.astype(np.uint8),
                     prompts=np.array([TX.block_to_prompt(*r.tolist()) for r in eb], dtype=str))
    atomic_savez(paths["meta"], **meta)
    atomic_savez(paths["cond"], ids=parts[:F_CLIP])
    T3 = time.time()

    return {
        "skip": False, "clip_id": cid, "rgb": rgb_t,
        "row": _row(t, player_px=player_px, boss_px=boss_px, **rowkw),
        "pt": {"prompts": TX.cell_prompts(blocks20), "actions_block": TX.cell_blocks(blocks20),
               "clip_id": cid, "subset": t["subset"], "scene": t["scene"], "run": t["run"], "start": s},
        "tm": {"json": T1 - T0, "decode": t_dec, "label": t_lab, "write": T3 - T2, "total": T3 - T0, "F": F},
    }


class ClipSet(torch.utils.data.Dataset):
    def __init__(self, tasks, out, vae):
        self.tasks, self.out, self.vae = tasks, out, vae

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        t = self.tasks[i]
        try:
            return prep_clip(t, self.out, self.vae)
        except Exception as e:                          # 带上 clip_id 再抛（DataLoader 会在主线程重抛）
            raise RuntimeError(f"prep 失败 {t['clip_id']}: {type(e).__name__}: {e}") from e


def _worker_init(_, parent_pid=None):
    if parent_pid is not None:
        die_with_parent(parent_pid)                    # rank 被杀 ⇒ worker 立刻退出（不等 DataLoader 的 5 s 看门狗）
    os.environ["OMP_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------
# rank：DataLoader + VAE
# ----------------------------------------------------------------------------
def load_vae(dev: str):
    """WanVideoVAE38（infer.load_vae 同一路径），权重转 fp32；TF32 开关保持 PyTorch 默认（不动）。"""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev,
                                            model_configs=[ModelConfig(path=VAE_PATH)],
                                            tokenizer_config=None, redirect_common_files=False)
    assert pipe.vae is not None, "VAE 没加载上"
    pipe.vae.model.float()
    return pipe


# uint8 → [-1, 1] 的查表：在 CPU 上按 infer.encode_video 的写法（.float().div_(127.5).sub_(1.0)）算出 256 个值。
# GPU 上的 div_(标量) 会变成乘倒数（差 1 ulp），查表保证与 CPU 路径逐位相同。
_U8_TO_PM1 = torch.arange(256, dtype=torch.float32).div_(127.5).sub_(1.0)


@torch.no_grad()
def vae_encode(pipe, rgb_u8: torch.Tensor, dev) -> torch.Tensor:
    """uint8 [81,H,W,3] → latent bf16 [48,21,H/16,W/16]（CPU）。

    等价于 infer.encode_video：x = permute(3,0,1,2)/127.5−1，pipe.vae.encode([x], tiled=False)。这里把 encode 的循环体
    （single_encode）直接用在 GPU 上的 x 上，省掉 encode 先把 fp32 视频（832×480 ⇒ 388 MB）搬回 CPU 再搬上来的往返；数值与布局相同
    （verify_prep.py 用官方 encode([x]) 抽查逐位一致）。"""
    lut = _U8_TO_PM1.to(dev)
    x = lut[rgb_u8.to(dev, non_blocking=True).long()].permute(3, 0, 1, 2)        # [3,81,H,W] fp32
    z = pipe.vae.single_encode(x.unsqueeze(0), dev).squeeze(0)
    z = z.to(torch.bfloat16).cpu()
    assert tuple(z.shape) == LAT_SHAPE, z.shape
    return z


def rank_main(rank: int, gpu: int, tasks: list, a: dict, q, parent_pid: int) -> None:
    die_with_parent(parent_pid)                        # 主进程没了（kill / SIGKILL）⇒ rank 跟着退出，不做孤儿
    out, vae = a["out"], not a["no_vae"]
    tag = f"[r{rank} gpu{gpu}]"
    log = lambda *x: print(tag, *x, flush=True)      # noqa: E731
    stats = {"rank": rank, "gpu": gpu, "n": 0, "n_skip": 0, "vae_s": 0.0, "wait_s": 0.0, "tm": collections.Counter()}
    if not tasks:
        q.put(stats)
        return
    rows_path = os.path.join(out, "index", f"rows_rank{rank}.jsonl")
    dl = torch.utils.data.DataLoader(ClipSet(tasks, out, vae), batch_size=None, shuffle=False,
                                     num_workers=a["workers_per_gpu"], prefetch_factor=2,
                                     worker_init_fn=functools.partial(_worker_init, parent_pid=os.getpid()),
                                     multiprocessing_context="fork")
    T0 = time.time()
    it = iter(dl)                                    # 先 fork worker（此时还没碰 CUDA），再加载 VAE
    pipe, dev = None, None
    if vae:
        dev = f"cuda:{gpu}"
        torch.cuda.set_device(dev)
        pipe = load_vae(dev)
        log(f"VAE {type(pipe.vae).__name__} fp32 loaded {time.time() - T0:.1f}s; cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
            f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}; {len(tasks)} clips, {a['workers_per_gpu']} workers")
    stats["t_load"] = time.time() - T0
    t_first = None
    with open_rows(rows_path) as rows_fh:
        while True:
            w = time.time()
            try:
                item = next(it)
            except StopIteration:
                break
            stats["wait_s"] += time.time() - w
            if not item["skip"]:
                if vae:
                    v = time.time()
                    lat = vae_encode(pipe, item["rgb"], dev)
                    pt = dict(item["pt"])
                    pt["latent"] = lat
                    atomic_torch_save(clip_paths(out, item["clip_id"])["latent"], pt)
                    stats["vae_s"] += time.time() - v
                    del item["rgb"]
                for k, x in item["tm"].items():
                    stats["tm"][k] += x
            else:
                stats["n_skip"] += 1
            rows_fh.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
            rows_fh.flush()
            stats["n"] += 1
            if t_first is None:
                t_first = time.time()
            n = stats["n"]
            if n % a["log_every"] == 0 or n == len(tasks):
                el = time.time() - t_first
                rate = (n - 1) / el if el > 0 and n > 1 else float("nan")
                done = max(1, n - stats["n_skip"])
                log(f"{n}/{len(tasks)}  {rate:.3f} clips/s steady  vae {stats['vae_s'] / done:.2f}s  wait {stats['wait_s'] / n:.2f}s  "
                    f"worker total {stats['tm']['total'] / done:.2f}s (dec {stats['tm']['decode'] / done:.2f} lab {stats['tm']['label'] / done:.2f} "
                    f"write {stats['tm']['write'] / done:.2f})  load {os.getloadavg()[0]:.0f}")
    stats.update(t_first=t_first, t_end=time.time(), wall=time.time() - T0,
                 gpu_peak_gib=(torch.cuda.max_memory_allocated(dev) / 2 ** 30) if vae else 0.0)
    stats["tm"] = dict(stats["tm"])
    q.put(stats)


# ----------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------
def merge(out: str, expect=None) -> list[dict]:
    """rows_rank*.jsonl → index/clips.jsonl（按 subset, scene, run, start 排序）+ index/runs.json。"""
    rows = read_rows(out)
    check_rows_res(rows.values(), out)
    order = {s: i for i, s in enumerate(SUBSETS)}
    rows = sorted(rows.values(), key=lambda r: (order[r["subset"]], r["scene"], r["run"], r["start"]))
    if expect is not None:
        miss = sorted(set(expect) - {r["clip_id"] for r in rows})
        if miss:
            raise SystemExit(f"merge：{len(miss)} 个 clip 没有 index 行，例如 {miss[:5]}")
    atomic_write_text(os.path.join(out, "index", "clips.jsonl"),
                      "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    runs = {}
    for r in rows:
        runs[r["run"]] = r["split"]
    runs["_rule"] = SPLIT_RULE
    atomic_write_text(os.path.join(out, "index", "runs.json"), json.dumps(runs, indent=1, ensure_ascii=False) + "\n")
    c = collections.Counter((r["subset"], r["split"], r["drop"], r["has_ext"]) for r in rows)
    print(f"merge: {len(rows)} clips, {len(runs) - 1} runs → index/clips.jsonl")
    for k in sorted(c):
        print(f"   subset={k[0]:<11} split={k[1]:<5} drop={k[2]!s:<5} has_ext={k[3]!s:<5} {c[k]}")
    return rows


def check_rows_res(rows, out: str) -> None:
    """index 行的 res（缺失 = 832x480 的旧行）必须都 = GC_RES：同一个 --out 里不能混两种分辨率的产物。"""
    bad = collections.Counter(r.get("res", GEO.LEGACY_RES) for r in rows if r.get("res", GEO.LEGACY_RES) != GEO.RES_TAG)
    if bad:
        raise SystemExit(f"{out}：已有 {dict(bad)} 的 index 行，本次 {GEO.RES_ENV}={GEO.RES_TAG}；"
                         f"换一个 --out，或设 {GEO.RES_ENV} 为那份数据的分辨率")


def check_out_res(out: str) -> None:
    """--out 里已有产物（prep_args.json 或 index 行）⇒ 它们的 res 必须 = GC_RES（旧根没写 res ⇒ 832x480）。"""
    if os.path.isfile(os.path.join(out, "index", "prep_args.json")):
        r = GEO.root_res(out)
        if r != GEO.RES_TAG:
            raise SystemExit(f"{out} 是 {r} 的数据根（index/prep_args.json），本次 {GEO.RES_ENV}={GEO.RES_TAG}；"
                             f"换一个 --out，或设 {GEO.RES_ENV}={r}")
    check_rows_res(read_rows(out).values(), out)


def copy_text_table(src: str, out: str) -> None:
    dst = os.path.join(out, "text_table_gc25d.pt")
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    tmp = _tmp(dst)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


class CpuMeter:
    """/proc/stat 全机 CPU 利用率采样（其它任务也算在内）。"""

    def __init__(self):
        self.prev, self.samples = self._read(), []

    @staticmethod
    def _read():
        with open("/proc/stat") as fh:
            v = [int(x) for x in fh.readline().split()[1:]]
        return sum(v), v[3] + v[4]

    def sample(self):
        tot, idle = self._read()
        dt, di = tot - self.prev[0], idle - self.prev[1]
        self.prev = (tot, idle)
        if dt > 0:
            self.samples.append((time.time(), 1.0 - di / dt))


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC_DEFAULT, help="recordings/data（<subset>/<scene>/<run>/）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0", help="逗号分隔的 GPU 下标（在 CUDA_VISIBLE_DEVICES 之内），每张卡一个 rank 进程")
    ap.add_argument("--runs", default="", help="只处理这些 run（逗号分隔，run 名跨 subset 唯一）")
    ap.add_argument("--max_runs", type=int, default=0)
    ap.add_argument("--clips_per_run", type=int, default=0, help="冒烟：每 run 均匀取 N 个 clip")
    ap.add_argument("--workers_per_gpu", type=int, default=12)
    ap.add_argument("--no_vae", action="store_true", help="不跑 VAE（不写 latents/，也不解码 RGB）")
    ap.add_argument("--buried", default=BURIED_LIST, help="drop 名单（buried player）")
    ap.add_argument("--text_table", default=TEXT_TABLE_SRC)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--merge_only", action="store_true")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out)
    check_out_res(out)                                  # 先查再建目录：不往别的分辨率的数据根里写
    for d in ("index", "latents", "cond", "cond_ext", "meta"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    runs = list_runs(args.src, [r for r in args.runs.split(",") if r], args.max_runs or None)
    buried = frozenset(x.strip() for x in open(args.buried) if x.strip())
    tasks = plan_tasks(runs, args.clips_per_run or None, buried)
    if args.merge_only:
        merge(out, [t["clip_id"] for t in tasks])
        return
    copy_text_table(args.text_table, out)
    vae = not args.no_vae
    done = done_clips(out, tasks, vae)
    todo = [t for t in tasks if t["clip_id"] not in done]
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    print(f"prep_v10: {len(runs)} runs, {len(tasks)} clips ({len(done)} already done, {len(todo)} to do) → {out}; "
          f"res {GEO.RES_TAG} (crop x0={GEO.CROP_X0} w={GEO.CROP_W}); gpus {gpus} × {args.workers_per_gpu} workers; vae={vae}",
          flush=True)
    with open(os.path.join(out, "index", "prep_args.json"), "w") as fh:
        json.dump({"argv": sys.argv, "args": vars(args), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "torch": torch.__version__, "vae": VAE_PATH, "res": GEO.RES_TAG,
                   "crop_x0": GEO.CROP_X0, "crop_w": GEO.CROP_W}, fh, indent=1)

    T0 = time.time()
    stats = []
    meter = CpuMeter()
    if todo:
        chunks = [list(c) for c in np.array_split(np.array(todo, dtype=object), len(gpus))]
        ctx = torch.multiprocessing.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=rank_main, args=(r, g, chunks[r], vars(args), q, os.getpid()), daemon=False)
                 for r, g in enumerate(gpus)]
        # kill <主进程> ⇒ SystemExit ⇒ finally 收掉 rank（rank / worker 另有 PDEATHSIG 兜底 SIGKILL）
        signal.signal(signal.SIGTERM, lambda sig, _f: sys.exit(128 + sig))
        try:
            for p in procs:
                p.start()
            while any(p.is_alive() for p in procs):
                for p in procs:
                    p.join(timeout=10.0 / len(procs))
                meter.sample()
                while not q.empty():
                    stats.append(q.get())
        finally:
            alive = [p for p in procs if p.is_alive()]
            for p in alive:
                p.terminate()
            for p in alive:
                p.join(timeout=60)
            if alive:
                print(f"prep_v10: 中断，已停掉 {len(alive)} 个 rank；重跑同一命令即可 resume", flush=True)
        while not q.empty():
            stats.append(q.get())
        bad = [(r, p.exitcode) for r, p in enumerate(procs) if p.exitcode != 0]
        if bad:
            raise SystemExit(f"rank 失败（rank, exitcode）：{bad}；已完成的 clip 下次运行会跳过")
    wall = time.time() - T0
    merge(out, [t["clip_id"] for t in tasks])

    # 吞吐 / CPU 汇总
    import resource
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    n_new = sum(s["n"] - s["n_skip"] for s in stats)
    if n_new:
        firsts = [s["t_first"] for s in stats if s.get("t_first")]
        ends = [s["t_end"] for s in stats if s.get("t_first")]
        steady = (n_new - len(firsts)) / max(1e-9, max(ends) - min(firsts)) if firsts else float("nan")
        cpu = [u for t, u in meter.samples if firsts and min(firsts) <= t <= max(ends)]
        tm = collections.Counter()
        for s in stats:
            tm.update(s["tm"])
        summ = {
            "clips": n_new, "gpus": len(gpus), "workers_per_gpu": args.workers_per_gpu, "wall_s": round(wall, 1),
            "clips_per_s_total": round(n_new / wall, 3), "clips_per_s_steady": round(steady, 3),
            "clips_per_s_per_gpu_steady": round(steady / len(gpus), 3),
            "vae_s_per_clip": round(sum(s["vae_s"] for s in stats) / n_new, 3),
            "main_wait_s_per_clip": round(sum(s["wait_s"] for s in stats) / max(1, sum(s["n"] for s in stats)), 3),
            "worker_s_per_clip": {k: round(v / n_new, 3) for k, v in tm.items() if k != "F"},
            "cpu_core_s_per_clip": round((ru.ru_utime + ru.ru_stime) / n_new, 2),
            "machine_cpu_util_mean": round(float(np.mean(cpu)), 3) if cpu else None,
            "machine_cores": os.cpu_count(), "load_avg": os.getloadavg(),
            "rank_load_s": [round(s.get("t_load", 0), 1) for s in stats],
            "gpu_peak_gib": [round(s.get("gpu_peak_gib", 0), 1) for s in stats],
        }
        full = 12800 / (8 * summ["clips_per_s_per_gpu_steady"]) if summ["clips_per_s_per_gpu_steady"] > 0 else float("nan")
        summ["extrapolated_full_12800_on_8gpu_min"] = round(full / 60 + 1, 1)
        print("throughput:", json.dumps(summ, indent=1), flush=True)
        with open(os.path.join(out, "index", f"prep_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as fh:
            json.dump({"summary": summ, "ranks": stats}, fh, indent=1)


if __name__ == "__main__":
    main()
