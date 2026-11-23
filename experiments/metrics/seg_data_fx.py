#!/usr/bin/env python3
"""Segmenter data with a 10th class "fx" (raw ids 22..25 = boss attack effects), for the new-category experiment.
  * relabels every existing pair of segdata/ (same jpg, symlinked) with fx -> 9 instead of 255;
  * adds recorded frames that contain fx (from fx-bearing train clips, frames where fx covers >= 300 px), so the class has
    enough examples; held-out = whole fx-bearing runs (~10 %), never runs of the paper test set.
Labels: 0..8 as seg_data.py, 9 fx, 255 ignore.
Usage: seg_data_fx.py --src segdata --out segdata_fx [--fx_clips 1500] [--workers 16]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, glob, json, os, sys, random, collections
from multiprocessing import Pool
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import numpy as np, cv2
import geometry as GEO
import prep_v10 as PV
DATA = RT + "/data/gc25d_v10_fxcond"
RAW = _mos.environ.get("MALIANG_RAW", _mos.path.join(MALIANG_ROOT, "data", "recordings", "data"))
TEST = _mos.path.join(MALIANG_ROOT, "experiments", "testset", "v1", "test_clips.json")
LUT = np.full(256, 255, np.uint8); LUT[:7] = np.arange(7)
LUT[[10, 12, 13, 14, 15, 16]] = 7; LUT[[11, 17, 18, 19, 20, 21]] = 8; LUT[[22, 23, 24, 25]] = 9


def scan(cid):
    ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"]
    n = (ids >= 22).sum((1, 2))
    return cid, [int(k) for k in np.nonzero(n >= 300)[0]]


def relabel(args):
    src, out, part, name = args
    cid, k = name.rsplit("__f", 1)
    ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"][int(k)]
    cv2.imwrite(f"{out}/{part}/lbl/{name}.png", LUT[ids])
    dst = f"{out}/{part}/img/{name}.jpg"
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(f"{src}/{part}/img/{name}.jpg"), dst)
    return 1


def extract(job):
    out, part, run_key, items = job
    t = PV.RunTars(f"{RAW}/{run_key[0]}/{run_key[1]}/{run_key[2]}")
    n = 0
    for cid, start, frames in items:
        ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"]
        for k in frames:
            name = f"{cid}__f{k:03d}"
            if os.path.exists(f"{out}/{part}/lbl/{name}.png"):
                continue
            img = cv2.imdecode(np.frombuffer(t.get(int(start) + k, "jpg"), np.uint8), cv2.IMREAD_COLOR)
            rgb = GEO.resample_rgb(img[..., ::-1])
            cv2.imwrite(f"{out}/{part}/img/{name}.jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{out}/{part}/lbl/{name}.png", LUT[ids[k]]); n += 1
    for fh in t.fh.values():
        fh.close()
    return n


def main():
    a = argparse.ArgumentParser(); a.add_argument("--src", required=True); a.add_argument("--out", required=True)
    a.add_argument("--fx_clips", type=int, default=1500); a.add_argument("--workers", type=int, default=16)
    args = a.parse_args()
    rows = [json.loads(l) for l in open(DATA + "/index/clips.jsonl")]
    rk = lambda r: (r["subset"], r["scene"], r["run"])
    test = set(json.load(open(TEST))); test_runs = {rk(r) for r in rows if r["clip_id"] in test}
    src_info = json.load(open(f"{args.src}/info.json")); old_val_runs = {tuple(x) for x in src_info.get("val_runs", [])}
    for part in ("train", "val"):
        os.makedirs(f"{args.out}/{part}/img", exist_ok=True); os.makedirs(f"{args.out}/{part}/lbl", exist_ok=True)
    jobs = [(args.src, args.out, part, os.path.basename(p)[:-4]) for part in ("train", "val")
            for p in sorted(glob.glob(f"{args.src}/{part}/img/*.jpg"))]
    tr = [r for r in rows if r["split"] == "train" and not r["drop"] and rk(r) not in test_runs]
    with Pool(args.workers) as p:
        n_rel = sum(p.imap_unordered(relabel, jobs, chunksize=64))
        hits = [h for h in p.imap_unordered(scan, [r["clip_id"] for r in tr], chunksize=32) if h[1]]
    byid = {r["clip_id"]: r for r in tr}
    fx_runs = sorted({rk(byid[c]) for c, _ in hits})
    rng = random.Random(20261112); rng.shuffle(fx_runs)
    held = set(fx_runs[: max(1, len(fx_runs) // 10)]) | (old_val_runs & set(fx_runs))
    rng.shuffle(hits); hits = hits[: args.fx_clips]
    g = collections.defaultdict(list)
    for cid, ks in hits:
        r = byid[cid]; part = "val" if rk(r) in held else "train"
        pick = ks if len(ks) <= 4 else [ks[i] for i in np.linspace(0, len(ks) - 1, 4).round().astype(int)]
        g[(part, rk(r))].append((cid, r["start"], pick))
    with Pool(args.workers) as p:
        n_new = sum(p.imap_unordered(extract, [(args.out, part, k, v) for (part, k), v in g.items()]))
    # any pair (old or new) whose run is a held-out fx run must sit in val
    moved = 0
    for p_ in glob.glob(f"{args.out}/train/lbl/*.png"):
        b = os.path.basename(p_)[:-4]; sc, run = b.split("__")[0], b.split("__")[1]
        if any((s, sc, run) in held for s in ("v10_real", "v9_stylized")):
            os.rename(p_, f"{args.out}/val/lbl/{b}.png"); os.rename(f"{args.out}/train/img/{b}.jpg", f"{args.out}/val/img/{b}.jpg"); moved += 1
    info = dict(relabelled=n_rel, fx_clips_total=len([1 for _ in fx_runs]), fx_clips_used=len(hits), fx_frames_added=n_new,
                fx_runs=len(fx_runs), held_fx_runs=sorted(map(list, held)), moved_to_val=moved,
                classes=["sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop", "player", "boss", "fx"])
    json.dump(info, open(f"{args.out}/info.json", "w"), indent=1); print("DONE", info)


if __name__ == "__main__":
    main()
