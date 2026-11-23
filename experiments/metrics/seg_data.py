#!/usr/bin/env python3
"""Pairs for the mIoU segmenter: recorded RGB frame (832x480, geometry.resample_rgb) <-> condition ids of the same frame.
Labels (metrics.SEM_CLASSES order): 0 sky, 1 terrain, 2 paving, 3 architecture, 4 rock, 5 vegetation, 6 prop,
7 player (10, 12..16), 8 boss (11, 17..21); fx 22..25 and 255 -> 255 (ignored).
train: split == train, drop == False, n_train clips stratified by subset, frames 0/27/54/80.
val:   split == val minus every clip of the paper test set's runs, frames 0/40 (segmenter accuracy on held-out GT).
Usage: seg_data.py --out <dir> [--n_train 2400] [--workers 16]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, os, sys, random, collections
from multiprocessing import Pool
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import numpy as np, cv2
import geometry as GEO
import prep_v10 as PV
DATA = RT + "/data/gc25d_v10_fxcond"
RAW = _mos.environ.get("MALIANG_RAW", _mos.path.join(MALIANG_ROOT, "data", "recordings", "data"))
TEST = _mos.path.join(MALIANG_ROOT, "experiments", "testset", "v1", "test_clips.json")
LUT = np.full(256, 255, np.uint8)
LUT[:7] = np.arange(7)
LUT[[10, 12, 13, 14, 15, 16]] = 7
LUT[[11, 17, 18, 19, 20, 21]] = 8


def work(job):
    out, part, run_key, items = job
    subset, scene, run = run_key
    t = PV.RunTars(f"{RAW}/{subset}/{scene}/{run}")
    n = 0
    for cid, start, frames in items:
        ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"]
        for k in frames:
            img = cv2.imdecode(np.frombuffer(t.get(int(start) + k, "jpg"), np.uint8), cv2.IMREAD_COLOR)
            rgb = GEO.resample_rgb(img[..., ::-1])
            name = f"{cid}__f{k:03d}"
            cv2.imwrite(f"{out}/{part}/img/{name}.jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{out}/{part}/lbl/{name}.png", LUT[ids[k]])
            n += 1
    for fh in t.fh.values():
        fh.close()
    return n


def main():
    a = argparse.ArgumentParser(); a.add_argument("--out", required=True)
    a.add_argument("--n_train", type=int, default=2400); a.add_argument("--workers", type=int, default=16)
    args = a.parse_args()
    test = set(json.load(open(TEST)))
    rows = [json.loads(l) for l in open(DATA + "/index/clips.jsonl")]
    test_runs = {(r["subset"], r["scene"], r["run"]) for r in rows if r["clip_id"] in test}
    rng = random.Random(20261112)
    tr = [r for r in rows if r["split"] == "train" and not r["drop"] and (r["subset"], r["scene"], r["run"]) not in test_runs]
    by = collections.defaultdict(list)
    for r in tr:
        by[r["subset"]].append(r)
    pick = []
    for s, lst in sorted(by.items()):
        rng.shuffle(lst); pick += lst[: round(args.n_train * len(lst) / len(tr))]
    va = [r for r in rows if r["split"] == "val" and not r["drop"] and (r["subset"], r["scene"], r["run"]) not in test_runs]
    jobs = []
    for part, lst, frames in (("train", pick, (0, 27, 54, 80)), ("val", va, (0, 40))):
        os.makedirs(f"{args.out}/{part}/img", exist_ok=True); os.makedirs(f"{args.out}/{part}/lbl", exist_ok=True)
        g = collections.defaultdict(list)
        for r in lst:
            g[(r["subset"], r["scene"], r["run"])].append((r["clip_id"], r["start"], frames))
        jobs += [(args.out, part, k, v) for k, v in g.items()]
    with Pool(args.workers) as p:
        n = sum(p.imap_unordered(work, jobs))
    json.dump(dict(train_clips=len(pick), val_clips=len(va), pairs=n, test_runs_excluded=sorted(map(list, test_runs)),
                   classes=["sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop", "player", "boss"]),
              open(f"{args.out}/info.json", "w"), indent=1)
    print("DONE", n)


if __name__ == "__main__":
    main()
