#!/usr/bin/env python3
"""Score experiment 6 cases (procmap unseen maps, sketch maps): segment the generated frames 1..80 with the mIoU
segmenter and compare with the numeric map the video was generated from (no recorded GT exists for these worlds).
Per case: mIoU (classes present in the map), per-class IoU, entity metrics (eval_common); grouped by theme / sketch kind.
Usage: exp6_score.py --cases DIR [DIR ...] [--gen_sub gen_v2_color_light] --out summary.json
       exp6_score.py --cases DIR [DIR ...] --video_dir solar/renders_unseen --kind canvas24 --proxy solar --out s.json
       (baseline renders named <case>.mp4, mapped to 832x480 / 16 fps frames 0..80 exactly like eval_common.load_method)"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, glob, json, os, sys
HERE = _mos.path.join(MALIANG_ROOT, "experiments", "metrics")
sys.path.insert(0, HERE)
import numpy as np
from eval_common import Scorer, LUT, read_video, load_method, conf_matrix, miou_from_conf, NAMES, entity_frame_stats, entity_summary


def main():
    a = argparse.ArgumentParser(); a.add_argument("--cases", nargs="+", required=True); a.add_argument("--out", required=True)
    a.add_argument("--gen_sub", default="gen", help="generation sub-dir: gen (variant 3) or gen_<variant>")
    a.add_argument("--video_dir", default=None, help="baseline renders <video_dir>/<case>.mp4 instead of <case>/<gen_sub>/")
    a.add_argument("--kind", default="native", choices=("native", "canvas24"))
    a.add_argument("--proxy", default=None, help="none / abl / solar (default: abl for gen_v1*/gen_v2*, else none)")
    args = a.parse_args(); sc = Scorer(MALIANG_ROOT + "/assets/seg_b2_final")
    per, groups, ents = [], {}, {}
    proxy = args.proxy or ("abl" if args.gen_sub.startswith(("gen_v1", "gen_v2")) else "none")
    for root in args.cases:
        for d in sorted(glob.glob(f"{root}/*/")):
            case = os.path.basename(d.rstrip("/"))
            g = ([f"{args.video_dir}/{case}.mp4"] if os.path.exists(f"{args.video_dir}/{case}.mp4") else []) if args.video_dir \
                else glob.glob(f"{d}/{args.gen_sub}/*_gen.mp4")
            if not g or not os.path.exists(f"{d}/ids.npz"):
                continue
            v = load_method(g[0], args.kind)[:81] if args.video_dir else read_video(g[0])[:81]; raw = np.load(f"{d}/ids.npz")["ids"][: len(v)]; lbl = LUT[raw]
            sg = sc.segment(v[1:]); c = conf_matrix(sg, lbl[1:]); m, iou = miou_from_conf(c)
            ent = entity_frame_stats(sg, lbl[1:], raw[1:], v[1:], proxy)
            kind = case.split("__")[-1] if not case.startswith(("synth", "human")) else case.split("__")[0].rstrip("0123456789")
            per.append(dict(case=case, kind=kind, miou=m, iou=iou)); groups.setdefault(kind, np.zeros((9, 9), np.int64))
            groups[kind] += c; ents.setdefault(kind, []).append(ent); print(case, round(m, 4), flush=True)
    summ = {k: dict(n=sum(p["kind"] == k for p in per), miou_pooled=miou_from_conf(cm)[0],
                    miou_case_mean=float(np.mean([p["miou"] for p in per if p["kind"] == k])), iou=miou_from_conf(cm)[1],
                    entity=entity_summary(ents[k]))
            for k, cm in groups.items()}
    json.dump(dict(groups=summ, per_case=per), open(args.out, "w"), indent=1)
    print(json.dumps({k: (v["n"], round(v["miou_pooled"], 4)) for k, v in summ.items()}))


if __name__ == "__main__":
    main()
