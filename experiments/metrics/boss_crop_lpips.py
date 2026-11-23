#!/usr/bin/env python3
"""All-clip boss-crop LPIPS for the interface-ablation teachers (1)(2)(3): every 4th generated frame (1..80) whose map
has >= 1500 boss pixels; crop = the Fig.-4 box around the boss (scale 1.5, 4:5); LPIPS vs the recorded frame.
Clip means, then paired bootstrap over clips and over recordings (B=2000) of (3)-(1) and (3)-(2).
Out: experiments/main/eval/boss_crop_lpips.json"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import json, sys
C = _mos.path.join(MALIANG_ROOT, "experiments"); sys.path.insert(0, C + "/metrics")
import numpy as np
import eval_common as EC
BOSS = [11, 17, 18, 19, 20, 21]; AR = 0.8
V = {"v1": "v1_color_vae", "v2": "v2_color_light", "v3": "v3_ids"}


def box(m, scale=1.5, lo=220):
    ys, xs = np.nonzero(m); cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = int(min(480 / AR, max(lo, scale * max(xs.max() - xs.min(), (ys.max() - ys.min()) / AR)))); sh = int(side * AR)
    x0 = int(min(max(0, cx - side / 2), 832 - side)); y0 = int(min(max(0, cy - sh / 2), 480 - sh)); return x0, y0, x0 + side, y0 + sh


sc = EC.Scorer(EC.MALIANG_ROOT + "/assets/seg_b2_final"); per = {}
for c in json.load(open(EC.TEST)):
    ids = np.load(f"{EC.DATA}/cond/{c}.npz")["ids"][:81]; fr = [f for f in range(1, 81, 4) if np.isin(ids[f], BOSS).sum() >= 1500]
    if not fr: continue
    gt = np.load(f"{C}/main/eval/_gt/{c}.npz")["gt"]
    vids = {k: EC.load_method(f"{C}/main/abl/{v}/{c}_gen.mp4", "native") for k, v in V.items()}
    r = {k: [] for k in V}
    for f in fr:
        x0, y0, x1, y1 = box(np.isin(ids[f], BOSS)); g = gt[f][y0:y1, x0:x1]
        for k in V:
            r[k].append(float(sc.lpips(vids[k][f][y0:y1, x0:x1][None], g[None])[0]))
    per[c] = {k: float(np.mean(v)) for k, v in r.items()}; print(c[-30:], len(fr), {k: round(v, 3) for k, v in per[c].items()}, flush=True)
cl = sorted(per); X = {k: np.array([per[c][k] for c in cl]) for k in V}
runs = sorted(set("__".join(c.split("__")[:2]) for c in cl)); rid = np.array([runs.index("__".join(c.split("__")[:2])) for c in cl])
out = {"n_clips": len(cl), "mean": {k: float(X[k].mean()) for k in V}, "diff": {}}
rng = np.random.default_rng(0)
for ref in ("v1", "v2"):
    d = X["v3"] - X[ref]
    bc = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    br = []
    for _ in range(2000):
        pick = rng.integers(0, len(runs), len(runs)); br.append(d[np.concatenate([np.nonzero(rid == q)[0] for q in pick])].mean())
    br = np.array(br)
    out["diff"][f"v3-{ref}"] = dict(mean=float(d.mean()), clip=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
                                    run=[float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))])
json.dump({"summary": out, "per_clip": per}, open(f"{C}/main/eval/boss_crop_lpips.json", "w"), indent=1)
print(json.dumps(out, indent=1))
