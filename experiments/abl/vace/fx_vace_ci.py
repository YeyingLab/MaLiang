#!/usr/bin/env python3
"""FX control, interface comparison: per-variant means and paired bootstrap CIs (clips resampled, B=2000) of
(3) - (1) and (3) - (2) on the clips finished by all variants. Also the deploy student for reference (not paired by sampler).
Output: experiments/exp6/fx_offline_ci.json"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import glob, json, os
import numpy as np
C = _mos.path.join(MALIANG_ROOT, "experiments", "exp6")
V = ["v5_color_vace", "v3_ids"]
K = ["change_in", "change_out", "psnr_out", "timing_acc"]
res = {v: {json.load(open(f))["clip_id"]: json.load(open(f)) for f in glob.glob(f"{C}/fx_offline_{v}/*/result.json")} for v in V}
dep = {json.load(open(f))["clip_id"]: json.load(open(f)) for f in glob.glob(f"{C}/fx_natural_ours/*/result.json")}
common = sorted(set.intersection(*[set(r) for r in res.values()]))
common = [c for c in common if all(res[v][c]["change_in"] is not None for v in V)]
out = {"n": len(common), "means": {}, "diff": {}}
X = {v: {k: np.array([res[v][c][k] for c in common], float) for k in K} for v in V}
for v in V:
    out["means"][v] = {k: float(X[v][k].mean()) for k in K}
out["means"]["deploy"] = {k: float(np.mean([dep[c][k] for c in common if dep.get(c) and dep[c][k] is not None])) for k in K}
rng = np.random.default_rng(0); idx = rng.integers(0, len(common), (2000, len(common)))
for ref in ("v5_color_vace",):
    for k in K:
        d = X["v3_ids"][k] - X[ref][k]; bs = d[idx].mean(1)
        out["diff"][f"v3-{ref}:{k}"] = dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                            p=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())))
json.dump(out, open(_mos.path.join(MALIANG_ROOT, "experiments", "abl", "vace", "fx_vace_ci.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

# recording-level bootstrap (clips of one recording resampled together), same rule as the main tables
runs = sorted(set("__".join(c.split("__")[:2]) for c in common)); rid = np.array([runs.index("__".join(c.split("__")[:2])) for c in common])
out["n_runs"] = len(runs); out["diff_run"] = {}
rng = np.random.default_rng(1)
for ref in ("v5_color_vace",):
    for k in K:
        d = X["v3_ids"][k] - X[ref][k]; bs = []
        for _ in range(2000):
            pick = rng.integers(0, len(runs), len(runs)); sel = np.concatenate([np.nonzero(rid == r)[0] for r in pick])
            bs.append(d[sel].mean())
        bs = np.array(bs)
        out["diff_run"][f"v3-{ref}:{k}"] = dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                                p=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())))
json.dump(out, open(_mos.path.join(MALIANG_ROOT, "experiments", "abl", "vace", "fx_vace_ci.json"), "w"), indent=1)
print("RUN", json.dumps({k: [round(v["mean"], 4), round(v["lo"], 4), round(v["hi"], 4), v["p"]] for k, v in out["diff_run"].items()}, indent=0), "n_runs", len(runs))
