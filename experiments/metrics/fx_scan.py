
import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import json, numpy as np, collections, sys
R = MALIANG_ROOT; C = _mos.path.join(MALIANG_ROOT, "experiments")
D = R + "/data/gc25d_v10_fxcond"
rows = [json.loads(l) for l in open(D + "/index/clips.jsonl")]
print(len(rows), collections.Counter((r["split"], r["drop"]) for r in rows))
print(collections.Counter(r["subset"] for r in rows))
test = set(json.load(open(C + "/testset/v1/test_clips.json")))
val = [r for r in rows if r["split"] != "train" and not r["drop"]]
print("non-train", len(val), collections.Counter(r["subset"] for r in val))
print(collections.Counter(r["run"] for r in val).most_common(20))
res = []
for r in val:
    ids = np.load(f"{D}/cond/{r['clip_id']}.npz")["ids"][1:81]
    n = ((ids >= 22) & (ids <= 25)).sum((1, 2)); res.append((r["clip_id"], int((n >= 300).sum()), float(n.mean())))
fxc = [x for x in res if x[1] >= 8]
print("non-train clips with >=8 FX frames (>=300px):", len(fxc), "in test set:", sum(x[0] in test for x in fxc))
print(collections.Counter(x[0].rsplit("__", 1)[0] for x in fxc))
print(fxc[:5])
print(json.dumps(rows[0])[:500])
