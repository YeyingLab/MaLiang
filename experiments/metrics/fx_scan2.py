
import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import json, numpy as np, collections, os
P = _mos.environ.get("MALIANG_PROD40", _mos.path.join(MALIANG_ROOT, "data", "production40")); D = P + "/root"
C = _mos.path.join(MALIANG_ROOT, "experiments")
rows = [json.loads(l) for l in open(D + "/index/clips.jsonl")]
test = set(json.load(open(C + "/testset/v1/test_clips.json")))
val = [r for r in rows if r["split"] == "val" and not r["drop"]]
res = []
for r in val:
    ids = np.load(f"{D}/cond/{r['clip_id']}.npz")["ids"][1:81]
    n = (ids >= 22).sum((1, 2)) - (ids == 255).sum((1, 2)) * 0
    n = ((ids >= 22) & (ids < 255)).sum((1, 2))
    res.append((r["clip_id"], int((n >= 300).sum()), float(n.mean())))
fxc = [x for x in res if x[1] >= 8]
print("val clips", len(val), "with >=8 FX frames:", len(fxc), "in paper test:", sum(x[0] in test for x in fxc))
print(collections.Counter(x[0].split("__")[1] for x in fxc))
print(sorted(fxc, key=lambda x: -x[1])[:5])
print(os.listdir(P + "/fx")[:10], os.listdir(P + "/staging")[:10])
m = [json.loads(l) for l in open(P + "/manifest.jsonl")][:2]; print(json.dumps(m)[:800])
