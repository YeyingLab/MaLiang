
import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import json, numpy as np, random, sys
D = _mos.path.join(MALIANG_ROOT, "data", "gc25d_v10_fxcond")
rows = [json.loads(l) for l in open(D + "/index/clips.jsonl")]
print(len(rows), list(rows[0].keys())[:20])
tr = [r for r in rows if r.get("split", "train") == "train"]
random.seed(0); smp = random.sample(tr, 400)
st = {}
for r in smp:
    ids = np.load(f"{D}/cond/{r['clip_id']}.npz")["ids"][::8]
    t = (ids == 1).mean(); p = (ids == 2).mean()
    ps = lambda g: (g == 2).sum() / max(1, ((g == 1) | (g == 2)).sum())
    st.setdefault(r.get("subset"), []).append((t, p, abs(ps(ids[0]) - ps(ids[-1]))))
for k, v in st.items():
    v = np.array(v)
    print(k, len(v), "terrain %.3f paving %.3f" % (v[:, 0].mean(), v[:, 1].mean()),
          "clips with paving>1%%: %.2f" % ((v[:, 1] > 0.01).mean()), "mean |dpaving share| first-last %.3f" % v[:, 2].mean(),
          "clips with change>0.1: %.3f" % ((v[:, 2] > 0.1).mean()))
