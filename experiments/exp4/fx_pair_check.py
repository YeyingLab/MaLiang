
import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import json, numpy as np, sys, os
C = _mos.path.join(MALIANG_ROOT, "experiments"); sys.path.insert(0, C + "/exp4")
import exp4_edit as X
import torch
P = _mos.path.join(_mos.environ.get("MALIANG_PROD40", _mos.path.join(MALIANG_ROOT, "data", "production40")), "root")
rows = [json.loads(l) for l in open(P + "/index/clips.jsonl")]
val = [r["clip_id"] for r in rows if r["split"] == "val" and not r["drop"]]
sel = []
for c in val:
    a = np.load(f"{P}/cond/{c}.npz")["ids"]; n = ((a[1:81] >= 22) & (a[1:81] < 255)).sum((1, 2))
    if (n >= 300).sum() >= 8 and os.path.exists(f"{X.DATA}/cond_ext/{c}.npz"):
        sel.append(c)
print("selected", len(sel))
vocab = "parts_fx"
ds = X.DG.GcRolloutDataset(X.DATA, "val", X.DATA + "/text_table_gc25d.pt", vocab=vocab, n_cells=X.CELLS, verbose=False)
s = ds.load(sel[0]); si = s["sem_ids"]; si = si.numpy() if torch.is_tensor(si) else np.asarray(si)
print("sem_ids", si.shape, si.dtype, np.unique(si)[:30], "CELLS", X.CELLS, "E0", X.E0)
base = np.load(f"{X.DATA}/cond/{sel[0]}.npz")["ids"]; fx = np.load(f"{P}/cond/{sel[0]}.npz")["ids"]
print("base", base.shape, "fx", fx.shape, "sem==base[:len]", bool((si[:len(base)] == base[:len(si)]).all()) if si.shape[1:] == base.shape[1:] else "shape differs")
bad = 0; fxpx = 0
for c in sel:
    b = np.load(f"{X.DATA}/cond/{c}.npz")["ids"]; f = np.load(f"{P}/cond/{c}.npz")["ids"]
    d = b != f; isfx = (f >= 22) & (f < 255)
    bad += int((d & ~isfx).sum()); fxpx += int(isfx.sum())
print("pixels differing outside FX:", bad, "FX px:", fxpx)
print("ext:", os.path.exists(f"{P}/cond_ext/{sel[0]}.npz"), os.path.exists(f"{X.DATA}/cond_ext/{sel[0]}.npz"))
json.dump(sel, open(C + "/exp4/fx_natural_clips.json", "w"))
