
import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import json, numpy as np, torch, os
P = _mos.path.join(_mos.environ.get("MALIANG_PROD40", _mos.path.join(MALIANG_ROOT, "data", "production40")), "root"); D = _mos.path.join(MALIANG_ROOT, "data", "gc25d_v10_fxcond")
man = {m["clip"]: m for m in map(json.loads, open(_mos.path.join(_mos.environ.get("MALIANG_PROD40", _mos.path.join(MALIANG_ROOT, "data", "production40")), "manifest.jsonl")))}
rows = [json.loads(l) for l in open(D + "/index/clips.jsonl")]
tr = [r["clip_id"] for r in rows if r["split"] == "train" and not r["drop"]]
both, same_lat, n_old_fx = 0, 0, 0; ex = []
for c in tr[:4000]:
    a = np.load(f"{D}/cond/{c}.npz")["ids"]; old = (a >= 22) & (a < 255)
    if not old.any(): continue
    n_old_fx += 1
    if man.get(c, {}).get("status") != "written": continue
    b = np.load(f"{P}/cond/{c}.npz")["ids"]; new = (b >= 22) & (b < 255)
    inter = (old & new).sum(); uni = (old | new).sum()
    la = torch.load(f"{D}/latents/{c}.pt", map_location="cpu", weights_only=False); lb = torch.load(f"{P}/latents/{c}.pt", map_location="cpu", weights_only=False)
    ta = la if torch.is_tensor(la) else la.get("latents", next(iter(la.values()))); tb = lb if torch.is_tensor(lb) else lb.get("latents", next(iter(lb.values())))
    d = float((ta.float() - tb.float()).abs().mean()) if ta.shape == tb.shape else None
    both += 1; ex.append((c, int(old.sum()), int(new.sum()), round(inter / max(uni, 1), 3), d))
    if len(ex) >= 12: break
print("old-FX train clips scanned:", n_old_fx, "rewritten in production40:", both)
for e in ex: print(e)
