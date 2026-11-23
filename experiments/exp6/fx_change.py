#!/usr/bin/env python3
"""FX injection test, appearance-change metric (no segmenter). base.mp4 and fx.mp4 are two rollouts of the same clip with the
same seed; they differ only in the injected FX ids from frame E0 on. Per frame f >= E0:
  change_in  = share of injected-FX pixels whose colour changed (mean |fx - base| over RGB > T)
  change_out = same share outside the 49x49-dilated injected mask (side effects elsewhere in the frame)
  psnr_out   = PSNR(fx, base) outside the dilated mask
Reported per clip and pooled; also change_in for T in (15, 25, 40) and the first frame where change_in >= half its clip mean.
Usage: fx_change.py DIR"""
import glob, json, sys, math
import numpy as np, cv2
D = sys.argv[1]; E0 = 37; TS = (15, 25, 40)
def frames(p):
    cap = cv2.VideoCapture(p); f = []
    while True:
        ok, im = cap.read()
        if not ok: break
        f.append(im.astype(np.int16))
    return f
res = []
for rj in sorted(glob.glob(f"{D}/*/result.json")):
    d = rj.rsplit("/", 1)[0]; r = json.load(open(rj)); b = frames(f"{d}/base.mp4"); x = frames(f"{d}/fx.mp4")
    m = np.load(f"{d}/inject.npz")["mask"]; cin = {t: [] for t in TS}; cout, pout = [], []
    for k in range(min(len(m), len(b) - E0)):
        f = E0 + k; diff = np.abs(x[f] - b[f]).mean(2); mk = m[k]
        out = cv2.dilate(mk.astype(np.uint8), np.ones((49, 49), np.uint8)) == 0
        if mk.sum() >= 50:
            for t in TS: cin[t].append(float((diff[mk] > t).mean()))
        cout.append(float((diff[out] > 25).mean()))
        sq = ((x[f] - b[f]).astype(np.float32) ** 2)[out].mean(); pout.append(10 * math.log10(255 ** 2 / max(sq, 1e-8)))
    ci = cin[25]; mean = float(np.mean(ci)) if ci else None
    lat = next((i for i, v in enumerate(ci) if v >= 0.5 * mean), None) if mean else None
    res.append(dict(clip_id=r["clip_id"], fx_px_mean=r["fx_px_mean"], change_in={t: float(np.mean(cin[t])) if cin[t] else None for t in TS},
                    change_out=float(np.mean(cout)), psnr_out=float(np.mean(pout)), latency_frames=lat, seg_fx_recall=r["fx_recall"]))
    print(r["clip_id"][:44], {t: round(res[-1]["change_in"][t] or 0, 3) for t in TS}, round(res[-1]["change_out"], 3), lat, flush=True)
agg = dict(n=len(res), change_in={t: float(np.mean([q["change_in"][t] for q in res if q["change_in"][t] is not None])) for t in TS},
           change_out=float(np.mean([q["change_out"] for q in res])), psnr_out=float(np.mean([q["psnr_out"] for q in res])),
           latency_frames_median=float(np.median([q["latency_frames"] for q in res if q["latency_frames"] is not None])),
           seg_fx_recall=float(np.mean([q["seg_fx_recall"] for q in res if q["seg_fx_recall"] is not None])))
json.dump(dict(summary=agg, clips=res), open(f"{D}/change_summary.json", "w"), indent=1); print(json.dumps(agg, indent=1))
