#!/usr/bin/env python3
"""Rows = cases; columns = frames 0/40/80 of each generation sub-dir given (e.g. gen gen_ours)."""
import glob, os, sys
import numpy as np, cv2
out, subs, cases = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
def fr(p, ks):
    cap = cv2.VideoCapture(p); f = []
    while True:
        ok, im = cap.read()
        if not ok: break
        f.append(im)
    return [cv2.resize(f[min(k, len(f) - 1)], (320, 184)) for k in ks]
rows = []
for c in cases:
    t = []
    for s in subs:
        g = glob.glob(f"{c}/{s}/*_gen.mp4")[0]; im = fr(g, (0, 40, 80))
        cv2.putText(im[0], f"{s} {os.path.basename(c)[:30]}", (4, 14), 0, 0.4, (255, 255, 255), 1); t += im
    rows.append(np.concatenate(t, 1))
cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80]); print("ok")
