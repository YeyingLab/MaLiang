#!/usr/bin/env python3
"""Contact sheet for the FX injection test: rows = clips, columns = base | injected run | injected-id overlay at 3 frames."""
import glob, json, os, sys
import numpy as np, cv2
D = sys.argv[1]; out = sys.argv[2]; names = sys.argv[3:]
def frames(p):
    cap = cv2.VideoCapture(p); f = []
    while True:
        ok, im = cap.read()
        if not ok: break
        f.append(im)
    return f
rows = []
for n in names:
    d = glob.glob(f"{D}/{n}*")[0]; b = frames(f"{d}/base.mp4"); x = frames(f"{d}/fx.mp4"); z = np.load(f"{d}/inject.npz")
    m = z["mask"]; E0 = 37
    for k in (12, 28, 40):
        f = E0 + k; o = x[f].copy(); o[m[k]] = (0.5 * o[m[k]] + [0, 0, 127]).astype(np.uint8)
        t = [cv2.resize(i, (416, 240)) for i in (b[f], x[f], o)]
        cv2.putText(t[0], f"{os.path.basename(d)[:28]} f{f}", (4, 16), 0, 0.45, (255, 255, 255), 1)
        rows.append(np.concatenate(t, 1))
cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("ok", len(rows))
