#!/usr/bin/env python3
"""CWM (code-world-model, MiniMax-H3 Ref2VA) inputs for the paper test set (same 100 clips as SolarRenderer / AGR).
CWM consumes, per frame at 24 fps, a metric depth map (float32 metres, 336x192) + a 12-class semantic id png (336x192),
plus an RGB anchor (frame 0) and one prompt per 124-frame window. Our data has no depth, so depth is built from structure
only (never from the recorded RGB):
  * relative/metric depth = Depth-Anything-V2-Metric-Outdoor-Large on the grey whitebox render of the same frame;
  * per-clip scale = median over frames of per-frame median( analytic ground-plane depth / predicted depth ) over terrain+paving pixels, where the analytic
    depth comes from the recorded camera (meta world_to_clip / world_to_view) intersected with Z = ground_z;
  * sky (id 0) -> 0 (invalid, CWM convention).
Semantic map ours -> CWM: 0 sky->1, 1 terrain->3, 2 paving->4, 3 architecture->6, 4 rock->3, 5 vegetation->5, 6 prop->11,
player parts->8 human, boss parts->9 animal, fx 22..25 / 255 -> 0 void.
Frames: clip frames s..s+104 at 16 fps -> 24 fps index i*16//24, first 124 (= one CWM window); same canvas mapping
(832x480 -> 1344x776 -> centre crop 1344x768) as the other baselines, then -> 336x192.
Usage: build_cwm_inputs.py --testset <v1 dir> --out <dir> [--device cuda]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, os, sys
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import numpy as np, cv2, torch
from PIL import Image
import geometry as GEO
import prep_v10 as PV
DATA = RT + "/data/gc25d_v10_fxcond"
RAW = _mos.environ.get("MALIANG_RAW", _mos.path.join(MALIANG_ROOT, "data", "recordings", "data"))
W, H = GEO.DST_W, GEO.DST_H
CW, CH = 336, 192
OUT_W, OUT_H, PAD_H = 1344, 768, 776
NF = 124
LUT = np.zeros(256, np.uint8)
for a, b in {0: 1, 1: 3, 2: 4, 3: 6, 4: 3, 5: 5, 6: 11}.items():
    LUT[a] = b
LUT[[10, 12, 13, 14, 15, 16]] = 8
LUT[[11, 17, 18, 19, 20, 21]] = 9
DA_REPO = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def to_cond(img, interp):
    x = cv2.resize(img, (OUT_W, PAD_H), interpolation=interp)
    o = (PAD_H - OUT_H) // 2
    return cv2.resize(x[o:o + OUT_H], (CW, CH), interpolation=interp)


def dst_pixel_rays(world_to_clip, world_to_view, xs, ys):
    """copy of capsules.src_pixel_rays(dst -> src pixel centres); capsules imports the training stack, so it is inlined."""
    sx = GEO._IDX_X[np.asarray(xs)].astype(np.float64); sy = GEO._IDX_Y[np.asarray(ys)].astype(np.float64)
    M = np.asarray(world_to_clip, np.float64).reshape(4, 4); V = np.asarray(world_to_view, np.float64).reshape(4, 4)
    Mi = np.linalg.inv(M)
    nx = ((sx + 0.5) / GEO.SRC_W - 0.5) * 2; ny = (0.5 - (sy + 0.5) / GEO.SRC_H) * 2

    def unproj(z):
        q = np.stack([nx, ny, np.full_like(nx, z), np.ones_like(nx)], 1) @ Mi
        return q[:, :3] / q[:, 3:4]
    p0, p1 = unproj(0.9), unproj(0.1)
    rd = p1 - p0; rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    fwd = V[:3, 2] / np.linalg.norm(V[:3, 2])
    if len(rd) and (rd @ fwd).mean() < 0:
        rd = -rd
    return p0 - rd * 5000.0, rd


_XY = None


def ground_depth_m(M, V, z0):
    """analytic view-space depth (m) of the plane Z = z0 for every W x H pixel; NaN where the ray misses."""
    global _XY
    if _XY is None:
        ys, xs = np.mgrid[0:H, 0:W]
        _XY = (xs.ravel(), ys.ravel())
    ro, rd = dst_pixel_rays(M, V, *_XY)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z0 - ro[:, 2]) / rd[:, 2]
    P = ro + t[:, None] * rd
    Vm = np.asarray(V, np.float64).reshape(4, 4)
    z = (np.concatenate([P, np.ones((len(P), 1))], 1) @ Vm)[:, 2] / 100.0
    z[~(t > 0) | ~np.isfinite(z) | (z <= 0)] = np.nan
    return z.reshape(H, W)


def main():
    a = argparse.ArgumentParser(); a.add_argument("--testset", required=True); a.add_argument("--out", required=True)
    a.add_argument("--device", default="cuda"); a.add_argument("--only", default=None)
    args = a.parse_args()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(DA_REPO)
    net = AutoModelForDepthEstimation.from_pretrained(DA_REPO, torch_dtype=torch.float16).to(args.device).eval()
    man = json.load(open(f"{args.testset}/manifest.json"))
    rows = {json.loads(l)["clip_id"]: json.loads(l) for l in open(DATA + "/index/clips.jsonl")}
    idx = [int(i * 16 // 24) for i in range(NF)]
    tars, report = {}, []
    for j, m in enumerate(man["clips"]):
        cid = m["clip_id"]
        if args.only and cid not in args.only.split(","):
            continue
        r = rows[cid]; od = f"{args.out}/{cid}"; dd = f"{od}/duv"
        if os.path.exists(f"{od}/done.json"):
            continue
        os.makedirs(dd, exist_ok=True)
        ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"]
        ext = np.load(f"{DATA}/{m['ext_source']}/{cid}.npz")["ids"][:28]
        ids = np.concatenate([ids, ext], 0)
        meta = np.load(f"{DATA}/meta/{cid}.npz")
        run_dir = f"{RAW}/{r['subset']}/{r['scene']}/{r['run']}"
        t = tars.setdefault(run_dir, PV.RunTars(run_dir))
        need = sorted(set(idx))
        wb = {}
        for k in need:
            img = cv2.imdecode(np.frombuffer(t.get(int(r["start"]) + k, "whitebox.jpg"), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            wb[k] = GEO.resample_rgb(img)
        depth, scales = {}, []
        for s in range(0, len(need), 16):
            ks = need[s:s + 16]
            inp = proc(images=[Image.fromarray(wb[k]) for k in ks], return_tensors="pt").to(args.device)
            with torch.no_grad():
                pred = net(pixel_values=inp["pixel_values"].half()).predicted_depth.float()
            pred = torch.nn.functional.interpolate(pred[:, None], (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k, p in zip(ks, pred):
                g = ground_depth_m(meta["world_to_clip"][k], meta["world_to_view"][k], float(meta["ground_z"][k]))
                gm = np.isin(ids[k], (1, 2)) & np.isfinite(g) & (p > 0.05)
                sc = float(np.median(g[gm] / p[gm])) if gm.sum() > 500 else np.nan
                scales.append(sc); depth[k] = (p, sc)
        sc_clip = float(np.nanmedian(scales)) if np.isfinite(scales).any() else 1.0
        for i, k in enumerate(idx):
            p, sc = depth[k]
            d = p * sc_clip   # one scale per clip: per-frame scales jitter (cv ~0.2) and would flicker
            d[ids[k] == 0] = 0.0
            d = to_cond(d.astype(np.float32), cv2.INTER_AREA)
            d[~np.isfinite(d) | (d < 0)] = 0.0
            d.astype("<f4").tofile(f"{dd}/{i:06d}.depth.f32")
            Image.fromarray(to_cond(LUT[ids[k]], cv2.INTER_NEAREST), "L").save(f"{dd}/{i:06d}.semantic_id.png")
        sp = f"{args.testset}/solar/{cid}"
        Image.open(f"{sp}/first.png").convert("RGB").save(f"{od}/first_frame.png")
        prompt = open(f"{sp}/prompt.txt").read().strip()
        json.dump(dict(prompt=f"[0.00s-5.17s] {prompt}", frames=NF, fps=24), open(f"{od}/prompt.json", "w"), indent=1)
        fin = np.array([s for s in scales if np.isfinite(s)])
        rep = dict(clip_id=cid, scale_median=sc_clip, scale_cv=float(fin.std() / fin.mean()) if len(fin) > 1 else None,
                   frames_without_ground=int(np.sum(~np.isfinite(scales))))
        json.dump(rep, open(f"{od}/done.json", "w")); report.append(rep)
        if len(tars) > 4:
            for tt in tars.values():
                for fh in tt.fh.values(): fh.close()
            tars = {}
        print(j, cid, rep, flush=True)
    json.dump(dict(clips=report, depth_model=DA_REPO, frames=NF, fps=24, semantic_lut={int(i): int(LUT[i]) for i in np.nonzero(LUT)[0]}),
              open(f"{args.out}/manifest_{os.getpid()}.json", "w"), indent=1)
    print("DONE", len(report))


if __name__ == "__main__":
    main()
