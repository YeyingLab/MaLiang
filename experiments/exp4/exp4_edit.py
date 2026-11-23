#!/usr/bin/env python3
"""Experiment 4 (edit while playing) with the deploy model (fxcond_v1 stage3 ema_step500, causal 4-step rollout).
For each test clip: two rollouts with the same seed / first frame / prompts, 27 cells (105 frames):
  base   = recorded numeric map;   edited = the same map with one world-anchored edit switched on at cell EDIT_CELL.
Edits (type = clip index % 3), all defined in world coordinates and re-projected every frame with the recorded camera:
  add      vertical cylinder (r 120 cm, h 260 cm) of rock / vegetation / architecture standing on the ground plane, placed on a
           ground pixel of frame E0 far from both characters; characters stay on top.
  ground   disk (r 350 cm) on the ground plane: terrain <-> paving swap of the ground classes inside it.
  remove   one static category (largest of prop / vegetation / rock / architecture covering 1..35 % of frame E0) -> the
           frame's dominant ground class, from E0 on.
Scores (frames >= E0, frame-level, segmenter = metrics/seg_b2/final):
  edit_acc      fraction of edited pixels the segmenter labels with the new class (edited run), per frame -> mean over the
                last 8 cells; baseline_acc = same pixels in the base run (how often the new class appears without the edit).
  latency       first frame >= E0 whose edit_acc reaches 50 % of the final level; reported in frames and in ms
                (cells x measured DiT ms/cell).
  keep_lpips / keep_psnr   edited vs base run outside the edit mask dilated by 24 px (unedited region preserved?).
Usage: exp4_edit.py --ckpt CK --out DIR [--n 50] [--vae tae]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, math, os, sys, time
RT = MALIANG_ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
MET_DIR = _mos.path.join(MALIANG_ROOT, "experiments", "metrics")
sys.path[:0] = [RT + "/code/gc25d", RT + "/code/gc25d/gcdistill", RT + "/code/arope", MET_DIR, MET_DIR + "/pylib"]
import numpy as np, cv2, torch
import eval_rollout as ER
import geometry as GEO
from eval_common import Scorer, LUT
DG = ER.DG
DATA = RT + "/data/gc25d_v10_fxcond"
TEST = _mos.path.join(MALIANG_ROOT, "experiments", "testset", "v1", "test_clips.json")
CELLS, EDIT_CELL = 27, 10
E0 = 4 * EDIT_CELL - 3                     # first frame of cell 10 (cell k>0 covers frames 4k-3 .. 4k)
W, H = GEO.DST_W, GEO.DST_H
CHAR = list(range(10, 26))


def rays(M, V):
    ys, xs = np.mgrid[0:H, 0:W]
    sx = GEO._IDX_X[xs.ravel()].astype(np.float64); sy = GEO._IDX_Y[ys.ravel()].astype(np.float64)
    M = np.asarray(M, np.float64).reshape(4, 4); V = np.asarray(V, np.float64).reshape(4, 4); Mi = np.linalg.inv(M)
    nx = ((sx + 0.5) / GEO.SRC_W - 0.5) * 2; ny = (0.5 - (sy + 0.5) / GEO.SRC_H) * 2

    def unproj(z):
        q = np.stack([nx, ny, np.full_like(nx, z), np.ones_like(nx)], 1) @ Mi
        return q[:, :3] / q[:, 3:4]
    p0, p1 = unproj(0.9), unproj(0.1); rd = p1 - p0; rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    fwd = V[:3, 2] / np.linalg.norm(V[:3, 2])
    if (rd @ fwd).mean() < 0:
        rd = -rd
    return p0 - rd * 5000.0, rd


def ground_hit(ro, rd, z0):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z0 - ro[:, 2]) / rd[:, 2]
    t[~(t > 0)] = np.inf
    return t, ro + np.where(np.isfinite(t), t, 0)[:, None] * rd


def cylinder_t(ro, rd, c, r, z0, h):
    ox, oy = ro[:, 0] - c[0], ro[:, 1] - c[1]
    a = rd[:, 0] ** 2 + rd[:, 1] ** 2; b = 2 * (ox * rd[:, 0] + oy * rd[:, 1]); cc = ox ** 2 + oy ** 2 - r * r
    disc = b * b - 4 * a * cc
    t = np.full(len(ro), np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    for tt in ((-b - sq) / (2 * np.maximum(a, 1e-12)), (-b + sq) / (2 * np.maximum(a, 1e-12))):
        z = ro[:, 2] + tt * rd[:, 2]
        hit = ok & (tt > 0) & (z >= z0) & (z <= z0 + h)
        t = np.where(hit & (tt < t), tt, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (z0 + h - ro[:, 2]) / rd[:, 2]
    p = ro + np.where(np.isfinite(tc), tc, 0)[:, None] * rd
    cap = (tc > 0) & ((p[:, 0] - c[0]) ** 2 + (p[:, 1] - c[1]) ** 2 <= r * r)
    return np.where(cap & (tc < t), tc, t)


def make_edit(ids, meta, kind, j):
    """ids uint8 [105,H,W] raw parts_fx -> (edited ids, mask [105,H,W] bool, target class, info)."""
    ed = ids.copy(); z0 = float(np.median(meta["ground_z"][:len(ids)]))
    f0 = ids[E0]
    if kind == "remove":
        cnt = {c: int((f0 == c).sum()) for c in (6, 5, 4, 3)}
        ok = [c for c, n in cnt.items() if 0.01 * f0.size <= n <= 0.35 * f0.size]
        if not ok:
            return None
        cat = max(ok, key=lambda c: cnt[c])
        g = 1 if (f0 == 1).sum() >= (f0 == 2).sum() else 2
        m = np.zeros(ids.shape, bool); m[E0:] = ids[E0:] == cat
        ed[m] = g
        return ed, m, g, dict(kind=kind, removed=int(cat), filled_with=int(g))
    ro, rd = rays(meta["world_to_clip"][E0], meta["world_to_view"][E0])
    tg, P = ground_hit(ro, rd, z0)
    ground = np.isin(f0, (1, 2)).ravel() & np.isfinite(tg)
    chars = np.isin(f0, CHAR)
    dist = cv2.distanceTransform((~chars).astype(np.uint8), cv2.DIST_L2, 5).ravel()
    ys, xs = np.mgrid[0:H, 0:W]
    central = ((np.abs(xs - W / 2) < 0.32 * W) & (np.abs(ys - H / 2) < 0.32 * H)).ravel()
    score = np.where(ground & central, dist, -1)
    if score.max() < 60:
        return None
    k = int(np.argmax(score)); c = P[k, :2]
    m = np.zeros(ids.shape, bool)
    if kind == "add":
        cls = (4, 5, 3)[j % 3]
        for f in range(E0, len(ids)):
            ro, rd = rays(meta["world_to_clip"][f], meta["world_to_view"][f])
            tg, _ = ground_hit(ro, rd, z0)
            to = cylinder_t(ro, rd, c, 120.0, z0, 260.0).reshape(H, W)
            mm = (to < tg.reshape(H, W)) & ~np.isin(ids[f], CHAR)
            m[f] = mm; ed[f][mm] = cls
        return ed, m, cls, dict(kind=kind, cls=int(cls), center_cm=c.tolist(), r_cm=120, h_cm=260)
    g_at = int(f0.ravel()[k]); new = 2 if g_at == 1 else 1
    for f in range(E0, len(ids)):
        ro, rd = rays(meta["world_to_clip"][f], meta["world_to_view"][f])
        tg, P = ground_hit(ro, rd, z0)
        inside = (np.isfinite(tg) & (((P[:, 0] - c[0]) ** 2 + (P[:, 1] - c[1]) ** 2) <= 350.0 ** 2)).reshape(H, W)
        mm = inside & np.isin(ids[f], (1, 2)) & (ids[f] != new)
        m[f] = mm; ed[f][mm] = new
    return ed, m, new, dict(kind=kind, new=int(new), center_cm=c.tolist(), r_cm=350)


def main():
    a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True); a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=50); a.add_argument("--vae", default="tae"); a.add_argument("--seed", type=int, default=0)
    args = a.parse_args(); os.makedirs(args.out, exist_ok=True); dev = torch.device("cuda")
    model = ER.load_student(args.ckpt, dev); vocab = DG.GM.sem_vocab(model.dit)
    ds = DG.GcRolloutDataset(DATA, "val", DATA + "/text_table_gc25d.pt", vocab=vocab, n_cells=CELLS, verbose=False)
    dec = ER.Decoder(args.vae, dev); sc = Scorer(MALIANG_ROOT + "/assets/seg_b2_final")
    clips = json.load(open(TEST))
    clips = [c for i, c in enumerate(clips) if i % 2 == 0][: args.n]          # 50 = every other clip (25 real / 25 stylized)
    res = []
    for j, cid in enumerate(clips):
        od = f"{args.out}/{cid}"
        if os.path.exists(f"{od}/result.json"):
            res.append(json.load(open(f"{od}/result.json"))); continue
        s = ds.load(cid); meta = np.load(f"{DATA}/meta/{cid}.npz")
        ids = s["sem_ids"].numpy() if torch.is_tensor(s["sem_ids"]) else np.asarray(s["sem_ids"])
        kind = ("add", "ground", "remove")[j % 3]
        e = None
        for kk in (kind, "add", "ground", "remove"):
            e = make_edit(ids, meta, kk, j)
            if e is not None:
                break
        if e is None:
            print("skip", cid); continue
        ed, mask, tgt, info = e
        gens, ms = {}, []
        for name, sem in (("base", ids), ("edited", ed)):
            b = DG.rollout_collate([{"z0": s["z0"], "sem_ids": torch.from_numpy(np.ascontiguousarray(sem)),
                                     "context": s["context"], "context_ids": s["context_ids"]}])
            timing = []
            lat = ER.deploy_rollout(model, b["z0"].to(dev), b["context"].to(dev), b["context_ids"].to(dev), b["sem_ids"].to(dev),
                                    CELLS - 1, seed=args.seed, timing=timing)
            gens[name] = dec(lat); ms.append(float(np.mean(timing[2:])))
        F = len(gens["base"]); seg_e = sc.segment(gens["edited"][E0:]); seg_b = sc.segment(gens["base"][E0:])
        tl = LUT[tgt]
        acc = [float((seg_e[i][mask[E0 + i]] == tl).mean()) if mask[E0 + i].sum() >= 50 else np.nan for i in range(F - E0)]
        bacc = [float((seg_b[i][mask[E0 + i]] == tl).mean()) if mask[E0 + i].sum() >= 50 else np.nan for i in range(F - E0)]
        final = float(np.nanmean(acc[-32:])) if np.isfinite(acc[-32:]).any() else float("nan")
        lat_f = next((i for i, v in enumerate(acc) if np.isfinite(v) and v >= 0.5 * final), None) if final > 0 else None
        keep = []
        for i in range(E0, F):
            km = cv2.dilate(mask[i].astype(np.uint8), np.ones((49, 49), np.uint8)) == 0
            d = np.abs(gens["edited"][i].astype(np.float32) - gens["base"][i].astype(np.float32))[km]
            keep.append(10 * math.log10(255 ** 2 / max(float((d ** 2).mean()), 1e-8)) if km.any() else np.nan)
        kl = []
        for i in range(E0, F, 4):
            km = (cv2.dilate(mask[i].astype(np.uint8), np.ones((49, 49), np.uint8)) == 0)[..., None]
            kl.append(float(sc.lpips(gens["edited"][i:i + 1] * km, gens["base"][i:i + 1] * km)[0]))
        r = dict(clip_id=cid, **info, edit_px_mean=float(mask[E0:].sum((1, 2)).mean()), edit_acc=final,
                 baseline_acc=float(np.nanmean(bacc[-32:])), latency_frames=lat_f,
                 latency_ms=(None if lat_f is None else math.ceil((lat_f + 1) / 4) * float(np.mean(ms))),
                 keep_psnr=float(np.nanmean(keep)), keep_lpips=float(np.mean(kl)), dit_ms_per_cell=float(np.mean(ms)),
                 acc_curve=[None if not np.isfinite(v) else round(v, 4) for v in acc])
        os.makedirs(od, exist_ok=True); json.dump(r, open(f"{od}/result.json", "w"), indent=1)
        np.savez_compressed(f"{od}/edit.npz", mask=mask[E0:], edited_ids=ed[E0:])
        import viz as VIZ
        VIZ.write_video(f"{od}/base.mp4", gens["base"]); VIZ.write_video(f"{od}/edited.mp4", gens["edited"])
        res.append(r); print(j, cid, {k: r[k] for k in ("kind", "edit_acc", "baseline_acc", "latency_frames", "keep_psnr", "keep_lpips")}, flush=True)
    agg = {}
    for kind in ("add", "ground", "remove", "all"):
        rr = [r for r in res if kind == "all" or r["kind"] == kind]
        if rr:
            agg[kind] = dict(n=len(rr), edit_acc=float(np.nanmean([r["edit_acc"] for r in rr])),
                             baseline_acc=float(np.nanmean([r["baseline_acc"] for r in rr])),
                             latency_frames=float(np.mean([r["latency_frames"] for r in rr if r["latency_frames"] is not None] or [np.nan])),
                             latency_ms=float(np.mean([r["latency_ms"] for r in rr if r["latency_ms"] is not None] or [np.nan])),
                             keep_psnr=float(np.nanmean([r["keep_psnr"] for r in rr])), keep_lpips=float(np.mean([r["keep_lpips"] for r in rr])))
    json.dump(dict(ckpt=args.ckpt, edit_cell=EDIT_CELL, e0=E0, cells=CELLS, vae=args.vae, summary=agg), open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
