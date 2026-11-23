#!/usr/bin/env python3
"""FX control, interface comparison: the natural-FX test of exp_fx_natural.py (deploy model) run on the three same-backbone
ablation teachers -- (1) colour video -> video VAE, (2) colour video -> light encoder, (3) integer ids -- which were all
trained on the same fxcond data, where the four FX classes are integers for (3) and palette colours for (1)/(2).
Same 94 held-out clips, same FX layouts (production40 val, frames 0..80), same seed 0 / first frame / prompts; each variant
generates the whole 81-frame clip twice with its own 50-step bidirectional sampler (base map vs map with FX), and the
scores are exactly those of exp_fx_natural.score (copied here; the deploy script imports the runtime code tree, this one
the ablation code tree).
Usage (from experiments/abl/code/gc25d): exp_fx_offline.py --ckpt CK --variant v3_ids --out DIR [--shard s --nshard n]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, math, os, sys, time, types
C = _mos.path.join(MALIANG_ROOT, "experiments")
RT = MALIANG_ROOT
sys.path[:0] = [C + "/abl/code/gc25d", C + "/metrics"]
import numpy as np, cv2, torch
import abl_sample as A
S, INF = A.S, A.INF
DATA = RT + "/data/gc25d_v10_fxcond"
P = _mos.path.join(_mos.environ.get("MALIANG_PROD40", _mos.path.join(MALIANG_ROOT, "data", "production40")), "root")
T, TH_ON, TH_PX = 25, 0.3, 300


def score(base, fx, mask):
    """identical to exp_fx_natural.score: base/fx uint8 [F,H,W,3], mask bool [81,H,W]."""
    union = cv2.dilate(mask[1:81].any(0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    cin, cout, pout, agree, on_curve = [], [], [], [], []
    for f in range(1, 81):
        diff = np.abs(fx[f].astype(np.int16) - base[f].astype(np.int16)).mean(2); m = mask[f]
        out = cv2.dilate(m.astype(np.uint8), np.ones((49, 49), np.uint8)) == 0
        if m.sum() >= 50:
            cin.append(float((diff[m] > T).mean()))
        cout.append(float((diff[out & ~union] > T).mean()) if (out & ~union).any() else 0.0)
        sq = ((fx[f].astype(np.float32) - base[f].astype(np.float32)) ** 2)[out].mean()
        pout.append(10 * math.log10(255 ** 2 / max(float(sq), 1e-8)))
        region = m if m.sum() >= TH_PX else union
        drawn = float((diff[region] > T).mean()) >= TH_ON
        agree.append(drawn == bool(m.sum() >= TH_PX)); on_curve.append(float((diff[union] > T).mean()))
    return dict(change_in=float(np.mean(cin)) if cin else None, change_out=float(np.mean(cout)), psnr_out=float(np.mean(pout)),
                timing_acc=float(np.mean(agree)), fx_frames=int(sum(mask[f].sum() >= TH_PX for f in range(1, 81))),
                fx_px_mean=float(mask[1:81].sum((1, 2)).mean()), union_change_curve=[round(v, 3) for v in on_curve])


def main():
    a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True); a.add_argument("--variant", required=True)
    a.add_argument("--out", required=True); a.add_argument("--shard", type=int, default=0); a.add_argument("--nshard", type=int, default=1)
    args = a.parse_args(); os.makedirs(args.out, exist_ok=True)
    g = types.SimpleNamespace(device="cuda", steps=50, shift=5.0, cfg=1.0, sem_cfg=1.0, seed=0)
    dit, info = S.load_dit(args.ckpt, False, "parts_fx", RT + "/models/Wan2.2-TI2V-5B", "cuda")
    vae = INF.load_vae("cuda", RT + "/models/Wan2.2-TI2V-5B")
    tt = S.TextTable(DATA + "/text_table_gc25d.pt")
    clips = json.load(open(C + "/exp4/fx_natural_clips.json"))[args.shard::args.nshard]
    import viz as VIZ
    res = []
    for j, cid in enumerate(clips):
        od = f"{args.out}/{cid}"
        if os.path.exists(f"{od}/result.json"):
            res.append(json.load(open(f"{od}/result.json"))); continue
        c = S.load_clip(DATA, cid); ids = c["ids"]
        fxm = np.load(f"{P}/cond/{cid}.npz")["ids"][: len(ids)]
        assert fxm.shape == ids.shape, (fxm.shape, ids.shape)
        mask = (fxm >= 22) & (fxm < 255)
        assert np.array_equal(fxm[~mask], ids[~mask]), f"{cid}: maps differ outside FX pixels"
        z0 = c["latent"][:, 0:1].unsqueeze(0).to("cuda"); gens = {}; t0 = time.time()
        for name, sem in (("base", ids), ("fx", fxm)):
            lat, _ = S.generate(g, dit, tt, z0, S.model_ids(sem, info["vocab"], "cond"), c["prompts"])
            gens[name] = INF.decode_video(vae, lat)
        r = dict(clip_id=cid, variant=args.variant, **score(gens["base"], gens["fx"], mask), sec=time.time() - t0)
        os.makedirs(od, exist_ok=True); json.dump(r, open(f"{od}/result.json", "w"))
        VIZ.write_video(f"{od}/base.mp4", gens["base"]); VIZ.write_video(f"{od}/fx.mp4", gens["fx"])
        np.savez_compressed(f"{od}/mask.npz", mask=mask[:81])
        res.append(r); print(j, cid, {k: r[k] for k in ("change_in", "change_out", "psnr_out", "timing_acc", "sec")}, flush=True)
    keys = ("change_in", "change_out", "psnr_out", "timing_acc")
    agg = {k: float(np.mean([r[k] for r in res if r[k] is not None])) for k in keys}; agg["n"] = len(res)
    json.dump(dict(variant=args.variant, ckpt=args.ckpt, shard=[args.shard, args.nshard], summary=agg),
              open(f"{args.out}/summary_{args.shard}of{args.nshard}.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
