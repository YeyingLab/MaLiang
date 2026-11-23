#!/usr/bin/env python3
"""FX control test on natural boss-attack FX (deploy model). The paper test clips carry no FX, and no held-out clip of the
fxcond_v1 data does either, so the FX layouts come from the boss-FX production data (bossfx_revision_20261108/production40):
the same val-split clips (never trained on) with boss attack FX ids (raw 22..25) composited into the id map. Outside the FX
pixels the two id maps are identical (checked: 0 differing pixels over the 94 clips). FX appearance in that data follows a
later FX design than the one fxcond_v1 was trained on, so GT RGB is NOT used; the test measures control only:
two rollouts per clip with the same seed / first frame / prompts -- base (map without FX) vs fx (map with FX, frames 0..80;
the 24 extension frames stay as recorded) -- and per frame 1..80:
  change_in   share of FX pixels whose colour changed (mean |fx - base| over RGB > 25), frames with >= 50 FX px
  change_out  same share outside the FX mask dilated 49x49 (side effects)       psnr_out  PSNR(fx, base) there
  timing      frame-level agreement between "FX in the map" (>= 300 px) and "FX drawn" (change_in >= 0.3 on the union of
              the clip's FX masks dilated 9x9, so frames without FX are checked for lingering / early FX too)
Usage: exp_fx_natural.py --ckpt CK --out DIR [--n 94]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, cv2, torch
import exp4_edit as X
ER, DG, DATA, CELLS = X.ER, X.DG, X.DATA, X.CELLS
P = _mos.path.join(_mos.environ.get("MALIANG_PROD40", _mos.path.join(MALIANG_ROOT, "data", "production40")), "root")
T, TH_ON, TH_PX = 25, 0.3, 300


def score(base, fx, mask):
    """base/fx uint8 [F,H,W,3] (frames 0..), mask bool [81,H,W]."""
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
    a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True); a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=94); a.add_argument("--vae", default="tae"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--shard", type=int, default=0); a.add_argument("--nshard", type=int, default=1)
    args = a.parse_args(); os.makedirs(args.out, exist_ok=True); dev = torch.device("cuda")
    model = ER.load_student(args.ckpt, dev); vocab = DG.GM.sem_vocab(model.dit)
    ds = DG.GcRolloutDataset(DATA, "val", DATA + "/text_table_gc25d.pt", vocab=vocab, n_cells=CELLS, verbose=False)
    dec = ER.Decoder(args.vae, dev)
    clips = json.load(open(HERE + "/fx_natural_clips.json"))[: args.n][args.shard::args.nshard]
    import viz as VIZ
    res = []
    for j, cid in enumerate(clips):
        od = f"{args.out}/{cid}"
        if os.path.exists(f"{od}/result.json"):
            res.append(json.load(open(f"{od}/result.json"))); continue
        s = ds.load(cid)
        ids = s["sem_ids"].numpy() if torch.is_tensor(s["sem_ids"]) else np.asarray(s["sem_ids"])
        fxm = np.load(f"{P}/cond/{cid}.npz")["ids"]; ed = ids.copy(); ed[: len(fxm)] = fxm
        mask = (fxm >= 22) & (fxm < 255)
        gens = {}
        for name, sem in (("base", ids), ("fx", ed)):
            b = DG.rollout_collate([{"z0": s["z0"], "sem_ids": torch.from_numpy(np.ascontiguousarray(sem)),
                                     "context": s["context"], "context_ids": s["context_ids"]}])
            lat = ER.deploy_rollout(model, b["z0"].to(dev), b["context"].to(dev), b["context_ids"].to(dev), b["sem_ids"].to(dev),
                                    CELLS - 1, seed=args.seed, timing=[])
            gens[name] = dec(lat)
        r = dict(clip_id=cid, **score(gens["base"], gens["fx"], mask))
        os.makedirs(od, exist_ok=True); json.dump(r, open(f"{od}/result.json", "w"))
        VIZ.write_video(f"{od}/base.mp4", gens["base"]); VIZ.write_video(f"{od}/fx.mp4", gens["fx"])
        np.savez_compressed(f"{od}/mask.npz", mask=mask[:81])
        res.append(r); print(j, cid, {k: r[k] for k in ("change_in", "change_out", "psnr_out", "timing_acc", "fx_frames")}, flush=True)
    keys = ("change_in", "change_out", "psnr_out", "timing_acc")
    agg = {k: float(np.mean([r[k] for r in res if r[k] is not None])) for k in keys}; agg["n"] = len(res)
    json.dump(dict(ckpt=args.ckpt, shard=[args.shard, args.nshard], summary=agg, threshold=T, on=TH_ON, min_px=TH_PX), open(f"{args.out}/summary_{args.shard}of{args.nshard}.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
