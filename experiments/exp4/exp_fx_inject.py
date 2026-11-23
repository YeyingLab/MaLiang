#!/usr/bin/env python3
"""New-category test (experiment 6, part 2): the boss-attack FX classes (raw 22..25) were added to the vocabulary by one
number each. The paper test clips (val split) contain no FX at all, so FX is *injected*:
  donor   = an FX burst recorded in a train clip (ids 22..25 over consecutive frames, >= 300 px), kept with its offset to the
            donor boss's screen centroid;
  target  = a test clip, from frame E0 (cell 10) on: donor FX frames are pasted at  target boss centroid + donor offset
            (frame by frame, looping the burst), on top of every class (FX draws over everything in the recorded ids).
Two rollouts per clip with the deploy model (same seed): base (recorded map) vs injected. Scores with the 10-class
segmenter (metrics/seg_b2_fx/final, class 9 = fx): fx_recall = share of injected FX pixels segmented as fx (injected run),
baseline = same pixels in the base run, precision proxy = fx pixels predicted outside the dilated injected mask (both runs),
keep_psnr outside the mask, latency as in exp4.
Usage: exp_fx_inject.py --ckpt CK --out DIR [--n 30]"""
import argparse, json, math, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, cv2, torch
import exp4_edit as X
from eval_common import Scorer
ER, DG, DATA, E0, CELLS = X.ER, X.DG, X.DATA, X.E0, X.CELLS
LUT10 = X.LUT.copy(); LUT10[[22, 23, 24, 25]] = 9
BOSS = [11, 17, 18, 19, 20, 21]


def donors(n, seed=0):
    rows = [json.loads(l) for l in open(DATA + "/index/clips.jsonl")]
    tr = [r for r in rows if r["split"] == "train" and not r["drop"]]
    random.Random(seed).shuffle(tr)
    out = []
    for r in tr:
        ids = np.load(f"{DATA}/cond/{r['clip_id']}.npz")["ids"]
        fx = ids >= 22; n_fx = fx.sum((1, 2))
        ks = np.nonzero(n_fx >= 300)[0]
        if len(ks) < 6:
            continue
        seq = []
        for k in range(ks[0], ks[-1] + 1):
            b = np.isin(ids[k], BOSS)
            if not b.any() or n_fx[k] < 50:
                continue
            by, bx = np.argwhere(b).mean(0)
            ys, xs = np.nonzero(fx[k])
            seq.append(dict(dy=ys - by, dx=xs - bx, v=ids[k][ys, xs]))
        if len(seq) >= 6:
            out.append(dict(clip_id=r["clip_id"], frames=seq))
        if len(out) >= n:
            break
    return out


def inject(ids, donor):
    ed = ids.copy(); mask = np.zeros(ids.shape, bool); H, W = ids.shape[1:]
    last = None
    for i, f in enumerate(range(E0, len(ids))):
        b = np.isin(ids[f], BOSS)
        if b.any():
            last = np.argwhere(b).mean(0)
        if last is None:
            continue
        d = donor["frames"][i % len(donor["frames"])]
        y = np.round(last[0] + d["dy"]).astype(int); x = np.round(last[1] + d["dx"]).astype(int)
        ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        ed[f][y[ok], x[ok]] = d["v"][ok]; mask[f][y[ok], x[ok]] = True
    return ed, mask


def main():
    a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True); a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=30); a.add_argument("--vae", default="tae"); a.add_argument("--seed", type=int, default=0)
    args = a.parse_args(); os.makedirs(args.out, exist_ok=True); dev = torch.device("cuda")
    model = ER.load_student(args.ckpt, dev); vocab = DG.GM.sem_vocab(model.dit)
    ds = DG.GcRolloutDataset(DATA, "val", DATA + "/text_table_gc25d.pt", vocab=vocab, n_cells=CELLS, verbose=False)
    dec = ER.Decoder(args.vae, dev); sc = Scorer(X.MET_DIR + "/seg_b2_fx/final")
    dn = donors(3 * args.n)
    rows = {json.loads(l)["clip_id"]: json.loads(l) for l in open(DATA + "/index/clips.jsonl")}
    clips = [c for c in json.load(open(X.TEST)) if rows[c].get("boss_visible_frac", 0) >= 0.5]
    res = []
    for j, cid in enumerate(clips[: args.n]):
        od = f"{args.out}/{cid}"
        if os.path.exists(f"{od}/result.json"):
            res.append(json.load(open(f"{od}/result.json"))); continue
        s = ds.load(cid)
        ids = s["sem_ids"].numpy() if torch.is_tensor(s["sem_ids"]) else np.asarray(s["sem_ids"])
        donor = dn[j % len(dn)]
        ed, mask = inject(ids, donor)
        if mask[E0:].sum() < 2000:
            print("skip (boss not visible after E0)", cid); continue
        gens, ms = {}, []
        for name, sem in (("base", ids), ("fx", ed)):
            b = DG.rollout_collate([{"z0": s["z0"], "sem_ids": torch.from_numpy(np.ascontiguousarray(sem)),
                                     "context": s["context"], "context_ids": s["context_ids"]}])
            timing = []
            lat = ER.deploy_rollout(model, b["z0"].to(dev), b["context"].to(dev), b["context_ids"].to(dev), b["sem_ids"].to(dev),
                                    CELLS - 1, seed=args.seed, timing=timing)
            gens[name] = dec(lat); ms.append(float(np.mean(timing[2:])))
        F = len(gens["base"]); sf = sc.segment(gens["fx"][E0:]); sb = sc.segment(gens["base"][E0:])
        rec, brec, fp_fx, fp_b = [], [], [], []
        for i in range(F - E0):
            m = mask[E0 + i]
            out = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
            if m.sum() >= 50:
                rec.append(float((sf[i][m] == 9).mean())); brec.append(float((sb[i][m] == 9).mean()))
            fp_fx.append(float((sf[i][out] == 9).mean())); fp_b.append(float((sb[i][out] == 9).mean()))
        lat_f = next((i for i, v in enumerate(rec) if v >= 0.5 * np.mean(rec)), None) if rec and np.mean(rec) > 0 else None
        keep = []
        for i in range(E0, F):
            km = cv2.dilate(mask[i].astype(np.uint8), np.ones((49, 49), np.uint8)) == 0
            d = (gens["fx"][i].astype(np.float32) - gens["base"][i].astype(np.float32))[km]
            keep.append(10 * math.log10(255 ** 2 / max(float((d ** 2).mean()), 1e-8)))
        r = dict(clip_id=cid, donor=donor["clip_id"], fx_px_mean=float(mask[E0:].sum((1, 2)).mean()),
                 fx_recall=float(np.mean(rec)) if rec else None, base_recall=float(np.mean(brec)) if brec else None,
                 fx_outside_injected=float(np.mean(fp_fx)), fx_outside_base=float(np.mean(fp_b)),
                 latency_frames=lat_f, keep_psnr=float(np.mean(keep)), dit_ms_per_cell=float(np.mean(ms)),
                 recall_curve=[round(v, 4) for v in rec])
        os.makedirs(od, exist_ok=True); json.dump(r, open(f"{od}/result.json", "w"), indent=1)
        import viz as VIZ
        VIZ.write_video(f"{od}/base.mp4", gens["base"]); VIZ.write_video(f"{od}/fx.mp4", gens["fx"])
        np.savez_compressed(f"{od}/inject.npz", mask=mask[E0:], ids=ed[E0:])
        res.append(r); print(j, cid, {k: r[k] for k in ("fx_recall", "base_recall", "fx_outside_injected", "keep_psnr")}, flush=True)
    keys = ("fx_recall", "base_recall", "fx_outside_injected", "fx_outside_base", "keep_psnr")
    agg = {k: float(np.mean([r[k] for r in res if r[k] is not None])) for k in keys}
    agg["n"] = len(res)
    json.dump(dict(ckpt=args.ckpt, e0=E0, summary=agg), open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
