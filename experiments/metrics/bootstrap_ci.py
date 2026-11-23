#!/usr/bin/env python3
"""95 % bootstrap confidence intervals for the main table (CPU only, no new generation).
Resamples the 100 test clips with replacement (B draws, the same draws for every method => paired differences). For each
draw every table metric is recomputed exactly as in eval_common.py: PSNR / SSIM / LPIPS = mean over clips; mIoU = pooled
over the summed per-clip confusion matrices; entity metrics = pooled per-frame counts, mean of player and boss; FVD = Frechet
distance of the drawn clips' I3D windows to the drawn clips' GT windows; temporal warping error from extra_eval.py.
--cluster run: resample whole recording runs (10 runs x 10 clips) instead of clips (stricter, clips of a run are correlated).
Reports point estimate, percentile 95 % CI, and Ours - X differences with CI and two-sided bootstrap p.
Usage: bootstrap_ci.py [--methods ours_ema500_tae,solar,cwm,agr] [--B 2000] [--cluster clip|run] [--tag abl]
(ablation: --methods v3_ids,v1_color_vae,v2_color_light,v4_none --tag abl => differences are v3_ids - X)"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy import linalg
EV = os.path.join(os.path.dirname(HERE), "main/eval")
NAMES = ["sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop", "player", "boss"]


def miou(conf):
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter; present = conf.sum(1) > 0
    return float(np.nanmean(np.where(union > 0, inter / np.maximum(union, 1), np.nan)[present]))


def fvd_stats(f):
    return f.mean(0), np.cov(f, rowvar=False)


def fvd(f1, f2):
    m1, s1 = fvd_stats(f1); m2, s2 = fvd_stats(f2)
    cs = linalg.sqrtm(s1 @ s2); cs = cs[0] if isinstance(cs, tuple) else cs
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def load(m, clips):
    per = {r["clip_id"]: r for r in json.load(open(f"{EV}/{m}/per_clip.json"))}
    ext = {r["clip_id"]: r for r in json.load(open(f"{EV}/{m}/extra_per_clip.json"))}
    z = np.load(f"{EV}/{m}/i3d_feats.npz"); order = [str(c) for c in z["clips"]]
    feats = z["feats"].reshape(len(order), -1, 400); fi = {c: i for i, c in enumerate(order)}
    d = dict(psnr=np.array([per[c]["psnr"] for c in clips]), ssim=np.array([per[c]["ssim"] for c in clips]),
             lpips=np.array([per[c]["lpips"] for c in clips]),
             conf=np.stack([np.array(ext[c]["conf"]).reshape(9, 9) for c in clips]),
             warp_gt=np.array([ext[c]["warp_gt"] for c in clips]), warp_self=np.array([ext[c]["warp_self"] for c in clips]),
             i3d=np.stack([feats[fi[c]] for c in clips]))
    for e in ("player", "boss"):
        for k in ("n", "tm", "iou_sum", "iou_n", "hal"):
            d[f"{e}_{k}"] = np.array([ext[c]["entity"][e][k] for c in clips], float)
    return d


def metrics(d, idx, gt_i3d, with_fvd):
    out = dict(psnr=d["psnr"][idx].mean(), ssim=d["ssim"][idx].mean(), lpips=d["lpips"][idx].mean(),
               miou=miou(d["conf"][idx].sum(0)), warp_gt=1e3 * d["warp_gt"][idx].mean(),
               warp_self=1e3 * d["warp_self"][idx].mean())
    tm, iou, hal = [], [], []
    for e in ("player", "boss"):
        s = {k: d[f"{e}_{k}"][idx].sum() for k in ("n", "tm", "iou_sum", "iou_n", "hal")}
        if s["n"]:
            tm.append(s["tm"] / s["n"]); hal.append(s["hal"] / s["n"])
        if s["iou_n"]:
            iou.append(s["iou_sum"] / s["iou_n"])
    out.update(time_match=np.mean(tm), pos_iou=np.mean(iou), hallucination=np.mean(hal))
    if with_fvd:
        out["fvd"] = fvd(d["i3d"][idx].reshape(-1, 400), gt_i3d[idx].reshape(-1, 400))
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument("--methods", default="ours_ema500_tae,solar,cwm,agr")
    a.add_argument("--B", type=int, default=2000); a.add_argument("--B_fvd", type=int, default=1000)
    a.add_argument("--cluster", choices=("clip", "run"), default="clip"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--tag", default="", help="output name suffix, e.g. abl (first method = reference)")
    args = a.parse_args()
    import eval_common as EC
    clips = json.load(open(EC.TEST)); ms = args.methods.split(",")
    gt_i3d = np.stack([np.load(f"{EV}/_gt/{c}.npz")["i3d"] for c in clips])
    data = {m: load(m, clips) for m in ms}
    gt_ext = {r["clip_id"]: r for r in json.load(open(f"{EV}/_gt/extra_per_clip.json"))} if os.path.exists(f"{EV}/_gt/extra_per_clip.json") else None
    runs = [c.rsplit("__", 1)[0] for c in clips]; groups = {r: [i for i, x in enumerate(runs) if x == r] for r in dict.fromkeys(runs)}
    rng = np.random.default_rng(args.seed); n = len(clips); full = np.arange(n)
    draws = []
    for b in range(args.B):
        if args.cluster == "clip":
            draws.append(rng.integers(0, n, n))
        else:
            g = list(groups.values()); pick = rng.integers(0, len(g), len(g)); draws.append(np.concatenate([g[i] for i in pick]))
    point = {m: metrics(data[m], full, gt_i3d, True) for m in ms}
    boot = {m: [metrics(data[m], idx, gt_i3d, b < args.B_fvd) for b, idx in enumerate(draws)] for m in ms}
    keys = ["psnr", "ssim", "lpips", "fvd", "miou", "time_match", "pos_iou", "hallucination", "warp_gt", "warp_self"]
    res = dict(cluster=args.cluster, B=args.B, B_fvd=args.B_fvd, n_clips=n, methods={}, diff_vs_ours={})
    ref = ms[0]
    for m in ms:
        res["methods"][m] = {}
        for k in keys:
            v = np.array([r[k] for r in boot[m] if k in r])
            res["methods"][m][k] = dict(point=float(point[m][k]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        if m == ref:
            continue
        res.setdefault("reference", ref); res["diff_vs_ours"][m] = {}
        for k in keys:
            dv = np.array([ro[k] - rm[k] for ro, rm in zip(boot[ref], boot[m]) if k in ro])
            p = float(min(1.0, 2 * min((dv <= 0).mean(), (dv >= 0).mean())))
            res["diff_vs_ours"][m][k] = dict(point=float(point[ref][k] - point[m][k]), lo=float(np.percentile(dv, 2.5)),
                                             hi=float(np.percentile(dv, 97.5)), p=p)
    if gt_ext:
        res["gt_warp"] = dict(warp_gt=1e3 * float(np.mean([gt_ext[c]["warp_gt"] for c in clips])))
    out = f"{EV}/bootstrap_ci_{args.tag + '_' if args.tag else ''}{args.cluster}.json"; json.dump(res, open(out, "w"), indent=1)
    for m in ms:
        print(m.ljust(16), "  ".join(f"{k} {res['methods'][m][k]['point']:.3f} [{res['methods'][m][k]['lo']:.3f},{res['methods'][m][k]['hi']:.3f}]" for k in keys))
    for m in ms[1:]:
        print(ref + "-" + m, "  ".join(f"{k} {res['diff_vs_ours'][m][k]['point']:+.3f} [{res['diff_vs_ours'][m][k]['lo']:+.3f},{res['diff_vs_ours'][m][k]['hi']:+.3f}] p={res['diff_vs_ours'][m][k]['p']:.3f}" for k in keys))
    print("->", out)


if __name__ == "__main__":
    main()
