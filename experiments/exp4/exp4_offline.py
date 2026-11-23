#!/usr/bin/env python3
"""Experiment 4, interface comparison (variants 1/2/3): the same 50 world-anchored edits as exp4_edit.py (deploy model),
but for the bidirectional SFT variants, which cannot edit while generating. Each variant regenerates the whole 81-frame
clip from the edited map (edit switched on at frame E0 = 37, identical masks / edited ids from exp4_edit's edit.npz),
same seed / first frame / prompts as its unedited test-set generation (main/abl/<variant>/<clip>_gen.mp4 = base run).
Scores (frames E0+4 .. 80, segmenter seg_b2): edit_acc (edited pixels segmented as the new class), baseline_acc (same in
the base run), keep_psnr / keep_lpips outside the edit mask dilated by 24 px. No latency (offline regeneration).
Usage (from experiments/abl/code/gc25d): exp4_offline.py --ckpt CK --variant v3_ids --out DIR"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, glob, json, math, os, sys, time, types
HERE = os.path.dirname(os.path.abspath(__file__))
C = _mos.path.join(MALIANG_ROOT, "experiments")
RT = MALIANG_ROOT
sys.path[:0] = [C + "/abl/code/gc25d", C + "/metrics"]
import numpy as np, cv2, torch
import abl_sample as A            # patches sample.py for colour-interface checkpoints (ids checkpoints fall through)
S, INF = A.S, A.INF
from eval_common import Scorer, LUT, read_video
E0, NF = 37, 81
DATA = RT + "/data/gc25d_v10_fxcond"


def main():
    a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True); a.add_argument("--variant", required=True)
    a.add_argument("--out", required=True); a.add_argument("--edits", default=C + "/exp4/ours_ema500_tae")
    args = a.parse_args(); os.makedirs(args.out, exist_ok=True)
    g = types.SimpleNamespace(device="cuda", steps=50, shift=5.0, cfg=1.0, sem_cfg=1.0, seed=0)
    dit, info = S.load_dit(args.ckpt, False, "parts_fx", RT + "/models/Wan2.2-TI2V-5B", "cuda")
    vae = INF.load_vae("cuda", RT + "/models/Wan2.2-TI2V-5B")
    tt = S.TextTable(DATA + "/text_table_gc25d.pt"); sc = Scorer(MALIANG_ROOT + "/assets/seg_b2_final")
    res = []
    for d in sorted(glob.glob(f"{args.edits}/*/edit.npz")):
        cid = os.path.basename(os.path.dirname(d)); meta = json.load(open(os.path.join(os.path.dirname(d), "result.json")))
        od = f"{args.out}/{cid}"
        if os.path.exists(f"{od}/result.json"):
            res.append(json.load(open(f"{od}/result.json"))); continue
        base_p = f"{C}/main/abl/{args.variant}/{cid}_gen.mp4"
        if not os.path.exists(base_p):
            print("no base run yet", cid); continue
        e = np.load(d); mask = e["mask"][: NF - E0]; tgt_ids = e["edited_ids"][: NF - E0]
        c = S.load_clip(DATA, cid); ids = c["ids"].copy(); ids[E0:] = tgt_ids
        tgt = {"add": meta.get("cls"), "ground": meta.get("new"), "remove": meta.get("filled_with")}[meta["kind"]]
        t0 = time.time()
        lat, _ = S.generate(g, dit, tt, c["latent"][:, 0:1].unsqueeze(0).to("cuda"), S.model_ids(ids, info["vocab"], "cond"), c["prompts"])
        ed = INF.decode_video(vae, lat); base = read_video(base_p)[:NF]
        se = sc.segment(ed[E0:]); sb = sc.segment(base[E0:]); tl = LUT[tgt]
        acc = [float((se[i][mask[i]] == tl).mean()) if mask[i].sum() >= 50 else np.nan for i in range(NF - E0)]
        bac = [float((sb[i][mask[i]] == tl).mean()) if mask[i].sum() >= 50 else np.nan for i in range(NF - E0)]
        keep, kl = [], []
        for i in range(NF - E0):
            km = cv2.dilate(mask[i].astype(np.uint8), np.ones((49, 49), np.uint8)) == 0
            dd = (ed[E0 + i].astype(np.float32) - base[E0 + i].astype(np.float32))[km]
            keep.append(10 * math.log10(255 ** 2 / max(float((dd ** 2).mean()), 1e-8)))
            if i % 4 == 0:
                kl.append(float(sc.lpips(ed[E0 + i:E0 + i + 1] * km[..., None], base[E0 + i:E0 + i + 1] * km[..., None])[0]))
        fin = [v for v in acc[4:] if np.isfinite(v)]; bfin = [v for v in bac[4:] if np.isfinite(v)]
        r = dict(clip_id=cid, kind=meta["kind"], target=int(tgt), variant=args.variant, edit_acc=float(np.mean(fin)) if fin else None,
                 baseline_acc=float(np.mean(bfin)) if bfin else None, keep_psnr=float(np.mean(keep)), keep_lpips=float(np.mean(kl)),
                 sec=time.time() - t0)
        os.makedirs(od, exist_ok=True); json.dump(r, open(f"{od}/result.json", "w"), indent=1)
        import viz as VIZ
        VIZ.write_video(f"{od}/edited.mp4", ed)
        res.append(r); print(cid, {k: r[k] for k in ("kind", "edit_acc", "baseline_acc", "keep_psnr")}, flush=True)
    agg = {}
    for kind in ("add", "ground", "remove", "all"):
        rr = [r for r in res if kind == "all" or r["kind"] == kind]
        if rr:
            agg[kind] = {k: float(np.nanmean([r[k] if r[k] is not None else np.nan for r in rr]))
                         for k in ("edit_acc", "baseline_acc", "keep_psnr", "keep_lpips")} | {"n": len(rr)}
    json.dump(dict(variant=args.variant, ckpt=args.ckpt, summary=agg), open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
