#!/usr/bin/env python3
"""Experiment 6 with the deploy model (fxcond_v1 stage3 ema_step500, causal 4-step rollout, TAEHV decode - the same path as
main/ours_ema500_tae). Same inputs as variant (3): <case>/first.png (FLUX first frame, VAE-encoded with fp32 single-frame
encode like sample_user.py), <case>/ids.npz (81 frames), <case>/prompts.json (21 cell prompts), seed 0.
Writes <case>/gen_ours/<case>_gen.mp4 (81 frames, 16 fps) and timing.json.
Usage: exp6_deploy.py --cases DIR [DIR ...] [--ckpt CK]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, glob, json, os, sys, time
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d", RT + "/code/gc25d/gcdistill", RT + "/code/arope"]
import numpy as np, torch
import eval_rollout as ER
DG = ER.DG
import infer as INF
import viz as VIZ
DATA = RT + "/data/gc25d_v10_fxcond"
CK = os.environ.get(
    "MALIANG_STUDENT_CKPT",
    os.path.join(RT, "outputs", "distill", "gc25d_v10", "stage3_dmd", "ema", "ema_step500.safetensors"),
)


def main():
    a = argparse.ArgumentParser(); a.add_argument("--cases", nargs="+", required=True); a.add_argument("--ckpt", default=CK)
    a.add_argument("--seed", type=int, default=0); args = a.parse_args(); dev = torch.device("cuda")
    model = ER.load_student(args.ckpt, dev); vocab = DG.GM.sem_vocab(model.dit)
    table = DG.load_text_table(DATA + "/text_table_gc25d.pt"); kid = DG.key_to_id(table)
    vae = INF.load_vae("cuda"); dec = ER.Decoder("tae", dev); n = 0
    for root in args.cases:
        for d in sorted(glob.glob(f"{root}/*/")):
            case = os.path.basename(d.rstrip("/")); od = f"{d}/gen_ours"; out = f"{od}/{case}_gen.mp4"
            if os.path.exists(out) or not all(os.path.exists(f"{d}/{x}") for x in ("first.png", "ids.npz", "prompts.json")):
                continue
            prompts = json.load(open(f"{d}/prompts.json")); raw = np.load(f"{d}/ids.npz")["ids"]
            assert len(prompts) == 21 and raw.shape[0] == 81, (len(prompts), raw.shape)
            z0 = INF.encode_first_frame(vae, INF.load_image_rgb(f"{d}/first.png"))
            ctx, cids = DG.cell_context(prompts, table, kid)
            sem = torch.from_numpy(np.ascontiguousarray(DG.to_vocab(raw, vocab)))[None]
            timing = []
            lat = ER.deploy_rollout(model, z0.to(dev), ctx[None].to(dev) if ctx.dim() == 3 else ctx.to(dev),
                                    cids[None].to(dev) if cids.dim() == 1 else cids.to(dev), sem.to(dev), 20, seed=args.seed,
                                    timing=timing)
            fr = dec(lat); os.makedirs(od, exist_ok=True); VIZ.write_video(out, fr)
            json.dump(dict(ckpt=args.ckpt, seed=args.seed, frames=int(len(fr)), dit_ms_per_cell=float(np.mean(timing[2:]))),
                      open(f"{od}/timing.json", "w")); n += 1
            print(case, len(fr), round(float(np.mean(timing[2:])), 1), flush=True)
    print("DONE", n)


if __name__ == "__main__":
    main()
