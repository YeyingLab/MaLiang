#!/usr/bin/env python3
"""sample.py for the interface-ablation checkpoints (variants 1 and 2), same clip mode / sampler / seed / outputs.
The checkpoint's metadata "cond_mode" picks the condition path exactly as train.py (abl patch) built it:
  color_light  colored ids (train.proxy_palette, seed 0) -> gc_wb_cond (WhiteboxCondEncoder), cond_type "whitebox";
  color_vae    colored ids -> frozen Wan VAE (fp32, single_encode) -> zero-init Conv3d(48 -> dim) "gc_vae_cond",
               added after the patch embedding (the VAE latent is computed once per clip; it does not depend on t).
  ids          falls through to the stock sample.py path.
sem_mode unknown: every id -> 255 before colouring (255 -> white), i.e. variant (4)'s input in each interface.
Usage: same CLI as sample.py, e.g. abl_sample.py --ckpt runs/color_light/step-5000.safetensors --clip_ids @todo.txt --out_dir D"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file
import sample as S
GM, INF = S.GM, S.INF
_STATE = {"mode": "ids", "pipe": None}


def proxy_palette(seed: int = 0) -> torch.Tensor:
    """copy of train.proxy_palette (train.py imports accelerate / deepspeed; the table must stay bit-identical)."""
    g = torch.Generator().manual_seed(seed)
    pal = torch.randint(16, 240, (256, 3), generator=g, dtype=torch.int64).to(torch.uint8)
    pal[GM.UNKNOWN_ID] = 255
    return pal


def load_dit(ckpt, base, vocab, model_dir, device):
    with safe_open(ckpt, "pt") as f:
        meta = f.metadata() or {}
    mode = meta.get("cond_mode", "ids"); _STATE["mode"] = mode
    if mode == "ids":
        return _orig_load_dit(ckpt, base, vocab, model_dir, device)
    S.GEO.check_ckpt_res(ckpt, "--ckpt")
    t0 = time.time()
    dit = GM.load_dit(model_dir, device=device)
    if mode == "color_light":
        GM.install_wb_cond(dit); dit.cond_type = "whitebox"
    elif mode == "color_vace":
        GM.install_vace(dit)
    else:
        ref = dit.patch_embedding.weight
        vc = nn.Conv3d(48, dit.dim, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
        dit.add_module("gc_vae_cond", vc.to(device=ref.device, dtype=ref.dtype))
    sd = load_file(ckpt)
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    cname = {"color_light": GM.WB_COND_NAME, "color_vae": "gc_vae_cond", "color_vace": "gc_vace"}[mode]
    assert not unexpected, unexpected[:5]
    assert not [k for k in missing if k.startswith(cname)], f"{cname} weights missing in {ckpt}"
    assert len(missing) == 0, f"{len(missing)} DiT keys missing, e.g. {missing[:3]}"
    dit = dit.to(device).eval().requires_grad_(False)
    dit.freqs = tuple(t.to(device) for t in dit.freqs)
    return dit, {"vocab": meta.get("vocab", "parts_fx"), "cond_mode": mode, "sem_cond_zero": False,
                 "dit_load_sec": time.time() - t0, "n_tensors": len(sd), "metadata": meta}


def load_vae(device, model_dir):
    pipe = _orig_load_vae(device, model_dir); _STATE["pipe"] = pipe
    return pipe


def model_ids(ids, vocab, sem_mode):
    if _STATE["mode"] == "ids":
        return _orig_model_ids(ids, vocab, sem_mode)
    return np.full_like(ids, GM.UNKNOWN_ID) if sem_mode == "unknown" else ids     # colour the raw parts_fx ids


def generate(args, dit, tt, z0, ids_model, prompts):
    mode = _STATE["mode"]
    if mode == "ids":
        return _orig_generate(args, dit, tt, z0, ids_model, prompts)
    assert args.cfg == 1.0 and args.sem_cfg == 1.0, "ablation sampling uses cfg = sem_cfg = 1 (as all table-2 rows)"
    dev = args.device
    context, context_ids = tt.context(prompts, dev)
    rgb = proxy_palette().to(dev)[torch.from_numpy(np.ascontiguousarray(ids_model)).to(dev).long()].unsqueeze(0)  # [1,81,H,W,3]
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    if mode == "color_light":
        lat = INF.sample(dit, z0, context, context_ids, rgb, steps=args.steps, shift=args.shift, cfg=1.0, sem_cfg=1.0,
                         seed=args.seed, forward=GM.gc_forward)
    else:
        vae = _STATE.get("vae32")
        if vae is None:     # train.py encodes the colour video with prep_v10.load_vae: the bf16 VAE cast to fp32 weights
            import copy
            vae = copy.deepcopy(_STATE["pipe"].vae); vae.model.float(); _STATE["vae32"] = vae
        with torch.no_grad():
            xv = rgb.permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1.0)
            zc = vae.single_encode(xv, dev).to(torch.bfloat16)
            extra = dit.gc_vae_cond(zc) if mode == "color_vae" else None
        vz = zc if mode == "color_vace" else None

        def fwd(d, latents, timestep, ctx, sem, **kw):
            return GM.gc_forward(d, latents, timestep, ctx, None, extra=extra, vace_z=vz, **kw)
        lat = INF.sample(dit, z0, context, context_ids, None, steps=args.steps, shift=args.shift, cfg=1.0, sem_cfg=1.0,
                         seed=args.seed, forward=fwd)
    torch.cuda.synchronize()
    return lat, time.time() - t0


_orig_load_dit, _orig_load_vae, _orig_model_ids, _orig_generate = S.load_dit, INF.load_vae, S.model_ids, S.generate
S.load_dit, INF.load_vae, S.model_ids, S.generate = load_dit, load_vae, model_ids, generate

if __name__ == "__main__":
    S.main()
