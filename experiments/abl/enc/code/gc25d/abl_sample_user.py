#!/usr/bin/env python3
"""sample_user.py (own first frame + id video + prompts, no GT) for the interface-ablation checkpoints: imports
abl_sample (which patches sample.load_dit / generate / model_ids / infer.load_vae for color_light / color_vae) and turns
sample.warmup into a no-op for those two modes (warmup only exists to make id-mode runs bit-identical to clip mode and
would push raw ids through a model that has no id encoder). Same CLI as sample_user.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_sample as A
_w = A.S.warmup
A.S.warmup = lambda args, dit, tt, ids_model, prompts: (_w(args, dit, tt, ids_model, prompts) if A._STATE["mode"] == "ids" else 0.0)
import sample_user as U
_cv = U.ckpt_vocab


def _ckpt_vocab(path):
    """colour-interface checkpoints have no gc_sem_cond to read the vocab from: take it from the metadata (train.py writes
    "vocab"; these runs used parts_fx). The raw ids are then coloured by abl_sample.model_ids."""
    with A.safe_open(path, "pt") as f:
        meta = f.metadata() or {}
    return meta.get("vocab", "parts_fx") if meta.get("cond_mode", "ids") != "ids" else _cv(path)


U.ckpt_vocab = _ckpt_vocab
if __name__ == "__main__":
    U.main()
