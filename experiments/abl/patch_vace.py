"""Patch a copy of experiments/abl/code/gc25d with a ControlNet-style (Wan VACE) condition path, cond_mode="color_vace":
colour-coded map -> frozen Wan VAE (as variant 1) -> Conv3d patch embed -> before_proj(zero) + x -> 15 trainable copies of
DiT blocks 0,2,...,28 -> after_proj(zero) hints added after the matching main blocks (VACE layout, DiffSynth wan_video_vace).
New small modules (patch embed, projections) use cond_lr; the copied blocks use the DiT lr."""
import sys, re
D = sys.argv[1]
M = open(f"{D}/model.py").read()
M = M.replace("""    extra: torch.Tensor | None = None,
) -> torch.Tensor:""", """    extra: torch.Tensor | None = None,
    vace_z: torch.Tensor | None = None,
) -> torch.Tensor:""", 1)
old = """    # blocks：v1 modules only ⇒ actionrope._block_forward (bit-identical to v1); v2 ⇒ gc_block_forward + injections
    inj_rms = []
    for i, block in enumerate(dit.blocks):"""
new = """    # paper ablation (color_vace): ControlNet-style branch, VACE layout
    hints = None
    if vace_z is not None:
        vm, vb = dit.gc_vace, dit.gc_vace_blocks
        c = vm.patch(vace_z.to(x.dtype))
        assert c.shape[2:] == (f, h, w), (tuple(c.shape), (f, h, w))
        c = vm.before_proj(rearrange(c, "b c f h w -> b (f h w) c")) + x
        hints = {}
        for j, (li, blk) in enumerate(zip(VACE_LAYERS, vb)):
            c = gradient_checkpoint_forward(
                functools.partial(_block_forward, blk, f=f, arm=None, idx=li, grid=None),
                use_gradient_checkpointing, use_gradient_checkpointing_offload, c, context, t_mod, freqs, None)
            hints[li] = vm.after_proj[j](c)
        if aux is not None:
            aux["sem_rms"] = torch.stack([v.detach().float().pow(2).mean() for v in hints.values()]).mean().sqrt()
            aux["x_rms"] = x.detach().float().pow(2).mean().sqrt()

    # blocks：v1 modules only ⇒ actionrope._block_forward (bit-identical to v1); v2 ⇒ gc_block_forward + injections
    inj_rms = []
    for i, block in enumerate(dit.blocks):"""
assert M.count(old) == 1; M = M.replace(old, new)
old = """                x, context, t_mod, freqs, None,
            )
    if aux is not None and inj_rms:"""
new = """                x, context, t_mod, freqs, None,
            )
        if hints is not None and i in hints:
            x = x + hints[i].to(x.dtype)
    if aux is not None and inj_rms:"""
assert M.count(old) == 1; M = M.replace(old, new)
M += '''

# ---------------------------------------------------------------- paper ablation: ControlNet-style (VACE) branch
VACE_LAYERS = tuple(range(0, 30, 2))


class GcVace(nn.Module):
    def __init__(self, dim, patch_size, in_dim=48, n=len(VACE_LAYERS)):
        super().__init__()
        self.patch = nn.Conv3d(in_dim, dim, kernel_size=tuple(patch_size), stride=tuple(patch_size))
        self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])
        for l in [self.before_proj, *self.after_proj]:
            nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)


def install_vace(dit, copy_weights=True):
    """adds dit.gc_vace (new, zero-init outputs) and dit.gc_vace_blocks (copies of DiT blocks 0,2,..,28)."""
    import copy
    ref = dit.patch_embedding.weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        vm = GcVace(dit.dim, dit.patch_size)
    dit.add_module("gc_vace", vm.to(device=ref.device, dtype=ref.dtype))
    dit.add_module("gc_vace_blocks", nn.ModuleList([copy.deepcopy(dit.blocks[i]) for i in VACE_LAYERS]))
    return dit
'''
open(f"{D}/model.py", "w").write(M)

T = open(f"{D}/train.py").read()
T = T.replace('choices=("ids", "color_light", "color_vae"), default="ids",', 'choices=("ids", "color_light", "color_vae", "color_vace"), default="ids",', 1)
T = T.replace('ABL_COND_NAME = {"ids": GM.SEM_COND_NAME, "color_light": GM.WB_COND_NAME, "color_vae": "gc_vae_cond"}',
              'ABL_COND_NAME = {"ids": GM.SEM_COND_NAME, "color_light": GM.WB_COND_NAME, "color_vae": "gc_vae_cond", "color_vace": "gc_vace"}', 1)
old = """        elif self.cond_mode == "color_vae":
            rgb = self.pal.to(sem.device)[sem.long()]"""
new = """        elif self.cond_mode in ("color_vae", "color_vace"):
            rgb = self.pal.to(sem.device)[sem.long()]"""
assert T.count(old) == 1; T = T.replace(old, new)
old = """            extra = self.dit.gc_vae_cond(zc.to(x0.dtype))
            sem = None"""
new = """            if self.cond_mode == "color_vae":
                extra = self.dit.gc_vae_cond(zc.to(x0.dtype))
            else:
                vace_z = zc.to(x0.dtype)
            sem = None"""
assert T.count(old) == 1; T = T.replace(old, new)
T = T.replace("""        sem, ctype, extra = batch["sem_ids"], None, None""", """        sem, ctype, extra, vace_z = batch["sem_ids"], None, None, None""", 1)
old = """            use_gradient_checkpointing=self.gradient_checkpointing and train, aux=aux, cond_type=ctype, extra=extra,
        )"""
new = """            use_gradient_checkpointing=self.gradient_checkpointing and train, aux=aux, cond_type=ctype, extra=extra,
            vace_z=vace_z,
        )"""
assert T.count(old) == 1; T = T.replace(old, new)
old = """    else:
        ref = dit.patch_embedding.weight
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            vc = nn.Conv3d(48, dit.dim, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
        nn.init.zeros_(vc.weight); nn.init.zeros_(vc.bias)
        dit.add_module("gc_vae_cond", vc.to(device=ref.device, dtype=ref.dtype))"""
new = """    else:
        if args.cond_mode == "color_vace":
            GM.install_vace(dit)
        else:
            ref = dit.patch_embedding.weight
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234)
                vc = nn.Conv3d(48, dit.dim, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
            nn.init.zeros_(vc.weight); nn.init.zeros_(vc.bias)
            dit.add_module("gc_vae_cond", vc.to(device=ref.device, dtype=ref.dtype))"""
assert T.count(old) == 1; T = T.replace(old, new)
open(f"{D}/train.py", "w").write(T)

S = open(f"{D}/abl_sample.py").read()
old = """    else:
        ref = dit.patch_embedding.weight
        vc = nn.Conv3d(48, dit.dim, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
        dit.add_module("gc_vae_cond", vc.to(device=ref.device, dtype=ref.dtype))"""
new = """    elif mode == "color_vace":
        GM.install_vace(dit)
    else:
        ref = dit.patch_embedding.weight
        vc = nn.Conv3d(48, dit.dim, kernel_size=tuple(dit.patch_size), stride=tuple(dit.patch_size))
        dit.add_module("gc_vae_cond", vc.to(device=ref.device, dtype=ref.dtype))"""
assert S.count(old) == 1; S = S.replace(old, new)
S = S.replace('cname = {"color_light": GM.WB_COND_NAME, "color_vae": "gc_vae_cond"}[mode]',
              'cname = {"color_light": GM.WB_COND_NAME, "color_vae": "gc_vae_cond", "color_vace": "gc_vace"}[mode]', 1)
old = """            zc = vae.single_encode(xv, dev).to(torch.bfloat16)
            extra = dit.gc_vae_cond(zc)

        def fwd(d, latents, timestep, ctx, sem, **kw):
            return GM.gc_forward(d, latents, timestep, ctx, None, extra=extra, **kw)"""
new = """            zc = vae.single_encode(xv, dev).to(torch.bfloat16)
            extra = dit.gc_vae_cond(zc) if mode == "color_vae" else None
        vz = zc if mode == "color_vace" else None

        def fwd(d, latents, timestep, ctx, sem, **kw):
            return GM.gc_forward(d, latents, timestep, ctx, None, extra=extra, vace_z=vz, **kw)"""
assert S.count(old) == 1; S = S.replace(old, new)
open(f"{D}/abl_sample.py", "w").write(S)
print("patched")
