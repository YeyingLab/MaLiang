"""Patch the .36 runtime copy so that ablation variant (2) -- colour-coded map -> light encoder (gc_wb_cond) -- can be
distilled and evaluated with the same data path, dropout and budget as the integer variant (3).
Everything is guarded by the environment variable COLOR_IDS=1; without it the code behaves exactly as before.
With COLOR_IDS=1:
  * a checkpoint with gc_wb_cond keys reports vocab "parts_fx" (the data loader then delivers raw parts_fx ids, as for (3));
  * GcDiT built as "ids" becomes "whitebox" (installs gc_wb_cond, the encoder variant (2) was trained with);
  * wb_embed colours a raw-id tensor [b,T,H,W] with the ablation palette (train.proxy_palette, seed 0; 255 -> white),
    exactly what experiments/abl/code/gc25d/abl_sample.py does at sampling time;
  * exported whitebox checkpoints carry cond_key "wb_rgb" and cond_mode "color_light";
  * model.sem_vocab answers "parts_fx" for a dit that only has gc_wb_cond (used in log lines)."""
import re, sys
M = sys.argv[1]
p = f"{M}/gcdistill/gc_dit.py"; s = open(p).read()
assert "COLOR_IDS" not in s, "already patched"
s = s.replace('''def ckpt_vocab(path: str) -> str | None:
    """ckpt 头里 gc_sem_cond.proj_in 的形状 → 词表（ids / parts）；没有 gc_sem_cond ⇒ None。"""
    hdr, _ = runlib.read_ckpt_header(path)
    return GM.vocab_from_state(hdr)''', '''def ckpt_vocab(path: str) -> str | None:
    """ckpt 头里 gc_sem_cond.proj_in 的形状 → 词表（ids / parts）；没有 gc_sem_cond ⇒ None。"""
    hdr, _ = runlib.read_ckpt_header(path)
    v = GM.vocab_from_state(hdr)
    if v is None and _COLOR_IDS and any(k.startswith(GM.WB_COND_NAME + ".") or k.startswith("dit." + GM.WB_COND_NAME + ".") for k in hdr):
        return "parts_fx"                       # COLOR_IDS: colour variant consumes raw parts_fx ids
    return v''')
s = s.replace('''def wb_embed(dit, wb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:''', '''import os as _os
_COLOR_IDS = _os.environ.get("COLOR_IDS", "0") == "1"
_PALETTE = None


def _id_palette(device):
    """train.proxy_palette(seed 0): the palette ablation variant (2) was trained with; 255 (unknown) -> white."""
    global _PALETTE
    if _PALETTE is None:
        g = torch.Generator().manual_seed(0)
        pal = torch.randint(16, 240, (256, 3), generator=g, dtype=torch.int64).to(torch.uint8)
        pal[GM.UNKNOWN_ID] = 255
        _PALETTE = pal
    return _PALETTE.to(device)


def wb_embed(dit, wb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:''')
s = s.replace('''    if wb.dtype != torch.uint8:
        raise TypeError(f"白模必须是 uint8 像素值，得到 {wb.dtype} —— 是不是把 ARoPE 的 known 掩码传进来了？")
    b, f = ref.shape[0], ref.shape[2]''', '''    if wb.dtype != torch.uint8:
        raise TypeError(f"白模必须是 uint8 像素值，得到 {wb.dtype} —— 是不是把 ARoPE 的 known 掩码传进来了？")
    if _COLOR_IDS and wb.dim() == 4:            # raw ids -> colour-coded RGB, as variant (2) was trained
        wb = _id_palette(wb.device)[wb.long()]
    b, f = ref.shape[0], ref.shape[2]''')
s = s.replace('''        if cond_type not in GM.COND_TYPES:
            raise ValueError(f"cond_type 只能是 {GM.COND_TYPES}，得到 {cond_type!r}")
        if dit is None:''', '''        if cond_type not in GM.COND_TYPES:
            raise ValueError(f"cond_type 只能是 {GM.COND_TYPES}，得到 {cond_type!r}")
        if _COLOR_IDS and cond_type == "ids":
            cond_type = "whitebox"
        if dit is None:''')
s = s.replace('''        ck = (metadata or {}).get("cond_key")''', '''        if _COLOR_IDS:
            metadata = dict(metadata or {}, cond_key="wb_rgb", cond_mode="color_light", vocab="parts_fx")
        ck = (metadata or {}).get("cond_key")''')
for k in ("_COLOR_IDS and cond_type == \"ids\"", "_id_palette(wb.device)", "cond_mode=\"color_light\"", "return \"parts_fx\""):
    assert k in s, k
open(p, "w").write(s)
p = f"{M}/model.py"; s = open(p).read()
s = s.replace('''def sem_vocab(dit) -> str:
    """dit 上已装的 gc_sem_cond 的词表。"""
    return getattr(dit, SEM_COND_NAME).vocab''', '''def sem_vocab(dit) -> str:
    """dit 上已装的 gc_sem_cond 的词表。"""
    if not hasattr(dit, SEM_COND_NAME) and hasattr(dit, WB_COND_NAME) and os.environ.get("COLOR_IDS", "0") == "1":
        return "parts_fx"
    return getattr(dit, SEM_COND_NAME).vocab''')
assert 'os.environ.get("COLOR_IDS"' in s and re.search(r"^import os", s, re.M)
open(p, "w").write(s)
# common_gc.sh: allow 3 GPUs
p = f"{M}/gcdistill/common_gc.sh"; s = open(p).read()
s = s.replace("  1|2|4|8) : ;;", "  1|2|3|4|8) : ;;"); open(p, "w").write(s)
print("patched")
