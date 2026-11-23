#!/usr/bin/env python3
"""Common evaluation for the paper comparison (table 1 main, table 2 ablation) on the 100-clip test set.
Every method is brought to the MaLiang frame grid before scoring: 832x480, 16 fps, source frames 0..80 of the clip.
  * native (Ours / ablation variants): <dir>/<clip>/gen.mp4 (or <dir>/<clip>.mp4), 832x480 @16 fps, frames 0..80.
  * canvas24 (SolarRenderer, AnyGameRenderer, CWM): 1344x768 @24 fps made from source frames by i -> floor(i*16/24) and
    832x480 -> 1344x776 -> centre crop 768; inverse = output frame ceil(1.5 k) for source frame k, 4-row edge pad,
    INTER_AREA to 832x480.
Scores use frames 1..80 (frame 0 is the given first frame for every method):
  PSNR / SSIM (code/gc25d/metrics.py), LPIPS (AlexNet, lpips 0.1.4), mIoU (SegFormer-B2 segmenter fine-tuned on recorded
  RGB <-> ids, 9 classes, fx / unknown ignored; also run on GT => ceiling), FVD (I3D torchscript from StyleGAN-V, 400-d,
  16-frame windows starting at frames 1, 17, 33, 49, 65 => 5 windows / clip, 224x224 bilinear), entity metrics (time
  match, position, unrendered, hallucination; see entity_frame_stats), each also on GT frames as the segmenter ceiling.
Writes <out>/<method>/per_clip.json and summary.json; GT features cached in <out>/_gt.
Usage: eval_common.py --method NAME --dir DIR --kind native|canvas24 --out OUT [--seg seg_b2/final]"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, math, os, sys
RT = MALIANG_ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [RT + "/code/gc25d", HERE + "/pylib"]
import numpy as np, cv2, torch
import torch.nn.functional as F
import metrics as MET
DATA = RT + "/data/gc25d_v10_fxcond"
RAW = _mos.environ.get("MALIANG_RAW", _mos.path.join(MALIANG_ROOT, "data", "recordings", "data"))
TEST = _mos.path.join(MALIANG_ROOT, "experiments", "testset", "v1", "test_clips.json")
W, H, NF = 832, 480, 81
LUT = np.full(256, 255, np.uint8); LUT[:7] = np.arange(7)
LUT[[10, 12, 13, 14, 15, 16]] = 7; LUT[[11, 17, 18, 19, 20, 21]] = 8
NAMES = ["sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop", "player", "boss"]
WIN = (1, 17, 33, 49, 65)


def read_video(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[..., ::-1])
    return np.stack(fr)


def load_method(path, kind):
    v = read_video(path)
    if kind == "native":
        assert v.shape[1:3] == (H, W), v.shape
        return v[:NF].copy()
    assert v.shape[1:3] == (768, 1344), v.shape
    out = []
    for k in range(NF):
        f = v[min(math.ceil(1.5 * k), len(v) - 1)]
        f = cv2.copyMakeBorder(f, 4, 4, 0, 0, cv2.BORDER_REPLICATE)
        out.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    return np.stack(out)


_ROWS = None


def gt_clip(cid):
    """recorded frames start..start+80 -> geometry.resample_rgb (same reader as build_agr_inputs / seg_data)."""
    global _ROWS
    import geometry as GEO
    import prep_v10 as PV   # heavy import (training stack) only when GT is not cached yet
    if _ROWS is None:
        _ROWS = {json.loads(l)["clip_id"]: json.loads(l) for l in open(DATA + "/index/clips.jsonl")}
    r = _ROWS[cid]; t = PV.RunTars(f"{RAW}/{r['subset']}/{r['scene']}/{r['run']}")
    out = [GEO.resample_rgb(cv2.imdecode(np.frombuffer(t.get(int(r["start"]) + k, "jpg"), np.uint8), cv2.IMREAD_COLOR)[..., ::-1])
           for k in range(NF)]
    for fh in t.fh.values():
        fh.close()
    return np.stack(out)


class Scorer:
    def __init__(self, seg_dir, dev="cuda"):
        import lpips
        from transformers import SegformerForSemanticSegmentation
        self.dev = dev
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        self.seg = SegformerForSemanticSegmentation.from_pretrained(seg_dir).to(dev).eval()
        self.i3d = torch.jit.load(os.path.join(MALIANG_ROOT, "assets", "i3d_torchscript.pt")).to(dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def segment(self, v):
        out = []
        for s in range(0, len(v), 16):
            x = torch.from_numpy(v[s:s + 16]).to(self.dev).permute(0, 3, 1, 2).float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = self.seg(pixel_values=(x - self.mean) / self.std).logits
            out.append(F.interpolate(lo.float(), (H, W), mode="bilinear", align_corners=False).argmax(1).byte().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def lpips(self, a, b):
        r = []
        for s in range(0, len(a), 16):
            x = torch.from_numpy(a[s:s + 16]).to(self.dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            y = torch.from_numpy(b[s:s + 16]).to(self.dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            r.append(self.lp(x, y).flatten().cpu())
        return torch.cat(r).numpy()

    @torch.no_grad()
    def i3d_feats(self, v):
        clips = np.stack([v[s:s + 16] for s in WIN])                       # [5,16,H,W,3]
        x = torch.from_numpy(clips).to(self.dev).permute(0, 4, 1, 2, 3).float()   # [5,3,16,H,W] 0..255
        x = F.interpolate(x.flatten(1, 2), (224, 224), mode="bilinear", align_corners=False).view(5, 3, 16, 224, 224)
        x = x / 127.5 - 1
        return self.i3d(x, rescale=False, resize=False, return_features=True).cpu().numpy()


def miou_from_conf(conf):
    inter = np.diag(conf); union = conf.sum(0) + conf.sum(1) - inter; present = conf.sum(1) > 0
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou[present])), {NAMES[c]: (float(iou[c]) if present[c] else None) for c in range(9)}


def conf_matrix(pred, lbl):
    v = lbl != 255
    return np.bincount(lbl[v].astype(np.int64) * 9 + pred[v], minlength=81).reshape(9, 9)


# ---------------------------------------------------------------------------------------------------------------
# Entity metrics (AnyGameRenderer protocol, adapted): entities = player (seg class 7) and boss (seg class 8); ground truth
# = the numeric map's entity masks. Per frame and entity:
#   present_gt   mask area >= MIN_PX;   present_gen  segmenter labels >= 30 % of the mask (dilated 15 px) as that entity
#   time match   share of frames where present_gen == present_gt
#   position     on frames present in both: IoU between the segmenter's entity pixels near the mask and the mask
#                (reported as mean IoU and success rate IoU >= 0.25)
#   unrendered   share of present_gt frames where >= 50 % of the mask still shows the method's proxy colour (dRGB < 45)
#                -- only for methods fed a coloured proxy video (SolarRenderer, AnyGameRenderer, variants 1/2)
#   hallucination share of frames with a connected component of an entity class (>= MIN_PX) that overlaps no dilated
#                ground-truth mask of that entity by >= 10 %
MIN_PX = 150
SOLAR_PROXY = {7: (220, 20, 60), 8: (255, 0, 0)}
AGR_PROXY = {7: (40, 90, 255), 8: (235, 40, 40)}


def proxy_palette(seed=0):
    g = torch.Generator().manual_seed(seed)
    pal = torch.randint(16, 240, (256, 3), generator=g, dtype=torch.int64).to(torch.uint8)
    pal[255] = 255
    return pal.numpy()


def entity_frame_stats(seg, lbl, raw, rgb, proxy):
    """seg/lbl/raw [F,H,W], rgb [F,H,W,3] -> per-entity lists of per-frame tuples."""
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)); out = {}
    pal = proxy_palette() if proxy == "abl" else None
    for c in (7, 8):
        rows = []
        for f in range(len(seg)):
            m = lbl[f] == c; area = int(m.sum()); pg = area >= MIN_PX
            md = cv2.dilate(m.astype(np.uint8), k15) > 0
            cls = seg[f] == c
            pres = bool(pg and cls[m].mean() >= 0.3) if area else False
            iou = None
            if pg and pres:
                near = cls & md; iou = float((near & m).sum() / max((near | m).sum(), 1))
            unr = None
            if pg and proxy in ("solar", "agr", "abl"):
                if proxy == "abl":
                    ref = pal[raw[f][m]].astype(np.int16)
                else:
                    ref = np.array((SOLAR_PROXY if proxy == "solar" else AGR_PROXY)[c], np.int16)[None]
                unr = bool((np.abs(rgb[f][m].astype(np.int16) - ref).max(-1) < 45).mean() >= 0.5)
            n, lab, st, _ = cv2.connectedComponentsWithStats(cls.astype(np.uint8), 8)
            hal = any(st[i, cv2.CC_STAT_AREA] >= MIN_PX and md[lab == i].mean() < 0.1 for i in range(1, n))
            rows.append((pg, pres if pg else bool(cls.sum() >= MIN_PX), iou, unr, hal))
        out[c] = rows
    return out


def entity_summary(rows_all):
    """rows_all: list over clips of entity_frame_stats outputs -> pooled metrics."""
    res = {}
    for c, name in ((7, "player"), (8, "boss")):
        rows = [r for d in rows_all for r in d[c]]
        if not rows:
            continue
        tm = float(np.mean([r[0] == r[1] for r in rows]))
        ious = [r[2] for r in rows if r[2] is not None]
        unr = [r[3] for r in rows if r[3] is not None]
        res[name] = dict(time_match=tm, pos_iou=float(np.mean(ious)) if ious else None,
                         pos_success=float(np.mean([i >= 0.25 for i in ious])) if ious else None,
                         unrendered=float(np.mean(unr)) if unr else None, hallucination=float(np.mean([r[4] for r in rows])),
                         frames_gt_present=int(sum(r[0] for r in rows)))
    keys = ("time_match", "pos_iou", "pos_success", "unrendered", "hallucination")
    res["mean"] = {k: (float(np.mean([res[n][k] for n in ("player", "boss") if n in res and res[n][k] is not None]))
                       if any(n in res and res[n][k] is not None for n in ("player", "boss")) else None) for k in keys}
    return res


def fvd(f1, f2):
    from scipy import linalg
    m1, m2 = f1.mean(0), f2.mean(0); s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    cs = linalg.sqrtm(s1 @ s2)
    cs = cs[0] if isinstance(cs, tuple) else cs        # scipy < 1.16 returns (sqrtm, err) only with disp=False
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def main():
    a = argparse.ArgumentParser(); a.add_argument("--method", required=True); a.add_argument("--dir", required=True)
    a.add_argument("--kind", choices=("native", "canvas24"), required=True); a.add_argument("--out", required=True)
    a.add_argument("--seg", default=os.path.join(MALIANG_ROOT, "assets", "seg_b2_final"))
    a.add_argument("--proxy", choices=("none", "solar", "agr", "abl"), default="none",
                   help="proxy colours the method was fed (for the unrendered rate)")
    args = a.parse_args()
    clips = json.load(open(TEST)); sc = Scorer(args.seg)
    gd = f"{args.out}/_gt"; md = f"{args.out}/{args.method}"; os.makedirs(gd, exist_ok=True); os.makedirs(md, exist_ok=True)
    per, feats, conf_all, conf_gt, missing = [], [], np.zeros((9, 9), np.int64), np.zeros((9, 9), np.int64), []
    ent_rows, ent_gt = [], []
    gt_feats = []
    for cid in clips:
        p = next((q for q in (f"{args.dir}/{cid}/gen.mp4", f"{args.dir}/{cid}_gen.mp4", f"{args.dir}/{cid}.mp4", f"{args.dir}/{cid}/rendered.mp4") if os.path.exists(q)), None)
        gcache = f"{gd}/{cid}.npz"
        if not os.path.exists(gcache):
            gt = gt_clip(cid)
            lbl = LUT[np.load(f"{DATA}/cond/{cid}.npz")["ids"][:NF]]
            np.savez_compressed(gcache, gt=gt, lbl=lbl, i3d=sc.i3d_feats(gt), seg=sc.segment(gt))
        g = np.load(gcache); gt, lbl = g["gt"], g["lbl"]
        gt_feats.append(g["i3d"]); conf_gt += conf_matrix(g["seg"][1:], lbl[1:])
        ent_gt.append(entity_frame_stats(g["seg"][1:], lbl[1:], np.zeros_like(lbl[1:]), gt[1:], "none"))
        if p is None:
            missing.append(cid); continue
        v = load_method(p, args.kind)
        if len(v) < NF:
            missing.append(cid); continue
        cm = MET.clip_metrics(v, gt)
        lp = sc.lpips(v[1:], gt[1:])
        sg = sc.segment(v[1:]); c = conf_matrix(sg, lbl[1:]); conf_all += c
        raw = np.load(f"{DATA}/cond/{cid}.npz")["ids"][1:NF]
        ent_rows.append(entity_frame_stats(sg, lbl[1:], raw, v[1:], args.proxy))
        feats.append(sc.i3d_feats(v))
        per.append(dict(clip_id=cid, psnr=cm.get("psnr_mean"), ssim=cm.get("ssim_mean"), lpips=float(lp.mean()),
                        miou=miou_from_conf(c)[0]))
        print(cid, per[-1], flush=True)
    json.dump(per, open(f"{md}/per_clip.json", "w"), indent=1)
    np.savez_compressed(f"{md}/i3d_feats.npz", feats=np.concatenate(feats) if feats else np.zeros((0, 400)),
                        conf=conf_all, clips=np.array([r["clip_id"] for r in per]))
    gtf = np.concatenate(gt_feats)
    res = dict(method=args.method, dir=args.dir, kind=args.kind, n=len(per), missing=missing,
               psnr=float(np.mean([r["psnr"] for r in per])), ssim=float(np.mean([r["ssim"] for r in per])),
               lpips=float(np.mean([r["lpips"] for r in per])),
               miou_pooled=miou_from_conf(conf_all)[0], iou_per_class=miou_from_conf(conf_all)[1],
               miou_clip_mean=float(np.mean([r["miou"] for r in per])),
               fvd=fvd(np.concatenate(feats), gtf) if len(feats) >= 20 else None, fvd_windows=len(per) * len(WIN),
               segmenter_ceiling_on_gt=miou_from_conf(conf_gt)[0],
               entity=entity_summary(ent_rows), entity_ceiling_on_gt=entity_summary(ent_gt), proxy=args.proxy,
               per_subset={s: dict(lpips=float(np.mean([r["lpips"] for r in per if ("__R10_" in r["clip_id"]) == (s == "real")])),
                                   miou=float(np.mean([r["miou"] for r in per if ("__R10_" in r["clip_id"]) == (s == "real")])))
                           for s in ("real", "stylized") if any((("__R10_" in r["clip_id"]) == (s == "real")) for r in per)})
    json.dump(res, open(f"{md}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("iou_per_class", "missing", "entity_ceiling_on_gt")}, indent=1))


if __name__ == "__main__":
    main()
