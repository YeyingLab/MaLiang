#!/usr/bin/env python3
"""Per-clip extras for the main table (no new generation), written next to eval_common.py's outputs:
  * per-clip 9x9 confusion matrix and per-clip entity counts (so pooled mIoU / entity metrics can be bootstrapped);
  * temporal warping error (Lai et al., ECCV 2018): E_warp = mean_t  sum_x M_t(x) ||O_t(x) - W(O_{t-1})(x)||^2 / sum_x M_t(x),
    RGB in [0,1], frames 1..80 (pairs 1-2 .. 79-80), RAFT-large (torchvision C_T_SKHT_V2) optical flow, occlusion mask M from a
    forward-backward check. Two flow sources:
      warp_gt   flow + occlusion from the GROUND-TRUTH video (the motion every method is asked to follow; Lai et al. use the
                input video's flow) -> penalises flicker and motion that does not follow the scene;
      warp_self flow + occlusion from the method's own video -> pure frame-to-frame stability of whatever it draws.
Same frame grid as eval_common (load_method). Usage: extra_eval.py --method NAME --dir DIR --kind native|canvas24 [--proxy P]
or --method _gt (ground truth itself). Writes <out>/<method>/extra_per_clip.json."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "weights/torch"))
import eval_common as EC
import numpy as np, torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


class Flow:
    def __init__(self, dev="cuda"):
        self.dev = dev
        self.net = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()

    @torch.no_grad()
    def __call__(self, a, b, bs=8):
        """a, b uint8 [N,H,W,3] -> flow a->b [N,2,H,W] (float32, on device)."""
        out = []
        for s in range(0, len(a), bs):
            x = torch.from_numpy(a[s:s + bs]).to(self.dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            y = torch.from_numpy(b[s:s + bs]).to(self.dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            out.append(self.net(x, y, num_flow_updates=12)[-1])
        return torch.cat(out)


def warp(x, flo):
    """backward warp x [N,C,H,W] with flow [N,2,H,W] (target -> source)."""
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    px = gx[None] + flo[:, 0]; py = gy[None] + flo[:, 1]
    grid = torch.stack((2 * px / (w - 1) - 1, 2 * py / (h - 1) - 1), -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def occ_mask(f_tb, f_bt):
    """f_tb: flow t->t-1, f_bt: flow t-1->t. Valid where the round trip closes (Sundaram et al. / Lai et al. check)."""
    f_bt_w = warp(f_bt, f_tb)
    d = (f_tb + f_bt_w).pow(2).sum(1); m = f_tb.pow(2).sum(1) + f_bt_w.pow(2).sum(1)
    return (d < 0.01 * m + 0.5).float()


@torch.no_grad()
def warp_err(v, f_tb, mask):
    """v uint8 [T,H,W,3] frames 1..80; f_tb/mask for pairs (t-1 -> t), t = 2..80 -> per-pair errors."""
    x = torch.from_numpy(v).to(f_tb.device).permute(0, 3, 1, 2).float() / 255
    cur, prev = x[1:], x[:-1]
    e = (cur - warp(prev, f_tb)).pow(2).sum(1)
    return ((e * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).cpu().numpy()


def flows(fl, v):
    cur, prev = v[1:], v[:-1]
    f_tb = fl(cur, prev); f_bt = fl(prev, cur)
    return f_tb, occ_mask(f_tb, f_bt)


def ent_counts(rows):
    out = {}
    for c, name in ((7, "player"), (8, "boss")):
        r = rows[c]; ious = [x[2] for x in r if x[2] is not None]; unr = [x[3] for x in r if x[3] is not None]
        out[name] = dict(n=len(r), tm=int(sum(x[0] == x[1] for x in r)), iou_sum=float(sum(ious)), iou_n=len(ious),
                         unr=int(sum(unr)), unr_n=len(unr), hal=int(sum(x[4] for x in r)))
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument("--method", required=True); a.add_argument("--dir", default="")
    a.add_argument("--kind", choices=("native", "canvas24"), default="native"); a.add_argument("--proxy", default="none")
    a.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "main/eval"))
    a.add_argument("--seg", default=os.path.join(EC.MALIANG_ROOT, "assets", "seg_b2_final")); a.add_argument("--limit", type=int, default=0)
    args = a.parse_args()
    clips = json.load(open(EC.TEST))[: args.limit or None]
    is_gt = args.method == "_gt"
    sc = None if is_gt else EC.Scorer(args.seg)
    fl = Flow(); md = f"{args.out}/{args.method}"; os.makedirs(md, exist_ok=True)
    res = []
    for cid in clips:
        g = np.load(f"{args.out}/_gt/{cid}.npz"); gt, lbl = g["gt"][1:], g["lbl"][1:]
        f_gt, m_gt = flows(fl, gt)
        if is_gt:
            v, sg = gt, g["seg"][1:]
            raw = np.zeros_like(lbl)
        else:
            p = next((q for q in (f"{args.dir}/{cid}/gen.mp4", f"{args.dir}/{cid}_gen.mp4", f"{args.dir}/{cid}.mp4",
                                  f"{args.dir}/{cid}/rendered.mp4") if os.path.exists(q)), None)
            if p is None:
                print("MISSING", cid, flush=True); continue
            v = EC.load_method(p, args.kind)[1:]
            sg = sc.segment(v)
            raw = np.load(f"{EC.DATA}/cond/{cid}.npz")["ids"][1:EC.NF]
        conf = EC.conf_matrix(sg, lbl)
        ent = ent_counts(EC.entity_frame_stats(sg, lbl, raw, v, "none" if is_gt else args.proxy))
        e_gt = warp_err(v, f_gt, m_gt)
        if is_gt:
            e_self, valid_self = e_gt, float(m_gt.mean())
        else:
            f_s, m_s = flows(fl, v); e_self = warp_err(v, f_s, m_s); valid_self = float(m_s.mean())
        res.append(dict(clip_id=cid, conf=conf.flatten().tolist(), entity=ent,
                        warp_gt=float(e_gt.mean()), warp_self=float(e_self.mean()),
                        valid_gt=float(m_gt.mean()), valid_self=valid_self))
        print(cid, f"warp_gt {res[-1]['warp_gt']*1e3:.3f}e-3 warp_self {res[-1]['warp_self']*1e3:.3f}e-3 "
              f"valid {res[-1]['valid_gt']:.2f}/{valid_self:.2f}", flush=True)
        torch.cuda.empty_cache()
    json.dump(res, open(f"{md}/extra_per_clip{'_limit' if args.limit else ''}.json", "w"))
    print("DONE", args.method, len(res), "warp_gt %.3fe-3 warp_self %.3fe-3" % (
        1e3 * np.mean([r["warp_gt"] for r in res]), 1e3 * np.mean([r["warp_self"] for r in res])), flush=True)


if __name__ == "__main__":
    main()
