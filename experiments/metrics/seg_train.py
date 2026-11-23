#!/usr/bin/env python3
"""mIoU segmenter: SegFormer-B2 (ADE20k init, new 9-class head) fine-tuned on recorded RGB <-> condition ids pairs
(seg_data.py). 832x480 input, random h-flip + mild colour jitter (photometric only; geometry must stay aligned).
Reports per-class IoU on the held-out val pairs (runs of the paper test set excluded) => segmenter ceiling for the paper.
Usage: seg_train.py --data segdata --out seg_b2 [--iters 12000] [--bs 8]"""
import argparse, glob, json, os, random, time
import numpy as np, cv2, torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
NC = 9
NAMES = ["sky", "terrain", "paving", "architecture", "rock", "vegetation", "prop", "player", "boss"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Pairs(torch.utils.data.Dataset):
    def __init__(self, root, train):
        self.imgs = sorted(glob.glob(f"{root}/img/*.jpg")); self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.imread(p)[..., ::-1].astype(np.float32) / 255.0
        lbl = cv2.imread(p.replace("/img/", "/lbl/").replace(".jpg", ".png"), 0)
        if self.train:
            if random.random() < 0.5:
                img, lbl = img[:, ::-1], lbl[:, ::-1]
            img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
        return torch.from_numpy(img.copy()).permute(2, 0, 1), torch.from_numpy(lbl.copy()).long()


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); inter = np.zeros(NC); union = np.zeros(NC); gt_px = np.zeros(NC)
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(pixel_values=norm(x)).logits
        pr = F.interpolate(lo.float(), y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        v = y != 255
        for c in range(NC):
            a, b = (pr == c) & v, (y == c) & v
            inter[c] += (a & b).sum().item(); union[c] += (a | b).sum().item(); gt_px[c] += b.sum().item()
    model.train()
    iou = {NAMES[c]: (inter[c] / union[c] if union[c] > 0 else None) for c in range(NC)}
    present = [v for c, v in enumerate(iou.values()) if v is not None and gt_px[c] > 0]
    return dict(miou=float(np.mean(present)), iou=iou, gt_frac={NAMES[c]: gt_px[c] / gt_px.sum() for c in range(NC)})


def main():
    a = argparse.ArgumentParser(); a.add_argument("--data", required=True); a.add_argument("--out", required=True)
    a.add_argument("--iters", type=int, default=12000); a.add_argument("--bs", type=int, default=8)
    a.add_argument("--lr", type=float, default=6e-5); a.add_argument("--init", default="nvidia/segformer-b2-finetuned-ade-512-512")
    args = a.parse_args(); os.makedirs(args.out, exist_ok=True); dev = "cuda"
    global NC, NAMES
    info = json.load(open(f"{args.data}/info.json"))
    if "classes" in info:
        NAMES = list(info["classes"]); NC = len(NAMES)        # 9, or 10 with the fx class (seg_data_fx.py)
    torch.manual_seed(0); random.seed(0)
    model = SegformerForSemanticSegmentation.from_pretrained(args.init, num_labels=NC, ignore_mismatched_sizes=True,
                                                             id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)}).to(dev)
    tr = torch.utils.data.DataLoader(Pairs(f"{args.data}/train", True), batch_size=args.bs, shuffle=True, num_workers=8,
                                     drop_last=True, persistent_workers=True)
    va = torch.utils.data.DataLoader(Pairs(f"{args.data}/val", False), batch_size=8, num_workers=4)
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([dict(params=body, lr=args.lr), dict(params=head, lr=args.lr * 10)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * (1 - s / args.iters) ** 1.0)
    step, t0, log = 0, time.time(), open(f"{args.out}/train_log.jsonl", "a")
    while step < args.iters:
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(pixel_values=norm(x)).logits
            loss = F.cross_entropy(F.interpolate(lo.float(), y.shape[-2:], mode="bilinear", align_corners=False), y, ignore_index=255)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                log.write(json.dumps(dict(step=step, loss=loss.item(), s_per_it=(time.time() - t0) / step)) + "\n"); log.flush()
            if step % 3000 == 0 or step == args.iters:
                ev = evaluate(model, va, dev); ev["step"] = step
                log.write(json.dumps(ev) + "\n"); log.flush(); print(step, ev["miou"], flush=True)
                model.save_pretrained(f"{args.out}/final")
                json.dump(ev, open(f"{args.out}/val_metrics.json", "w"), indent=1)
            if step >= args.iters:
                break
    print("DONE")


if __name__ == "__main__":
    main()
