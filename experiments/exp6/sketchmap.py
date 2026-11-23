#!/usr/bin/env python3
"""Hand-drawn maps for experiment 6: a top-down sketch painted with five flat colours becomes a numeric world that is
rendered with a test clip's real camera + characters (procmap.py's world / renderer, new theme "sketch").
  template  <clip> -> <out>/template.png + template.json: top-down canvas of the area the camera will see (20 cm / px),
            pale blue = ground the camera sees during the clip (draw there), white = ground outside the view; the player path (blue), boss path (purple), frame-0 view (light blue) and the
            no-build zone around the characters (pink) are drawn as guides. Draw on a copy of it.
  build     <sketch.png> <template.json> -> procmap case dir (ids.npz, prompts.json, map0.png, ...), same layout as procmap.
  synth     <template.json> -> a programmatic sketch in the same colour code (wobbly freehand blobs / strokes) for testing
            the pipeline; NOT a human drawing and must never be reported as one.
Drawing rule: ground (paving) may be painted anywhere, also over the pink no-build zone; objects (buildings, trees, rocks,
props) only outside pink — anything touching it is shrunk (buildings) or dropped, so characters are never buried.
Colour code (nearest colour within 90 RGB units, anything else = terrain):
  paving grey (150,150,150) | building red (220,40,40) | vegetation green (40,160,60) | rock brown (120,75,35) |
  prop yellow (240,200,0)
Buildings: each red blob -> house on its min-area rectangle; green blob -> trees (1 per ~9 m2); brown -> rocks (1 per ~4 m2,
size from the blob); yellow -> barrels / crates (1 per ~1.5 m2); grey -> paving texture. Objects that would touch the
no-build zone are dropped by World.add exactly as in procmap.
Usage: sketchmap.py template --clip CID --out D | build --sketch S.png --template D/template.json --case NAME --out ROOT
       | synth --template D/template.json --seed K --out S.png"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, math, os, sys
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import numpy as np, cv2, torch
import procmap as PM
PM.DATA = RT + "/data/gc25d_v10_fxcond"            # test clips' meta / latents (gc25d_v10 lacks most of them)
PM.load_source.__defaults__ = (PM.DATA,)
RES_CM = 20.0
LEGEND = {PM.PAVING: (150, 150, 150), PM.ARCH: (220, 40, 40), PM.VEG: (40, 160, 60), PM.ROCK: (120, 75, 35), PM.PROP: (240, 200, 0)}


def classify(rgb):
    cols = np.array(list(LEGEND.values()), np.float32); keys = np.array(list(LEGEND.keys()))
    d = np.linalg.norm(rgb[..., None, :].astype(np.float32) - cols, axis=-1)
    k = d.argmin(-1); out = np.where(d.min(-1) < 90, keys[k], PM.TERRAIN).astype(np.uint8)
    # guides (blue / purple / pink / light blue) are far from every legend colour => terrain
    return out


class RasterGround(PM.Ground):
    def __init__(self, paving, x0, y0, res):
        super().__init__(); self.pav, self.x0, self.y0, self.res = paving, x0, y0, res

    def classify(self, xy, dev, chunk=1 << 16):
        xy = np.asarray(xy, np.float64)
        c = np.floor((xy[:, 0] - self.x0) / self.res).astype(int); r = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (c >= 0) & (c < self.pav.shape[1]) & (r >= 0) & (r < self.pav.shape[0])
        out = np.full(len(xy), PM.TERRAIN, np.uint8)
        out[ok] = np.where(self.pav[r[ok], c[ok]], PM.PAVING, PM.TERRAIN)
        return out


def world_for(clip, seed=0):
    return PM.World(PM.load_source(clip), seed)


def template(args):
    w = world_for(args.clip)
    W_, H_ = int(math.ceil(w.nx * PM.GRID_CM / RES_CM)), int(math.ceil(w.ny * PM.GRID_CM / RES_CM))
    img = np.full((H_, W_, 3), 255, np.uint8)
    blk = cv2.resize(w.blocked.astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST)
    img[blk > 0] = (255, 215, 230)                                   # pink: no-build zone
    vis = cv2.resize((w.vdist <= 0).astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST)
    img[(vis > 0) & (blk == 0)] = (228, 240, 255)                  # pale blue: seen by the camera during the clip -> draw here
    q = lambda P: (((np.asarray(P) - [w.x0, w.y0]) / RES_CM) * 16).round().astype(np.int32)
    cs, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (170, 210, 255), 1)                 # light blue: area the camera sees during the clip
    for P, col in ((w.boss_path, (150, 60, 200)), (w.path, (30, 80, 255))):
        if len(P) > 1:
            cv2.polylines(img, [q(P)], False, col, 2, cv2.LINE_AA, 4)
    cv2.polylines(img, [q(w.foot[0])], True, (90, 170, 255), 2, cv2.LINE_AA, 4)
    for i, (k, col) in enumerate(LEGEND.items()):
        cv2.rectangle(img, (8, 8 + 22 * i), (26, 24 + 22 * i), col, -1)
        cv2.putText(img, PM.CLASS_NAMES[k], (32, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(f"{args.out}/template.png", img[..., ::-1])
    near = cv2.resize((w.vdist <= 300).astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST) & (blk == 0)
    cv2.imwrite(f"{args.out}/template_region.png", near * 255)        # seen area + 3 m, minus the no-build zone (synth)
    cv2.imwrite(f"{args.out}/template_view.png", (vis > 0).astype(np.uint8) * 255)  # seen area incl. the no-build zone
    json.dump(dict(clip=args.clip, x0=float(w.x0), y0=float(w.y0), res_cm=RES_CM, width=W_, height=H_,
                   legend={PM.CLASS_NAMES[k]: list(v) for k, v in LEGEND.items()}, legend_box=[0, 0, 140, 8 + 22 * len(LEGEND)]),
              open(f"{args.out}/template.json", "w"), indent=1)
    print(f"template {W_}x{H_} px ({W_ * RES_CM / 100:.0f} x {H_ * RES_CM / 100:.0f} m) -> {args.out}")


def theme_sketch(w, cls_map, t):
    """cls_map: uint8 [H,W] at RES_CM in template coords."""
    x0, y0, res = t["x0"], t["y0"], t["res_cm"]
    lx0, ly0, lx1, ly1 = t["legend_box"]
    cls_map = cls_map.copy(); cls_map[ly0:ly1, lx0:lx1] = PM.TERRAIN       # ignore the legend
    pav = cls_map == PM.PAVING
    w.ground = RasterGround(pav, x0, y0, res)
    g = cv2.resize(pav.astype(np.uint8), (w.nx, w.ny), interpolation=cv2.INTER_NEAREST) > 0
    w.gmap[g] = PM.PAVING
    to_w = lambda c, r: np.array([x0 + (c + 0.5) * res, y0 + (r + 0.5) * res])
    rng = w.rng; stats = {}
    blk_t = cv2.resize(w.blocked.astype(np.uint8), (cls_map.shape[1], cls_map.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    for k in (PM.ARCH, PM.ROCK, PM.VEG, PM.PROP):
        n_lab, lab, st, cen = cv2.connectedComponentsWithStats((cls_map == k).astype(np.uint8), 8)
        placed = 0
        for i in range(1, n_lab):
            area_m2 = st[i, cv2.CC_STAT_AREA] * (res / 100) ** 2
            if area_m2 < 0.5:
                continue
            ys, xs = np.nonzero(lab == i)
            pts = np.stack([xs, ys], 1)
            if k == PM.ARCH:
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts.astype(np.float32))
                c = to_w(cx, cy); yaw = math.radians(ang)
                for f in (1.0, 0.8, 0.6, 0.45):          # shrink a drawn building until it clears the characters' zone
                    wid, dep = max(rw * res * f, 300), max(rh * res * f, 300)
                    if w.add(PM.house(w, c, wid, dep, yaw, rng), [PM._rect(c, wid + 20, dep + 20, yaw)], name="sketch_house"):
                        placed += 1; break
                continue
            pts = pts[~blk_t[pts[:, 1], pts[:, 0]]]       # scatter only on drawn cells outside the no-build zone
            if not len(pts):
                continue
            per = {PM.VEG: 9.0, PM.ROCK: 4.0, PM.PROP: 1.5}[k]
            n = max(1, int(round(area_m2 / per)))
            for j in rng.choice(len(pts), size=min(n, len(pts)), replace=False):
                p = to_w(*pts[j]) + rng.uniform(-res / 2, res / 2, 2)
                if k == PM.VEG:
                    o = PM.tree(w, p) if area_m2 > 3 else PM.bush(w, p)
                elif k == PM.ROCK:
                    o = PM.rock(w, p, 60, min(260, 60 + 40 * math.sqrt(area_m2)))
                else:
                    o = PM.barrel_or_crate(w, p)
                placed += w.add(*o, name=f"sketch_{PM.CLASS_NAMES[k]}")
        stats[PM.CLASS_NAMES[k]] = placed
    w.sketch_stats = stats


def build(args):
    t = json.load(open(args.template))
    sk = cv2.imread(args.sketch, cv2.IMREAD_COLOR)[..., ::-1]
    if sk.shape[:2] != (t["height"], t["width"]):
        sk = cv2.resize(sk, (t["width"], t["height"]), interpolation=cv2.INTER_NEAREST)
    cls_map = classify(sk)
    PM.THEME_FN["sketch"] = lambda w: theme_sketch(w, cls_map, t)
    PM.THEME_DESC["sketch"] = args.desc
    rep = PM.build_case(t["clip"], "sketch", args.seed, args.case, args.out, "cuda", False)
    json.dump(dict(sketch=os.path.abspath(args.sketch), template=os.path.abspath(args.template), desc=args.desc,
                   human_drawn=args.human), open(os.path.join(args.out, args.case, "sketch.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(args.out, args.case, "sketch.png"), sk[..., ::-1])
    print("built", args.case, rep.get("checks_ok"))


def synth(args):
    """programmatic stand-in for a hand drawing (wobbly blobs and strokes); for pipeline tests only."""
    t = json.load(open(args.template)); rng = np.random.default_rng(args.seed)
    W_, H_ = t["width"], t["height"]
    img = np.full((H_, W_, 3), 255, np.uint8)

    reg = cv2.imread(os.path.join(os.path.dirname(args.template), "template_region.png"), 0) > 0
    cand = np.argwhere(reg)[:, ::-1].astype(float)                 # (x, y) cells the camera sees

    def blob(col, r):
        c = cand[rng.integers(len(cand))]; a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rr = r * (1 + 0.35 * np.sin(a * rng.integers(2, 5) + rng.uniform(0, 6))) * rng.uniform(0.7, 1.2, len(a))
        p = (c + np.stack([np.cos(a), np.sin(a)], 1) * rr[:, None]).astype(np.int32)
        cv2.fillPoly(img, [p], col, cv2.LINE_AA)

    for _ in range(int(rng.integers(1, 3))):                        # roads: thick wobbly strokes
        view = np.argwhere(cv2.imread(os.path.join(os.path.dirname(args.template), "template_view.png"), 0) > 0)[:, ::-1]
        p = np.cumsum(rng.normal(0, 10, (12, 2)), 0) + view[rng.integers(len(view))]   # ground may run under the characters
        cv2.polylines(img, [p.astype(np.int32)], False, LEGEND[PM.PAVING], int(rng.integers(12, 22)), cv2.LINE_AA)
    for col, n, r in ((LEGEND[PM.ARCH], rng.integers(2, 5), 14), (LEGEND[PM.VEG], rng.integers(3, 9), 9),
                      (LEGEND[PM.ROCK], rng.integers(2, 6), 6), (LEGEND[PM.PROP], rng.integers(2, 7), 4)):
        for _ in range(int(n)):
            blob(col, r * rng.uniform(0.6, 1.4))
    cv2.imwrite(args.out, img[..., ::-1]); print("synthetic sketch ->", args.out)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("template"); a.add_argument("--clip", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("build"); b.add_argument("--sketch", required=True); b.add_argument("--template", required=True)
    b.add_argument("--case", required=True); b.add_argument("--out", required=True); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--desc", default="a game town drawn from a sketch: dirt ground, stone-paved roads, buildings, trees, rocks, barrels and crates")
    b.add_argument("--human", action="store_true", help="the sketch was drawn by a person (recorded in sketch.json)")
    c = sub.add_parser("synth"); c.add_argument("--template", required=True); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    dict(template=template, build=build, synth=synth)[args.cmd](args)


if __name__ == "__main__":
    main()
