#!/usr/bin/env python3
"""AnyGameRenderer inputs for the paper test set (same 100 clips as SolarRenderer). CPU only.
primitive.mp4: the recording's grey whitebox render (raw tars, frames s..s+104) with flat proxy colours painted from
the id map: player parts -> blue, boss parts -> red, boss attack effects -> yellow; 16 -> 24 fps (158 frames = 17*9+5),
832x480 -> 1344x768 (same resize / crop as the SolarRenderer inputs). spec.json: AnyGameRenderer's caption spec, built
only from information MaLiang also gets (scene name, look); colours present in the clip only.
Usage: build_agr_inputs.py --testset <v1 dir> --out <dir>"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, os, subprocess, sys
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import numpy as np, cv2
import geometry as GEO
import prep_v10 as PV
DATA = RT + "/data/gc25d_v10_fxcond"
RAW = _mos.environ.get("MALIANG_RAW", _mos.path.join(MALIANG_ROOT, "data", "recordings", "data"))
OUT_W, OUT_H, PAD_H = 1344, 768, 776
COL = {"blue": (40, 90, 255), "red": (235, 40, 40), "yellow": (250, 220, 30)}
PLAYER = [10, 12, 13, 14, 15, 16]; BOSS = [11, 17, 18, 19, 20, 21]; FX = [22, 23, 24, 25]


def to_canvas(img, interp):
    x = cv2.resize(img, (OUT_W, PAD_H), interpolation=interp)
    o = (PAD_H - OUT_H) // 2
    return x[o:o + OUT_H]


def spec_for(r, present):
    sc = " ".join(r["scene"].split("_")[1:]).lower()
    real = r["subset"] != "v9_stylized"
    look = "photoreal" if real else "stylised painterly"
    subj = []
    if "blue" in present:
        subj.append(dict(color="blue", desc="the player's character: a small armoured adventurer with a sword, seen from a high 45-degree game camera",
                         many=False, motion="moving and attacking as recorded"))
    if "red" in present:
        subj.append(dict(color="red", desc="a large boss monster fighting the player, much bigger than the player, swinging a heavy weapon",
                         many=False, motion="moving and attacking as recorded"))
    if "yellow" in present:
        subj.append(dict(color="yellow", desc="a short burst of glowing magical energy, sparks and shockwave from the boss's attack",
                         many=True, motion="appearing and fading as recorded"))
    return dict(shape="cylinder",
                setting=(f"a {look} 2.5D action role-playing game arena in a {sc}, seen from a fixed high camera at about "
                         f"45 degrees: open walkable ground with paved paths, buildings and walls, rocks, trees and plants, "
                         f"crates and barrels, under consistent soft daylight"),
                subjects=subj, sound="footsteps, weapon swings and impacts, ambient wind")


def main():
    a = argparse.ArgumentParser(); a.add_argument("--testset", required=True); a.add_argument("--out", required=True)
    args = a.parse_args()
    man = json.load(open(f"{args.testset}/manifest.json"))
    rows = {json.loads(l)["clip_id"]: json.loads(l) for l in open(DATA + "/index/clips.jsonl")}
    tars = {}; jobs = []
    for j, m in enumerate(man["clips"]):
        cid = m["clip_id"]; r = rows[cid]; od = f"{args.out}/{cid}"; os.makedirs(od, exist_ok=True)
        ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"]
        ext = np.load(f"{DATA}/{m['ext_source']}/{cid}.npz")["ids"][:28]
        ids = np.concatenate([ids, ext], 0)
        run_dir = f"{RAW}/{r['subset']}/{r['scene']}/{r['run']}"
        t = tars.setdefault(run_dir, PV.RunTars(run_dir))
        present = set()
        frames16 = []
        for k in range(105):
            wb = cv2.imdecode(np.frombuffer(t.get(int(r["start"]) + k, "whitebox.jpg"), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            img = GEO.resample_rgb(wb).copy()
            for name, cls in (("blue", PLAYER), ("red", BOSS), ("yellow", FX)):
                msk = np.isin(ids[k], cls)
                if msk.any():
                    img[msk] = COL[name]; present.add(name)
            frames16.append(img)
        idx = [int(i * 16 // 24) for i in range(158)]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
                              "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", f"{od}/primitive.mp4"],
                             stdin=subprocess.PIPE)
        for i in idx:
            p.stdin.write(np.ascontiguousarray(to_canvas(frames16[i], cv2.INTER_AREA)).tobytes())
        p.stdin.close(); assert p.wait() == 0
        json.dump(spec_for(r, present), open(f"{od}/spec.json", "w"), indent=1)
        jobs.append(dict(name=cid, colors=sorted(present)))
        if len(tars) > 4:
            for tt in tars.values():
                for fh in tt.fh.values(): fh.close()
            tars = {}
        print(j, cid, sorted(present), flush=True)
    json.dump(dict(clips=jobs, frames=158, fps=24, proxy_colors=COL, rule="whitebox + flat proxy colours from ids"),
              open(f"{args.out}/manifest.json", "w"), indent=1)
    print("DONE", len(jobs))


if __name__ == "__main__":
    main()
