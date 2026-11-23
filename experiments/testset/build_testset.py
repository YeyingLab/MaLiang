#!/usr/bin/env python3
"""paper test set (shared by experiments B/C/D) + SolarRenderer inputs. CPU only.

Test set: N val clips of gc25d_v10_fxcond (the data of the fxcond_v1 model used in the paper), has_ext and not drop,
round-robin over val runs (stratified). Each clip has 109 frames of ids (81 cond + 28 extension; ext_look_change
clips use cond_ext_canvas, the canvas the model's stage 3 reads).
SolarRenderer inputs per clip (<out>/solar/<clip>/):
  structure.mp4  the 109 id frames coloured with a CARLA-style class palette (its trained `semantic` condition),
                 16 -> 24 fps by frame repetition (160 frames >= the required 158), 832x480 -> 1344x776 nearest,
                 centre crop to 1344x768, lossless H.264 (yuv444p, crf 0)
  first.png      GT frame 0 (832x480, as the model sees it) with the same resize / crop
  prompt.txt     six-part prompt: conditions.DEFINE["semantic"] + a templated description built ONLY from information
                 MaLiang also gets (scene name, look, per-0.25 s action text) - no caption of the GT video
manifest.json lists clips, transform, palette and the class mapping.
Usage: build_testset.py --out <dir> --n 100
"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, json, os, subprocess, sys, collections
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import numpy as np, cv2
import geometry as GEO
import prep_v10 as PV
DATA = RT + "/data/gc25d_v10_fxcond"
RAW = _mos.environ.get("MALIANG_RAW", _mos.path.join(MALIANG_ROOT, "data", "recordings", "data"))
CARLA = {"unlabeled": (0, 0, 0), "road": (128, 64, 128), "building": (70, 70, 70), "wall": (102, 102, 156),
         "vegetation": (107, 142, 35), "terrain": (152, 251, 152), "sky": (70, 130, 180), "pedestrian": (220, 20, 60),
         "rider": (255, 0, 0), "static": (110, 190, 160), "dynamic": (170, 120, 50)}
CLASS_OF = {0: "sky", 1: "terrain", 2: "road", 3: "building", 4: "wall", 5: "vegetation", 6: "static",
            10: "pedestrian", 11: "rider", 255: "unlabeled"}
for k in range(12, 17): CLASS_OF[k] = "pedestrian"          # player parts
for k in range(17, 22): CLASS_OF[k] = "rider"               # boss parts
for k in range(22, 26): CLASS_OF[k] = "dynamic"             # boss attack effects
PAL = np.zeros((256, 3), np.uint8)
for k, c in CLASS_OF.items(): PAL[k] = CARLA[c]
OUT_W, OUT_H, PAD_H = 1344, 768, 776


def to_canvas(img, interp):
    x = cv2.resize(img, (OUT_W, PAD_H), interpolation=interp)
    o = (PAD_H - OUT_H) // 2
    return x[o:o + OUT_H]


def pick(rows, n):
    by = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["run"], r["start"])):
        by[r["run"]].append(r)
    runs = sorted(by, key=lambda k: (by[k][0]["subset"], k))
    out, i = [], 0
    while len(out) < n and any(by.values()):
        r = runs[i % len(runs)]
        if by[r]:
            lst = by[r]; out.append(lst.pop(len(lst) // 2 if len(out) % 2 else 0))   # spread starts within a run
        i += 1
    return out


def scene_words(scene):
    return " ".join(scene.split("_")[1:]).lower()


def look_words(look):
    parts = [p for p in (look or "").split("|") if p and p not in ("no", "none", "orig", "full", "pbr", "photo")]
    return ", ".join(p.replace("_", " ") for p in parts) or "natural"


def caption(r, prompts):
    sc, lk = scene_words(r["scene"]), look_words(r.get("look_name"))
    style = "a stylised, painterly game look" if r["subset"] == "v9_stylized" else "a realistic, physically based game look"
    acts, prev = [], None
    for p in prompts:
        p = p.strip().rstrip(".")
        for pre in ("In a 2.5D isometric view, ", "In a 2.5D isometric view "):
            if p.startswith(pre):
                p = p[len(pre):]
        if p != prev:
            acts.append(p)
        prev = p
    events = " Then ".join(a[0].upper() + a[1:] + "." for a in acts[:8])
    boss = r.get("boss_visible_frac", 0) > 0.05
    d = (f"The target video is gameplay footage from a 2.5D action role-playing game, seen from a fixed high camera that looks "
         f"down at about forty-five degrees and follows the player, with {style} and a {lk} mood. "
         f"[Shot 1] The scene is a {sc}: open walkable ground with paved paths, surrounding buildings and walls, rocks, "
         f"trees and plants, and scattered props such as crates, barrels and furniture, all keeping their place and "
         f"outline from frame to frame while the camera glides smoothly with the player. "
         f"The player is a small armoured adventurer carrying a weapon, always near the middle of the frame. ")
    if boss:
        d += ("A much larger boss creature shares the arena with the player, moving and striking with heavy, deliberate "
              "motions; when it attacks, short bright flashes of magical energy, sparks and shockwaves burst around its "
              "weapon and on the ground where the blows land, then fade within a fraction of a second. ")
    d += (f"Over the clip the action unfolds as follows. {events} "
          "Characters move with natural weight, their feet stay planted on the ground and their shadows follow them. "
          "Materials read clearly: stone and plaster on the buildings, packed earth, grass and dust on the ground, worn "
          "wood and metal on the props, leaves that sway slightly in a light breeze. Lighting is consistent across the "
          "whole clip, with soft shadows that fall in one direction and gentle ambient light filling the darker corners. "
          "Small details such as pebbles, cracks in the paving, fallen leaves and patches of moss add texture without "
          "cluttering the view, and distant parts of the scene stay calm and slightly softer than the area around the "
          "player. The camera never cuts; it keeps the same height and angle and moves only to keep the player in view, "
          "so the scene slides past steadily as the player travels. The overall impression is a polished, readable "
          "action game in which every surface, character and effect sits naturally in the same lit world, with crisp "
          "silhouettes for the characters, clear separation between walkable ground and obstacles, and smooth, "
          "continuous motion from the first frame to the last.")
    words = d.split()
    if len(words) > 495:
        d = " ".join(words[:495]).rstrip(",;") + "."
    filler = (" The palette stays coherent throughout, edges remain clean, and nothing pops in or out of existence "
              "between frames.")
    while len(d.split()) < 360:
        d += filler
    summ = (f"A 2.5D action role-playing game clip in a {sc}: the player "
            + ("fights a large boss" if boss else "explores the area") + " while the camera follows from a fixed high angle.")
    return dict(summary=summ, detailed_description=d,
                overall_soundscape="Footsteps, weapon swings and impacts, ambient wind and distant birds.",
                non_diegetic_music="N/A")


def assemble(free):
    sys.path.insert(0, os.environ.get("SOLAR_CAPTION_DIR", ""))
    import conditions as CD
    return CD.assemble(free, "full", "semantic")


def main():
    a = argparse.ArgumentParser(); a.add_argument("--out", required=True); a.add_argument("--n", type=int, default=100)
    a.add_argument("--conditions_dir", required=True)
    args = a.parse_args()
    os.environ["SOLAR_CAPTION_DIR"] = args.conditions_dir
    rows = [json.loads(l) for l in open(DATA + "/index/clips.jsonl")]
    val = [r for r in rows if r["split"] == "val" and not r.get("drop") and r.get("has_ext")]
    clips = pick(val, args.n)
    os.makedirs(args.out + "/solar", exist_ok=True)
    import torch
    man = []
    tars = {}
    for j, r in enumerate(clips):
        cid = r["clip_id"]; od = f"{args.out}/solar/{cid}"; os.makedirs(od, exist_ok=True)
        ids = np.load(f"{DATA}/cond/{cid}.npz")["ids"]
        sub = "cond_ext_canvas" if r.get("ext_look_change") else "cond_ext"
        ext = np.load(f"{DATA}/{sub}/{cid}.npz")["ids"][:28]
        ids109 = np.concatenate([ids, ext], 0)
        assert ids109.shape == (109, 480, 832), ids109.shape
        idx = [int(i * 16 // 24) for i in range(160)]
        frames = [to_canvas(PAL[ids109[i]], cv2.INTER_NEAREST) for i in idx]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{OUT_W}x{OUT_H}", "-r", "24", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                              "-crf", "0", "-pix_fmt", "yuv444p", f"{od}/structure.mp4"], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close(); assert p.wait() == 0
        run_dir = f"{RAW}/{r['subset']}/{r['scene']}/{r['run']}"
        t = tars.setdefault(run_dir, PV.RunTars(run_dir))
        src = cv2.imdecode(np.frombuffer(t.get(int(r["start"]), "jpg"), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        f0 = GEO.resample_rgb(src)
        cv2.imwrite(f"{od}/first.png", to_canvas(f0, cv2.INTER_AREA)[..., ::-1])
        cv2.imwrite(f"{od}/first_832x480.png", f0[..., ::-1])
        lat = torch.load(f"{DATA}/latents/{cid}.pt", map_location="cpu", weights_only=False)
        prompts = list(lat["prompts"])
        ext_p = np.load(f"{DATA}/{sub}/{cid}.npz")
        free = caption(r, prompts)
        txt = assemble(free)
        open(f"{od}/prompt.txt", "w").write(txt)
        man.append(dict(clip_id=cid, subset=r["subset"], scene=r["scene"], run=r["run"], start=r["start"],
                        ext_source=sub, words=len(free["detailed_description"].split())))
        if len(tars) > 4:
            for tt in tars.values():
                for fh in tt.fh.values(): fh.close()
            tars = {}
        print(j, cid, man[-1]["words"], flush=True)
    json.dump(dict(clips=man, n=len(man), data=DATA, fps_in=16, fps_out=24, frames_in=109, frames_out=160,
                   frame_index_rule="out i -> in floor(i*16/24)", resize=f"832x480 -> {OUT_W}x{PAD_H} then centre crop {OUT_W}x{OUT_H}",
                   palette={str(k): list(map(int, PAL[k])) for k in CLASS_OF}, class_map=CLASS_OF,
                   caption_rule="template from scene name, look_name and per-cell action prompts; no GT caption"),
              open(f"{args.out}/manifest.json", "w"), indent=1)
    json.dump([m["clip_id"] for m in man], open(f"{args.out}/test_clips.json", "w"), indent=1)
    print("DONE", len(man), collections.Counter(m["subset"] for m in man), collections.Counter(m["run"] for m in man))


if __name__ == "__main__":
    main()
