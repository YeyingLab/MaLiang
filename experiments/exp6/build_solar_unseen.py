#!/usr/bin/env python3
"""SolarRenderer inputs for experiment 6 (unseen procmap maps + sketch maps), built exactly like the test-set inputs
(build_testset.py: CARLA palette, 16 -> 24 fps by repetition, 832x480 -> 1344x776 nearest, centre crop 1344x768,
six-part prompt from the source clip's scene / look / per-cell actions):
  structure.mp4  the case's 81 id frames; SolarRenderer needs >= 158 output frames (105 source frames), so the last id
                 frame is held for source frames 81..108 - only source frames 0..80 are ever scored;
  first.png      the case's FLUX first frame (the same image variant (3) starts from), same resize / crop.
Usage: build_solar_unseen.py --cases DIR [DIR ...] --out DIR --conditions_dir <SolarRenderer caption dir>"""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import argparse, glob, json, os, subprocess, sys
HERE = _mos.path.join(MALIANG_ROOT, "experiments", "testset")
sys.path.insert(0, HERE)
import build_testset as B
np, cv2, torch = B.np, B.cv2, None


def main():
    a = argparse.ArgumentParser(); a.add_argument("--cases", nargs="+", required=True); a.add_argument("--out", required=True)
    a.add_argument("--conditions_dir", required=True)
    args = a.parse_args(); os.environ["SOLAR_CAPTION_DIR"] = args.conditions_dir
    import torch as T
    rows = {json.loads(l)["clip_id"]: json.loads(l) for l in open(B.DATA + "/index/clips.jsonl")}
    n = 0
    for root in args.cases:
        for d in sorted(glob.glob(f"{root}/*/")):
            case = os.path.basename(d.rstrip("/")); od = f"{args.out}/{case}"
            if not (os.path.exists(f"{d}/ids.npz") and os.path.exists(f"{d}/first.png")) or os.path.exists(f"{od}/prompt.txt"):
                continue
            src = json.load(open(f"{d}/source.json")); cid = src["clip_id"]; r = rows[cid]
            ids = np.load(f"{d}/ids.npz")["ids"]; assert ids.shape[0] == 81, ids.shape
            ids109 = np.concatenate([ids, np.repeat(ids[-1:], 28, 0)], 0)
            os.makedirs(od, exist_ok=True)
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                  "-s", f"{B.OUT_W}x{B.OUT_H}", "-r", "24", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                  "-crf", "0", "-pix_fmt", "yuv444p", f"{od}/structure.mp4"], stdin=subprocess.PIPE)
            for i in range(160):
                p.stdin.write(np.ascontiguousarray(B.to_canvas(B.PAL[ids109[int(i * 16 // 24)]], cv2.INTER_NEAREST)).tobytes())
            p.stdin.close(); assert p.wait() == 0
            f0 = cv2.imread(f"{d}/first.png")
            assert f0.shape[:2] == (480, 832), f0.shape
            cv2.imwrite(f"{od}/first.png", B.to_canvas(f0, cv2.INTER_AREA)); cv2.imwrite(f"{od}/first_832x480.png", f0)
            prompts = json.load(open(f"{d}/prompts.json"))
            open(f"{od}/prompt.txt", "w").write(B.assemble(B.caption(r, prompts)))
            json.dump(dict(case=case, source_clip=cid, held_tail_frames=28), open(f"{od}/case.json", "w")); n += 1
            print(case, flush=True)
    print("DONE", n)


if __name__ == "__main__":
    main()
