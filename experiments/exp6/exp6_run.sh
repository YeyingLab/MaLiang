#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Experiment 6 (unseen maps): 10 test clips (one per run) x 4 procmap themes = 40 numeric worlds that were never recorded.
#   1. procmap.py   -> <case>/ids.npz (81 f, parts ids), prompts.json, map0.png  (source clip's real camera + characters)
#   2. FLUX.2 klein-4B map2frame (best_of 4) -> <case>/first.png               (first frame drawn from the map only)
#   3. variant (3) sft_fxcond_v1 step-5000 via sample_user.py -> <case>/gen/<case>_gen.mp4
# Each stage skips finished cases, so the script can be re-run after a crash. Scoring: exp6_score.py.
# Usage: [GPU=<index>] bash exp6_run.sh      (default: a GPU from experiments/gpu_pick.sh with >= 30 GB free)
set -uo pipefail
RT=$MALIANG_ROOT; C=$MALIANG_ROOT/experiments; E=$C/exp6; source $C/gpu_pick.sh; GPU="${GPU:-$(pick 30)}"
OUT=${OUT:-$E/cases}; mkdir -p $OUT          # OUT=$E/sketch/cases SKIP_PROCMAP=1 for the hand-drawn maps
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
CK=$RT/outputs/sft_fxcond_v1/step-5000.safetensors
CLIPS=$(python3 -c "import json;print(' '.join(json.load(open('$C/testset/v1/test_clips.json'))[:10]))")
cd $RT
[ -n "${SKIP_PROCMAP:-}" ] && CLIPS=""
for c in $CLIPS; do for th in village forest plaza canyon; do
  case=${c}__$th
  [ -f $OUT/$case/ids.npz ] || .venv/bin/python -B $E/run_procmap.py --clip $c --theme $th --seed 0 --case $case --out $OUT --no_video \
      >> $E/procmap.log 2>&1 || echo "procmap FAIL $case" >> $E/procmap.log
done; done
python3 - "$OUT" > $E/flux_jobs.json <<'EOF'
import json, os, sys
out = sys.argv[1]
print(json.dumps([dict(mode="map2frame", map=f"{out}/{c}/ids.npz", out=f"{out}/{c}/first.png", best_of=4, save_map=True)
                  for c in sorted(os.listdir(out)) if os.path.exists(f"{out}/{c}/ids.npz") and not os.path.exists(f"{out}/{c}/first.png")]))
EOF
[ "$(cat $E/flux_jobs.json)" = "[]" ] || $C/envs/flux/bin/python $E/firstframe_flux_local.py batch --model klein4b --jobs $E/flux_jobs.json >> $E/flux.log 2>&1
for d in $OUT/*/; do
  case=$(basename $d)
  [ -f $d/first.png ] || continue
  [ -f $d/gen/${case}_gen.mp4 ] && continue
  .venv/bin/python -B code/gc25d/sample_user.py --ckpt $CK --first_frame $d/first.png --ids $d/ids.npz --prompts $d/prompts.json \
      --out $d/gen --name $case --steps 50 >> $E/sample.log 2>&1 || echo "sample FAIL $case" >> $E/sample.log
done
echo EXP6_GEN_DONE >> $E/sample.log
