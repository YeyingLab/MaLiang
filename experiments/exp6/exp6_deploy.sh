#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Experiment 6, deploy model on all 60 unseen-map cases, then score (gen_ours, no proxy).
C=$MALIANG_ROOT/experiments; E=$C/exp6; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE=$C/hf_cache
cd $RT && CUDA_VISIBLE_DEVICES=$(pick 30) .venv/bin/python -B $E/exp6_deploy.py --cases $E/cases $E/sketch/cases > $E/deploy.log 2>&1
CUDA_VISIBLE_DEVICES=$(pick 10) .venv/bin/python $E/exp6_score.py --cases $E/cases $E/sketch/cases --gen_sub gen_ours --proxy none \
  --out $E/score_ours.json > $E/score_ours.log 2>&1
echo EXP6_OURS_DONE >> $E/score_ours.log
