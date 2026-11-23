#!/bin/bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# FX control test (natural FX layouts), deploy model, 3 shards. Shard 2 starts once ablation (1) step-5000 frees GPUs 1/6/7.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
CK=$MALIANG_ROOT/models/fxcond_v1_stage3_ema_step500.safetensors
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
one() { local s=$1 g=$2; cd $RT && for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g .venv/bin/python -B $C/exp4/exp_fx_natural.py --ckpt $CK --out $C/exp6/fx_natural_ours --shard $s --nshard 3 >> $C/exp6/fx_natural_ours.s$s.log 2>&1 && break; sleep 60; g=$(pick 35); done; }
one 0 4 &
one 1 5 &
( until [ -f $C/abl/runs/color_vae/step-5000.safetensors ]; do sleep 120; done; sleep 300; one 2 $(pick 35) ) &
wait; echo FX_NATURAL_DONE >> $C/exp6/fx_natural_ours.s0.log
