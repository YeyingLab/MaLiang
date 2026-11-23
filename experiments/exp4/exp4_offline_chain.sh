#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Experiment 4 interface comparison, variants 3 -> 2 -> 1, each as soon as its checkpoint and its 100 base generations exist.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE=$C/hf_cache
while pgrep -u $(id -un) -f '[e]xp4_offline.py' >/dev/null; do sleep 120; done   # a variant already in flight finishes first
for spec in v3_ids:$RT/outputs/sft_fxcond_v1/step-5000.safetensors \
            v2_color_light:$C/abl/runs/color_light/step-5000.safetensors v1_color_vae:$C/abl/runs/color_vae/step-5000.safetensors; do
  v=${spec%%:*}; ck=${spec#*:}
  until [ -f $ck ] && [ $(ls $C/main/abl/$v/*_gen.mp4 2>/dev/null | wc -l) -ge 100 ]; do sleep 600; done
  for t in 1 2 3; do
    cd $RT && CUDA_VISIBLE_DEVICES=$(pick 30) .venv/bin/python -B $C/exp4/exp4_offline.py --ckpt $ck --variant $v \
        --out $C/exp4/offline_$v >> $C/exp4/offline_$v.log 2>&1 && break; sleep 120
  done
done
echo EXP4_OFFLINE_DONE >> $C/exp4/offline_v1_color_vae.log
