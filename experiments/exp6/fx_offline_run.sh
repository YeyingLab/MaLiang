#!/bin/bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# FX control, interface comparison (variants 1/2/3, 50-step teachers), 2 shards per variant, one fixed GPU per shard
# (only one worker on GPU 0, which also hosts a demo). Per-clip results are resumed.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE=$C/hf_cache
one() { local v=$1 ck=$2 s=$3 g=$4; for t in 1 2 3; do echo "[$(date "+%F %T")] $v shard $s try $t GPU $g" >> $C/exp6/fx_offline_$v.s$s.log
  cd $C/abl/code/gc25d && CUDA_VISIBLE_DEVICES=$g $RT/.venv/bin/python -B $C/exp4/exp_fx_offline.py --ckpt $ck --variant $v \
     --out $C/exp6/fx_offline_$v --shard $s --nshard 2 >> $C/exp6/fx_offline_$v.s$s.log 2>&1 && break; sleep 90; g=$(pick 40); done; }
V3=$RT/outputs/sft_fxcond_v1/step-5000.safetensors; V2=$C/abl/runs/color_light/step-5000.safetensors; V1=$C/abl/runs/color_vae/step-5000.safetensors
one v3_ids $V3 0 4 & one v3_ids $V3 1 5 & one v2_color_light $V2 0 1 & one v2_color_light $V2 1 6 & one v1_color_vae $V1 0 7 & one v1_color_vae $V1 1 0 &
wait; echo FX_OFFLINE_DONE >> $C/exp6/fx_offline_v1_color_vae.s0.log
