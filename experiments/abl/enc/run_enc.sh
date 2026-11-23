#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# paper encoder-design ablation (same recipe as run_abl.sh / variant 3, integer ids):
#   NAME=nearest POOL=nearest MASKP=0.5  -> one id per 4x4 block instead of class fractions
#   NAME=nomask  POOL=frac    MASKP=0    -> class fractions, no random region masking
# Usage: NAME=.. POOL=.. MASKP=.. GPUS=a,b PORT=.. bash run_enc.sh
set -euo pipefail
W=$MALIANG_ROOT; A=$W/experiments/abl; E=$A/enc; RT=$MALIANG_ROOT
NAME="${NAME:?}"; POOL="${POOL:?}"; MASKP="${MASKP:?}"; GPUS="${GPUS:?}"; PORT="${PORT:-29671}"
n=$(echo "$GPUS" | tr "," "\n" | grep -c .)
case "$n" in 2) GA=6 ;; 3) GA=4 ;; 4) GA=3 ;; *) echo "ABORT: unsupported GPU count"; exit 1 ;; esac
bad=$(nvidia-smi -i "$GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F", *" "(\$3-\$2)/1024<60{printf \"GPU%s \",\$1}")
[ -z "$bad" ] || { echo "ABORT: <60 GB free on $bad"; exit 1; }
OUT=$E/runs/$NAME; mkdir -p "$OUT"
export GC_POOL_MODE=$POOL CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=8
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio TMPDIR=$W/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
exec $RT/.venv/bin/accelerate launch --config_file $W/configs/accelerate_zero2_${n}gpu_ga${GA}.yaml --main_process_port $PORT \
  $E/code/gc25d/train.py --cond_mode ids --vocab parts_fx --data_root $RT/data/gc25d_v10_fxcond --mask_p $MASKP \
  --model_dir $RT/models/Wan2.2-TI2V-5B --lr 1e-5 --cond_lr 1e-4 --warmup_steps 500 --grad_clip 1.0 --max_steps 5000 \
  --batch_size 1 --grad_accum $GA --num_workers 4 --save_every 2500 --no-save_state --val_every 500 --val_n 16 \
  --seed 0 --output "$OUT"
