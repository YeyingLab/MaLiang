#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# paper interface ablation: MODE=color_vae|color_light GPUS=a,b[,c] bash run_abl.sh
# Same recipe as sft_fxcond_v1 (variant 3): from Wan2.2 base, gc25d_v10_fxcond, vocab parts_fx, global batch 12,
# lr 1e-5 / cond_lr 1e-4, warmup 500, seed 0; 5000 steps; weights at 2500 and 5000; no optimizer state (disk).
set -euo pipefail
W=$MALIANG_ROOT; A=$W/experiments/abl; RT=$MALIANG_ROOT
MODE="${MODE:?}"; GPUS="${GPUS:?}"; PORT="${PORT:-29661}"
n=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
case "$n" in 2) GA=6 ;; 3) GA=4 ;; 4) GA=3 ;; *) echo "ABORT: unsupported GPU count"; exit 1 ;; esac
bad=$(nvidia-smi -i "$GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', *' '($3-$2)/1024<60{printf "GPU%s ",$1}')
[ -z "$bad" ] || { echo "ABORT: <60 GB free on $bad"; exit 1; }
OUT=$A/runs/$MODE; mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=8
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio TMPDIR=$W/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
exec $RT/.venv/bin/accelerate launch --config_file $W/configs/accelerate_zero2_${n}gpu_ga${GA}.yaml --main_process_port $PORT \
  $A/code/gc25d/train.py --cond_mode $MODE --vocab parts_fx --data_root $RT/data/gc25d_v10_fxcond \
  --model_dir $RT/models/Wan2.2-TI2V-5B --lr 1e-5 --cond_lr 1e-4 --warmup_steps 500 --grad_clip 1.0 --max_steps 5000 \
  --batch_size 1 --grad_accum $GA --num_workers 4 --save_every 2500 --no-save_state --val_every 500 --val_n 16 \
  --seed 0 --output "$OUT"
