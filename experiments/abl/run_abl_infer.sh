#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Table 2 generations on the 100-clip test set (81 frames, 50 steps, seed 0) with sample.py clip mode (abl code copy,
# patched to also write <clip>_gen.mp4).
# Guarded for shared GPUs on node 12: GPU from the shared rule (experiments/gpu_pick.sh) with >= FREE_GB free, skips clips
# already written, retries.
# Usage: [SAMPLER=abl_sample.py for variants 1/2] NAME=v3_ids CK=<ckpt> [SEM_MODE=cond|unknown] [CODE=<gc25d dir>] [FREE_GB=60] bash run_abl_infer.sh
set -uo pipefail
RT=$MALIANG_ROOT; C=$MALIANG_ROOT/experiments
NAME="${NAME:?}"; CK="${CK:?}"; SEM_MODE="${SEM_MODE:-cond}"; CODE="${CODE:-$C/abl/code/gc25d}"
FREE_GB="${FREE_GB:-60}"; source $C/gpu_pick.sh; MAX_TRY="${MAX_TRY:-20}"
O=$C/main/abl/$NAME; LOG=$C/main/abl/$NAME.log; mkdir -p $O
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $RT
for t in $(seq 1 $MAX_TRY); do
  python3 - "$C/testset/v1/test_clips.json" "$O" > $O/todo.txt <<'EOF'
import json, os, sys
for c in json.load(open(sys.argv[1])):
    if not os.path.exists(f"{sys.argv[2]}/{c}_gen.mp4"):
        print(c)
EOF
  n=$(grep -c . $O/todo.txt); [ "$n" = "0" ] && { echo "[$(date '+%F %T')] ALL_DONE" >> $LOG; exit 0; }
  gpu=$(pick $FREE_GB)
  echo "[$(date '+%F %T')] try $t on GPU$gpu, $n clips, sem_mode $SEM_MODE" >> $LOG
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -B $CODE/${SAMPLER:-sample.py} --ckpt $CK --data_root $RT/data/gc25d_v10_fxcond --raw_root ${MALIANG_RAW:-$MALIANG_ROOT/data/recordings/data} --model_dir $RT/models/Wan2.2-TI2V-5B \
      --clip_ids @$O/todo.txt --out_dir $O --sem_mode $SEM_MODE --steps 50 --seed 0 --no_recon_ceiling >> $LOG 2>&1
  echo "[$(date '+%F %T')] exit $?" >> $LOG; sleep 60
done
