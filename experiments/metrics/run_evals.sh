#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Sequential common evals (one at a time: they share the GT cache). Usage: GPU=4 bash run_evals.sh name:dir:kind[:proxy] ...
# proxy (for the unrendered rate) defaults from the name: solar* -> solar, agr* -> agr, v1_*/v2_* -> abl, else none.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT
export HF_HUB_CACHE=$C/hf_cache AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope CUDA_VISIBLE_DEVICES=${GPU:-4}
cd $C/metrics
for spec in "$@"; do IFS=: read name dir kind proxy <<< "$spec"
  if [ -z "$proxy" ]; then case $name in solar*) proxy=solar;; agr*) proxy=agr;; v1_*|v2_*) proxy=abl;; *) proxy=none;; esac; fi
  while pgrep -u $(id -un) -f '[e]val_common.py' >/dev/null; do sleep 60; done
  $RT/.venv/bin/python -u eval_common.py --method $name --dir $dir --kind $kind --proxy $proxy --out $C/main/eval > $C/main/eval_$name.log 2>&1
  echo "$name rc=$? $(date '+%F %T')" >> $C/main/eval_queue.log
done
