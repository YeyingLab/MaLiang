#!/bin/bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Ablation significance: wait for extra_eval on the 4 variants, then paired bootstrap (reference = v3_ids) by clip and by run.
C=$MALIANG_ROOT/experiments; R=$MALIANG_ROOT; cd $C/metrics
until [ $(grep -l "^DONE" logs_extra/v1_color_vae.log logs_extra/v2_color_light.log logs_extra/v3_ids.log logs_extra/v4_none.log 2>/dev/null | wc -l) -eq 4 ]; do
  grep -l Traceback logs_extra/v[1-4]_*.log >/dev/null 2>&1 && { echo "ABORT: extra_eval failed" >> logs_extra/abl_ci.log; exit 1; }; sleep 60; done
for cl in clip run; do CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $R/.venv/bin/python -u bootstrap_ci.py --methods v3_ids,v1_color_vae,v2_color_light,v4_none --tag abl --cluster $cl >> logs_extra/abl_ci.log 2>&1; done
echo ABL_CI_DONE >> logs_extra/abl_ci.log
