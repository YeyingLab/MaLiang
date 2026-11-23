#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# After both runs are scored: paired bootstrap vs v3_ids (clip and recording level) and the FX CI.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT; E=$C/abl/enc
until [ -f $E/SCORED_nearest ] && [ -f $E/SCORED_nomask ]; do sleep 120; done
cd $C/metrics
for cl in clip run; do CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $RT/.venv/bin/python -u bootstrap_ci.py --methods v3_ids,enc_nearest,enc_nomask --tag enc --cluster $cl >> $E/ci.log 2>&1; done
$RT/.venv/bin/python $E/fx_enc_ci.py > $E/fx_enc_ci.out 2>&1
echo CI_DONE >> $E/ci.log
