#!/bin/bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# After eval_students.sh: common metrics + extras for both students, paired bootstrap (ref = stu_v3_ids), FX CI.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT; L=$C/abl/cmp/score.log
until [ -f $C/abl/cmp/EVAL_DONE ]; do sleep 60; done
echo "[$(date +%T)] eval done, scoring" >> $L
cd $C/metrics
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for spec in stu_v3_ids:none stu_v2_color_light:abl; do n=${spec%%:*}; p=${spec#*:}
  CUDA_VISIBLE_DEVICES=4 $RT/.venv/bin/python -u eval_common.py --method $n --dir $C/main/$n --kind native --proxy $p --out $C/main/eval > $C/main/eval_$n.log 2>&1 &
  CUDA_VISIBLE_DEVICES=5 $RT/.venv/bin/python -u extra_eval.py --method $n --dir $C/main/$n --kind native --proxy $p > logs_extra/$n.log 2>&1 &
  wait; echo "[$(date +%T)] scored $n" >> $L
done
for cl in clip run; do CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $RT/.venv/bin/python -u bootstrap_ci.py --methods stu_v3_ids,stu_v2_color_light --tag stu --cluster $cl >> $L 2>&1; done
python3 $C/exp6/fx_stu_ci.py > $C/exp6/fx_stu_ci.out 2>&1
echo SCORE_DONE >> $L
