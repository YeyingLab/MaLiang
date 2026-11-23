#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Re-run experiment 6 for one interface variant (cases whose gen_<v>/<case>_gen.mp4 is missing), up to 3 passes, then score.
# Usage: V=v2_color_light CK=<step-5000> bash exp6_variant_redo.sh
C=$MALIANG_ROOT/experiments; E=$C/exp6; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE=$C/hf_cache
v=${V:?}; ck=${CK:?}; L=$E/variants_$v.redo.log
for pass in 1 2 3; do
  for d in $E/cases/*/ $E/sketch/cases/*/; do
    case=$(basename $d)
    [ -f $d/first.png ] || continue
    [ -f $d/gen_$v/${case}_gen.mp4 ] && continue
    cd $RT && CUDA_VISIBLE_DEVICES=$(pick 30) .venv/bin/python -B $C/abl/code/gc25d/abl_sample_user.py --ckpt $ck \
      --model_dir $RT/models/Wan2.2-TI2V-5B --data_root $RT/data/gc25d_v10_fxcond --first_frame $d/first.png --ids $d/ids.npz \
      --prompts $d/prompts.json --out $d/gen_$v --name $case --steps 50 >> $L 2>&1 || { echo "FAIL $case pass $pass" >> $L; sleep 60; }
  done
done
echo "[$(date '+%F %T')] generated $(ls $E/cases/*/gen_$v/*_gen.mp4 $E/sketch/cases/*/gen_$v/*_gen.mp4 2>/dev/null | wc -l)" >> $L
CUDA_VISIBLE_DEVICES=$(pick 10) $RT/.venv/bin/python $E/exp6_score.py --cases $E/cases $E/sketch/cases --gen_sub gen_$v --out $E/score_$v.json > $E/score_$v.log 2>&1
echo "EXP6_${v}_DONE" >> $L
