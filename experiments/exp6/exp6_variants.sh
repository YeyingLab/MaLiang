#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Experiment 6 for variants 1 and 2: same cases (unseen procmap maps + sketch maps), same FLUX first frame / ids /
# prompts / seed as variant 3, generated with abl_sample_user.py into <case>/gen_<variant>/. Waits for each checkpoint
# and for the variant-3 pass (first frames) to exist; scores with exp6_score.py --gen_sub.
C=$MALIANG_ROOT/experiments; E=$C/exp6; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope DIFFSYNTH_SKIP_DOWNLOAD=True \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE=$C/hf_cache
until grep -q EXP6_ALL_DONE $E/score.log 2>/dev/null; do sleep 600; done
for spec in v2_color_light:$C/abl/runs/color_light/step-5000.safetensors v1_color_vae:$C/abl/runs/color_vae/step-5000.safetensors; do
  v=${spec%%:*}; ck=${spec#*:}
  until [ -f $ck ]; do sleep 600; done
  for d in $E/cases/*/ $E/sketch/cases/*/; do
    case=$(basename $d)
    [ -f $d/first.png ] || continue
    [ -f $d/gen_$v/${case}_gen.mp4 ] && continue
    cd $RT && CUDA_VISIBLE_DEVICES=$(pick 30) .venv/bin/python -B $C/abl/code/gc25d/abl_sample_user.py --ckpt $ck \
      --model_dir $RT/models/Wan2.2-TI2V-5B --data_root $RT/data/gc25d_v10_fxcond --first_frame $d/first.png --ids $d/ids.npz \
      --prompts $d/prompts.json --out $d/gen_$v --name $case --steps 50 >> $E/variants_$v.log 2>&1 || echo "FAIL $case" >> $E/variants_$v.log
  done
  $RT/.venv/bin/python $E/exp6_score.py --cases $E/cases $E/sketch/cases --gen_sub gen_$v --out $E/score_$v.json > $E/score_$v.log 2>&1
done
