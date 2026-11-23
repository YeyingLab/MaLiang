#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# FX new-category chain: segdata_fx done -> train 10-class segmenter (seg_b2_fx) -> FX injection test with the deploy model.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT; source $C/gpu_pick.sh
export HF_HUB_CACHE=$C/hf_cache AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope \
       DIFFSYNTH_SKIP_DOWNLOAD=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q '^DONE' $C/metrics/seg_data_fx.log 2>/dev/null; do grep -q Traceback $C/metrics/seg_data_fx.log && exit 1; sleep 120; done
cd $C/metrics
while pgrep -u $(id -un) -f '[s]eg_train.py --data segdata_fx' >/dev/null; do sleep 120; done   # run already in flight
until grep -q '^DONE' seg_train_fx.log 2>/dev/null; do
  rm -rf seg_b2_fx; CUDA_VISIBLE_DEVICES=$(pick 25) $RT/.venv/bin/python -u seg_train.py --data segdata_fx --out seg_b2_fx > seg_train_fx.log 2>&1
  grep -q '^DONE' seg_train_fx.log || { mv seg_train_fx.log seg_train_fx.fail.$(date +%s).log; sleep 300; }
done
CK=$MALIANG_ROOT/models/fxcond_v1_stage3_ema_step500.safetensors
cd $RT && CUDA_VISIBLE_DEVICES=$(pick 30) .venv/bin/python -B $C/exp4/exp_fx_inject.py --ckpt $CK --out $C/exp6/fx_inject_ours --n 30 > $C/exp6/fx_inject_ours.log 2>&1
echo FX_CHAIN_DONE >> $C/exp6/fx_inject_ours.log
