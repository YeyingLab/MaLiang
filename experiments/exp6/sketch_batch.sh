#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Hand-drawn-map branch of experiment 6. Waits for the procmap branch (exp6_run.sh) to finish, then
#   * human sketches: sketch/human/<clip>__<k>.png (drawn on tpl_<clip>/template_x3.png) -> case human<k>__<clip>
#   * synthetic stand-ins (sketchmap.py synth, seeds 1-2 per clip) -> case synth<k>__<clip>   (labelled as programmatic)
# and runs FLUX first frame + variant (3) on every sketch case (exp6_run.sh with OUT=sketch/cases, SKIP_PROCMAP=1).
C=$MALIANG_ROOT/experiments; E=$C/exp6; RT=$MALIANG_ROOT; GPU=${GPU:-3}
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope CUDA_VISIBLE_DEVICES=$GPU
while pgrep -u $(id -un) -f '[e]xp6_run.sh' >/dev/null; do sleep 300; done   # the procmap branch must be finished
GPU=$GPU bash $E/exp6_run.sh                                    # retry procmap cases that failed (OOM on a shared GPU)
cd $E; P=$RT/.venv/bin/python
for t in sketch/tpl_*/template.json; do
  c=$(basename $(dirname $t)); c=${c#tpl_}
  for k in 1 2; do
    [ -f sketch/cases/synth${k}__$c/ids.npz ] && continue
    $P sketchmap.py synth --template $t --seed $k --out sketch/tpl_$c/synth$k.png >> sketch/build.log 2>&1
    $P sketchmap.py build --sketch sketch/tpl_$c/synth$k.png --template $t --case synth${k}__$c --out sketch/cases >> sketch/build.log 2>&1
  done
done
for s in sketch/human/*.png; do
  [ -f "$s" ] || continue
  b=$(basename $s .png); c=${b%__*}; k=${b##*__}
  [ -f sketch/cases/human${k}__$c/ids.npz ] && continue
  $P sketchmap.py build --sketch $s --template sketch/tpl_$c/template.json --case human${k}__$c --out sketch/cases --human >> sketch/build.log 2>&1
done
OUT=$E/sketch/cases SKIP_PROCMAP=1 GPU=$GPU bash $E/exp6_run.sh
echo SKETCH_BATCH_DONE >> sketch/build.log
