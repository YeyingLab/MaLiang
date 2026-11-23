#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# After experiment 6 generation + scoring: SolarRenderer inputs for every case that has a FLUX first frame.
C=$MALIANG_ROOT/experiments; RT=$MALIANG_ROOT
until grep -q EXP6_ALL_DONE $C/exp6/score.log 2>/dev/null; do sleep 300; done
export AROPE_ROOT=$RT/code/arope PYTHONPATH=$RT/third_party/DiffSynth-Studio:$RT/code/arope
$RT/.venv/bin/python $C/exp6/build_solar_unseen.py --cases $C/exp6/cases $C/exp6/sketch/cases --out $C/exp6/solar_inputs   --conditions_dir $C/testset/caption_skill > $C/exp6/solar_inputs.log 2>&1
