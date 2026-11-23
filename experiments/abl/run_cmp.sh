#!/bin/bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Same-budget student distillation of the two interface teachers (paper ablation follow-up):
#   v3_ids          integer map  -> class-pooling encoder (gc_sem_cond)        teacher sft_fxcond_v1 step-5000
#   v2_color_light  colour-coded map -> light encoder (gc_wb_cond)             teacher experiments/abl/runs/color_light step-5000
# Identical recipe for both: stage 1 (teacher forcing) 2000 steps, then stage 2 (causal consistency, deploy 4-step axis)
# 2000 steps from the stage-1 result; effective batch 6 (GPUs x GA); same seeds, data and dropout.
# A failed try restarts that stage from its newest exported dit (fresh optimizer), counting the steps already done.
M="${M:-$MALIANG_ROOT}"; G="${G:-$M/code/gc25d/gcdistill}"; D="${D:-$M/data/gc25d_v10_fxcond}"; O="${O:-$M/outputs/cmp}"; mkdir -p $O
GPUS="${GPUS:-0}"; NGPU="${NGPU:-$(echo "$GPUS" | tr ',' '\n' | grep -c .)}"; GA="${GA:-3}"; CFG="${CFG:-$M/configs/accelerate_zero2_${NGPU}gpu_ga${GA}.yaml}"
NEED="${NEED:-95}"; S1="${S1:-2000}"; S2="${S2:-2000}"; LOG=$O/run_${VARIANT}.log; TEACHER="${TEACHER:?}"; COL="${COL:?}"
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
common() { echo DATA_ROOT=$D TEXT_TABLE=$D/text_table_gc25d.pt VOCAB=parts_fx SEED=0 GPU_MIN_FREE_GB=$NEED DISTILL_NGPU=$NGPU \
  DISTILL_GPUS=$GPUS GRAD_ACCUM=$GA EFF_BATCH_OVERRIDE=1 ACC_CFG=$CFG NUM_WORKERS=2; }
newest() { ls $1/dit_step*.safetensors 2>/dev/null | sed 's/.*dit_step\([0-9]*\).safetensors/\1 &/' | sort -n | tail -1; }
stage1() {   # $1 variant  $2 teacher  $3 color(0/1)
  local v=$1 init=$2 col=$3 done_=0 t
  for t in 1 2 3 4 5; do
    [ -f $O/$v/S1_DONE ] && return 0
    local out=$O/$v/s1_t$t; mkdir -p $out
    log "$v stage1 try $t: init $init (+$done_ steps done), $((S1 - done_)) steps -> $out"
    env $(common) COLOR_IDS=$col INIT=$init MAX_STEPS=$((S1 - done_)) SAVE_EVERY=500 KEEP_DITS=1 SAVE_STATE=0 VAL_EVERY=0 \
        RESUME=none OUT=$out PORT=29641 bash $G/run_stage1.sh >> $out/console.log 2>&1
    local nw; nw=$(newest $out/dits)
    if [ -n "$nw" ]; then local n=${nw%% *}; done_=$((done_ + n)); init=${nw#* }; fi
    if [ $done_ -ge $S1 ]; then ln -sfn $init $O/$v/stage1_final.safetensors; touch $O/$v/S1_DONE; log "$v stage1 done ($init)"; return 0; fi
    log "$v stage1 try $t ended at $done_/$S1 steps; tail:"; tail -5 $out/console.log | tee -a $LOG; sleep 120
  done; return 1
}
stage2() {   # $1 variant  $2 color
  local v=$1 col=$2 ar; ar=$(readlink -f $O/$v/stage1_final.safetensors)
  for t in 1 2 3; do
    [ -f $O/$v/S2_DONE ] && return 0
    local out=$O/$v/s2; mkdir -p $out
    log "$v stage2 try $t: AR $ar, $S2 steps -> $out"
    env $(common) COLOR_IDS=$col AR=$ar MAX_STEPS=$S2 SAVE_EVERY=500 KEEP_EMAS=1 KEEP_DITS=1 RESUME=auto FORCE_RUN_DIR=1 \
        OUT=$out PORT=29642 bash $G/run_stage2.sh >> $out/console.log 2>&1
    if ls $out/ema/ema_step${S2}.safetensors >/dev/null 2>&1; then touch $O/$v/S2_DONE; log "$v stage2 done"; return 0; fi
    log "$v stage2 try $t failed; tail:"; tail -5 $out/console.log | tee -a $LOG; sleep 120
  done; return 1
}
log "start $VARIANT on GPUs $GPUS (NGPU $NGPU x GA $GA, cfg $(basename $CFG)), teacher $TEACHER, COLOR_IDS=$COL"
stage1 $VARIANT $TEACHER $COL && stage2 $VARIANT $COL && log "CMP_DONE $VARIANT"
