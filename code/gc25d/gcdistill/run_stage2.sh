#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# ─────────────────────────────────────────────────────────────────────────────
# stage2：因果一致性蒸馏（CD）—— 把 stage1 的多步因果 AR 模型压成 4 步学生。ARoPE stage2_cd/run.sh 的 maliang 版。
#
#   DeepSpeed ZeRO-2，等效 batch 8，lr 2e-6，5000 步（gc25d_v8 口径）。
#   ★ 产物是 <OUT>/ema/ema_step{N}.safetensors —— 部署权重、也是 stage3 的 STUDENT。dits/ 只为续训存在。
#
#   AR=<stage1 的 dits/dit_step8000.safetensors> bash code/gc25d/gcdistill/run_stage2.sh
#   # 小规模冒烟：
#   DISTILL_NGPU=2 DISTILL_GPUS=6,7 EFF_BATCH_OVERRIDE=1 MAX_STEPS=4 SAVE_EVERY=4 LOG_EVERY=1 EMA_START=2 \
#     AR=<abs> OUT=$MALIANG_ROOT/outputs/_smoke/distill/stage2 bash code/gc25d/gcdistill/run_stage2.sh
#
# 这里的每一个配方常量都与 cd_gc.py 的 argparse 默认值、config_stage2.json 逐项相同（tests/test_distill_config.py）。
# 只有一处**故意**与 argparse 默认值不同：--resume auto（argparse 默认 None）。
# ─────────────────────────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_gc.sh
source "$HERE/common_gc.sh"

# ---- 输入输出 ----
# P1 忠实路线：teacher 与 student 的 init 是**同一份** stage1 产物
STAGE1_OUT="${STAGE1_OUT:-$M/outputs/distill/fxcond_v1/stage1_tf}"
AR="${AR:-$STAGE1_OUT/dits/dit_step8000.safetensors}"
DATA_ROOT="${DATA_ROOT:-$M/data/gc25d_v10}"
TEXT_TABLE="${TEXT_TABLE:-$DATA_ROOT/text_table_gc25d.pt}"
OUT="${OUT:-$M/outputs/distill/fxcond_v1/stage2_cd}"

# ---- 并行与端口 ----
PORT="${PORT:-29632}"
if [ "$DISTILL_NGPU" = "8" ]; then
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
  ACC_CFG="${ACC_CFG:-$M/configs/accelerate_zero2_8gpu_ga${GRAD_ACCUM}.yaml}"
else
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
  ACC_CFG="${ACC_CFG:-$M/configs/accelerate_zero2_${DISTILL_NGPU}gpu_ga${GRAD_ACCUM}.yaml}"
fi

# ---- 一致性轴 = 部署采样表 ----
CD_AXIS="${CD_AXIS:-deploy}"
DENOISE_LIST="${DENOISE_LIST:-1000,750,500,250}"
DISCRETE_N="${DISCRETE_N:-48}"
CD_TAIL_T="${CD_TAIL_T:-0.0}"
SOLVER_STEPS="${SOLVER_STEPS:-1}"

# ---- 配方（= cd_gc.py 的 argparse 默认值 = config_stage2.json）----
VOCAB="${VOCAB:-auto}"
BASE_INIT="${BASE_INIT:-skeleton}"
TEACHER_CAUSAL="${TEACHER_CAUSAL:-1}"     # 1 = P1 忠实（因果教师 = stage1 产物）
CFG_SCALE="${CFG_SCALE:-1.0}"             # CF++ 铁律：CD / DMD 都是 1.0
EMA_DECAY="${EMA_DECAY:-0.99}"
EMA_START="${EMA_START:-200}"
LR="${LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETAS="${ADAM_BETAS:-0.0,0.999}"     # CF 约定 beta1=0
WARMUP="${WARMUP:-100}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
MAX_STEPS="${MAX_STEPS:-5000}"
TEXT_DROPOUT="${TEXT_DROPOUT:-0.0}"       # CD 不做 CFG ⇒ dropout 全 0
COND_DROPOUT="${COND_DROPOUT:-0.0}"
MASK_P="${MASK_P:-0.0}"
WINDOW="${WINDOW:-0}"
WINDOW_SINK="${WINDOW_SINK:-0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_EMAS="${KEEP_EMAS:-25}"              # stage3 的 init 是从这些里挑的
KEEP_DITS="${KEEP_DITS:-2}"
LOG_EVERY="${LOG_EVERY:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-0}"
SYNTHETIC="${SYNTHETIC:-0}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-auto}"
FORCE_RUN_DIR="${FORCE_RUN_DIR:-0}"

# ---- 前置检查 ----
require_abs_out "$OUT"
if [ ! -f "$AR" ]; then
  echo "ABORT: 没有 stage1 检查点: $AR（AR=<abs> 指定；teacher 与 student init 共用这一个变量）"
  [ -d "$STAGE1_OUT/dits" ] && ls -1 "$STAGE1_OUT/dits" | tail -8 | sed 's/^/         /'
  exit 1
fi
case "$AR" in
  */ema/*) echo "[stage2][warn] AR 指向 ema/：stage1 没有 EMA，确认这真是 stage1 的产物" ;;
esac
if [ "$SYNTHETIC" = "0" ]; then
  [ -f "$DATA_ROOT/index/clips.jsonl" ] || { echo "ABORT: 没有数据索引: $DATA_ROOT/index/clips.jsonl"; exit 1; }
  [ -f "$TEXT_TABLE" ] || { echo "ABORT: 没有文本表: $TEXT_TABLE"; exit 1; }
  require_data_res "$DATA_ROOT" || exit 1
fi
[ -f "$ACC_CFG" ] || { echo "ABORT: 没有 accelerate 配置: $ACC_CFG"; exit 1; }
if [ $((DISTILL_NPROC * GRAD_ACCUM)) -ne 8 ] && [ "${EFF_BATCH_OVERRIDE:-0}" != "1" ]; then
  echo "ABORT: 等效 batch = $DISTILL_NPROC 卡 × batch 1 × GA $GRAD_ACCUM = $((DISTILL_NPROC * GRAD_ACCUM))，配方是 8（8 卡 GA1）。"
  echo "       冒烟 / 换配方：EFF_BATCH_OVERRIDE=1（cd_gc.py 禁止续训中途改 GA —— step 0 就锁死）。"
  exit 1
fi
mkdir -p "$OUT"

ARGS=(
  --teacher_ckpt "$AR" --student_ckpt "$AR" --base_init "$BASE_INIT"
  --teacher_causal "$TEACHER_CAUSAL" --teacher_forcing
  --windowed_train "$WINDOW" --windowed_sink "$WINDOW_SINK"
  --cd_axis "$CD_AXIS" --denoise_list "$DENOISE_LIST"
  --discrete_cd_N "$DISCRETE_N" --cd_tail_t "$CD_TAIL_T"
  --teacher_solver_steps "$SOLVER_STEPS" --real_guidance_scale "$CFG_SCALE"
  --ema_decay "$EMA_DECAY" --ema_start "$EMA_START"
  --data_root "$DATA_ROOT" --text_table "$TEXT_TABLE" --vocab "$VOCAB" --synthetic "$SYNTHETIC"
  --text_dropout "$TEXT_DROPOUT" --cond_dropout "$COND_DROPOUT" --mask_p "$MASK_P"
  --lr "$LR" --weight_decay "$WEIGHT_DECAY" --adam_betas "$ADAM_BETAS"
  --warmup "$WARMUP" --grad_clip "$GRAD_CLIP"
  --batch_size 1 --grad_accum "$GRAD_ACCUM" --max_steps "$MAX_STEPS"
  --gradient_checkpointing
  --save_every "$SAVE_EVERY" --keep_last_emas "$KEEP_EMAS" --keep_last_dits "$KEEP_DITS"
  --log_every "$LOG_EVERY" --num_workers "$NUM_WORKERS" --seed "$SEED"
  --output "$OUT"
)
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ -n "$RESUME" ] && [ "$RESUME" != "none" ]; then
  ARGS+=(--resume "$RESUME")
fi
if [ "$FORCE_RUN_DIR" = "1" ]; then
  ARGS+=(--force_run_dir)
fi

echo "[stage2] teacher = student init = $AR（P1）"
echo "[stage2] out=$OUT  port=$PORT  cfg=$(basename "$ACC_CFG")  gpus=$DISTILL_GPUS  等效 batch=$((DISTILL_NPROC * GRAD_ACCUM))  res=$GC_RES"
echo "[stage2] axis=$CD_AXIS denoise_list=$DENOISE_LIST solver=$SOLVER_STEPS cfg_scale=$CFG_SCALE"
echo "[stage2] lr=$LR betas=$ADAM_BETAS warmup=$WARMUP clip=$GRAD_CLIP max_steps=$MAX_STEPS ema=$EMA_DECAY@$EMA_START"
echo "[stage2] 产物 = $OUT/ema/ema_step{N}.safetensors（部署 + stage3 的 STUDENT）；dits/ 只为续训"

"$ACC" launch \
  --config_file "$ACC_CFG" \
  --num_processes "$DISTILL_NPROC" --main_process_port "$PORT" \
  "$HERE/cd_gc.py" "${ARGS[@]}" "$@" 2>&1 | tee -a "$OUT/launch.log"
