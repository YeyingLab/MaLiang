#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# ─────────────────────────────────────────────────────────────────────────────
# stage1：teacher forcing 因果化（gc25d SFT → 因果 AR 扩散模型）。ARoPE distill/stage1_ar/run.sh 的 maliang 版。
#
#   DeepSpeed ZeRO-2，等效 batch 8，lr 1e-5，8000 步（gc25d_v8 口径）。
#   产物 <OUT>/dits/dit_step{N}.safetensors（831 键）同时做 stage2 的 --teacher_ckpt 和 --student_ckpt。
#
#   SFT=<outputs/sft_fxcond_v1/step-N.safetensors> bash code/gc25d/gcdistill/run_stage1.sh
#   # 小规模冒烟：等效 batch 不是 8，必须显式 EFF_BATCH_OVERRIDE=1；OUT 必须写绝对路径
#   DISTILL_NGPU=2 DISTILL_GPUS=6,7 EFF_BATCH_OVERRIDE=1 MAX_STEPS=4 SAVE_EVERY=4 LOG_EVERY=1 VAL_EVERY=0 \
#     OUT=$MALIANG_ROOT/outputs/_smoke/distill/stage1 bash code/gc25d/gcdistill/run_stage1.sh
#
# 这里的每一个配方常量都与 ar_tf_gc.py 的 argparse 默认值、config_stage1.json 逐项相同
# （tests/test_distill_config.py 逐项比；改配方 = 同时改这三处）。只有两处**故意**与 argparse 默认值不同：
#   --save_state（argparse 默认关：不想让随手一跑就写两个 ~60 GB 的 state 槽）
#   --resume auto（argparse 默认 None：不想让随手一跑就悄悄接着上次的 step 跑）
# ─────────────────────────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_gc.sh
source "$HERE/common_gc.sh"          # set -euo pipefail / M / PY / ACC / 环境 / 选卡 / 空闲守卫

# ---- 输入输出（都可以用环境变量覆盖）----
SFT="${SFT:-$M/outputs/sft_fxcond_v1/step-10667.safetensors}"     # 双向 SFT
INIT="${INIT:-$SFT}"
DATA_ROOT="${DATA_ROOT:-$M/data/gc25d_v10}"
TEXT_TABLE="${TEXT_TABLE:-$DATA_ROOT/text_table_gc25d.pt}"
OUT="${OUT:-$M/outputs/distill/fxcond_v1/stage1_tf}"

# ---- 并行与端口 ----
PORT="${PORT:-29631}"                # 固定分配：stage1 29631 / stage2 29632 / stage3 29636
# 等效 batch **恒为 8**；卡数变的是墙钟，不是配方。GA 与 yaml 按 DISTILL_NGPU 一起选。
if [ "$DISTILL_NGPU" = "8" ]; then
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
  ACC_CFG="${ACC_CFG:-$M/configs/accelerate_zero2_8gpu_ga${GRAD_ACCUM}.yaml}"
else
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
  ACC_CFG="${ACC_CFG:-$M/configs/accelerate_zero2_${DISTILL_NGPU}gpu_ga${GRAD_ACCUM}.yaml}"
fi

# ---- 配方（= ar_tf_gc.py 的 argparse 默认值 = config_stage1.json）----
VOCAB="${VOCAB:-auto}"                    # auto：按 init ckpt 的 gc_sem_cond 判断（parts / ids）
BASE_INIT="${BASE_INIT:-skeleton}"        # 不读 Wan 基础权重，直接从 831 键 ckpt 严格装满（逐位相同）
LR="${LR:-1e-5}"
COND_LR="${COND_LR:-0.0}"                 # 0 = gc_sem_cond 与 DiT 同 lr
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETAS="${ADAM_BETAS:-0.9,0.999}"     # SFT 的口径（stage1 = SFT 目标 + 因果掩码）
WARMUP="${WARMUP:-200}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"            # 三处都是 10.0（stage1 gnorm≈2.5，1.0 会步步都裁）
MAX_STEPS="${MAX_STEPS:-8000}"
TEXT_DROPOUT="${TEXT_DROPOUT:-0.1}"       # SFT 同值
COND_DROPOUT="${COND_DROPOUT:-0.05}"
MASK_P="${MASK_P:-0.5}"
TF_AUG="${TF_AUG:-0}"
WINDOW="${WINDOW:-0}"                     # 全因果；F=21 下 sink=0 的窗口会把 cell 0 挤出视野
WINDOW_SINK="${WINDOW_SINK:-0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_DITS="${KEEP_DITS:-6}"
VAL_EVERY="${VAL_EVERY:-500}"
VAL_N="${VAL_N:-16}"
VAL_TS="${VAL_TS:-200,500,800}"
LOG_EVERY="${LOG_EVERY:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-0}"
SYNTHETIC="${SYNTHETIC:-0}"               # >0：全尺寸合成数据（只给 prep 写完之前的冒烟用）
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-auto}"                  # auto：有 <OUT>/state 就续训；none 关掉

# ---- 前置检查 ----
require_abs_out "$OUT"
[ -f "$INIT" ]    || { echo "ABORT: 没有 init 检查点: $INIT（用 SFT=<abs> 或 INIT=<abs> 指定双向 SFT）"; exit 1; }
if [ "$SYNTHETIC" = "0" ]; then
  [ -f "$DATA_ROOT/index/clips.jsonl" ] || { echo "ABORT: 没有数据索引: $DATA_ROOT/index/clips.jsonl"; exit 1; }
  [ -f "$TEXT_TABLE" ] || { echo "ABORT: 没有文本表: $TEXT_TABLE"; exit 1; }
  require_data_res "$DATA_ROOT" || exit 1
fi
[ -f "$ACC_CFG" ] || { echo "ABORT: 没有 accelerate 配置: $ACC_CFG（GRAD_ACCUM=$GRAD_ACCUM DISTILL_NGPU=$DISTILL_NGPU）"; exit 1; }
# 等效 batch 断言：eff = 卡数 × batch 1 × GA，配方钉死 8。小规模冒烟显式 EFF_BATCH_OVERRIDE=1。
if [ $((DISTILL_NPROC * GRAD_ACCUM)) -ne 8 ] && [ "${EFF_BATCH_OVERRIDE:-0}" != "1" ]; then
  echo "ABORT: 等效 batch = $DISTILL_NPROC 卡 × batch 1 × GA $GRAD_ACCUM = $((DISTILL_NPROC * GRAD_ACCUM))，配方是 8（8 卡 GA1）。"
  echo "       冒烟 / 换配方：EFF_BATCH_OVERRIDE=1，并自行保证 ACC_CFG 的 gradient_accumulation_steps == GRAD_ACCUM。"
  exit 1
fi
mkdir -p "$OUT"

ARGS=(
  --init_ckpt "$INIT" --base_init "$BASE_INIT"
  --data_root "$DATA_ROOT" --text_table "$TEXT_TABLE" --vocab "$VOCAB" --synthetic "$SYNTHETIC"
  --text_dropout "$TEXT_DROPOUT" --cond_dropout "$COND_DROPOUT" --mask_p "$MASK_P"
  --teacher_forcing --tf_noise_aug_max_idx "$TF_AUG"
  --windowed_train "$WINDOW" --windowed_sink "$WINDOW_SINK"
  --lr "$LR" --cond_lr "$COND_LR" --weight_decay "$WEIGHT_DECAY" --adam_betas "$ADAM_BETAS"
  --warmup "$WARMUP" --grad_clip "$GRAD_CLIP"
  --batch_size 1 --grad_accum "$GRAD_ACCUM" --max_steps "$MAX_STEPS"
  --gradient_checkpointing
  --save_every "$SAVE_EVERY" --keep_last_dits "$KEEP_DITS" --save_state
  --val_every "$VAL_EVERY" --val_n "$VAL_N" --val_timesteps "$VAL_TS"
  --log_every "$LOG_EVERY" --num_workers "$NUM_WORKERS" --seed "$SEED"
  --output "$OUT"
)
# 写成 if 而不是 `[ cond ] && ARGS+=(...)`：set -e 下 && 链的写法太容易踩坑
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ -n "$RESUME" ] && [ "$RESUME" != "none" ]; then
  ARGS+=(--resume "$RESUME")
fi

echo "[stage1] init=$INIT"
echo "[stage1] out=$OUT  port=$PORT  cfg=$(basename "$ACC_CFG")  gpus=$DISTILL_GPUS  等效 batch=$((DISTILL_NPROC * GRAD_ACCUM))  res=$GC_RES"
echo "[stage1] lr=$LR betas=$ADAM_BETAS warmup=$WARMUP clip=$GRAD_CLIP max_steps=$MAX_STEPS dropout=$TEXT_DROPOUT/$COND_DROPOUT/$MASK_P"

# train.log 由 ar_tf_gc.py 自己带时间戳写；launch.log 收 NCCL / DeepSpeed / traceback 的原始输出
"$ACC" launch \
  --config_file "$ACC_CFG" \
  --num_processes "$DISTILL_NPROC" --main_process_port "$PORT" \
  "$HERE/ar_tf_gc.py" "${ARGS[@]}" "$@" 2>&1 | tee -a "$OUT/launch.log"
