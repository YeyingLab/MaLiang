#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# ─────────────────────────────────────────────────────────────────────────────
# stage3：self-forcing DMD（stage2 的 EMA → 自回归抛光）。ARoPE stage3_dmd/run.sh 的 maliang 版。
#
#   FSDP1 FULL_SHARD（**不是** ZeRO-2：student + critic 两个可训模型），500 步，SAVE_EVERY=50。
#   部署几何 sink 3 / window 17 / cap 20，打分尾巴 frankenstein（gc25d_v8 ckpt 的 tail_mode），rollout 21..27 个 cell，
#   条件全部来自真实数据（cond + cond_ext，has_ext 的 clip）。产物 <OUT>/ema/ema_step{N} —— **逐个里程碑挑**。
#   stage3 没有 GA：等效 batch = 卡数 = 8（小规模冒烟要 EFF_BATCH_OVERRIDE=1）。
#
#   STUDENT=<stage2 的 ema/ema_step5000.safetensors> SFT=<双向 SFT> bash code/gc25d/gcdistill/run_stage3.sh
#   # 小规模冒烟：，首次必须量显存：
#   DISTILL_NGPU=2 DISTILL_GPUS=6,7 EFF_BATCH_OVERRIDE=1 MAX_STEPS=3 SAVE_EVERY=3 LOG_EVERY=1 PROFILE_MEM=1 \
#     ROLLOUT_MAX=22 STUDENT=<abs ema> SFT=<abs> OUT=$MALIANG_ROOT/outputs/_smoke/distill/stage3 \
#     bash code/gc25d/gcdistill/run_stage3.sh
#
# 这里的每一个配方常量都与 dmd_gc.py 的 argparse 默认值、config_stage3.json 逐项相同（tests/test_distill_config.py）。
# 只有一处**故意**与 argparse 默认值不同：--resume auto（argparse 默认 None）。
# ⚠️ 几何铁律：rope_cap == sink_size + kv_window == F−1 == 20。采样表铁律：stage2 的 CD 轴、stage3、部署是同一张 4 步表。
# ─────────────────────────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_gc.sh
source "$HERE/common_gc.sh"

# ---- 输入输出 ----
STAGE2_OUT="${STAGE2_OUT:-$M/outputs/distill/fxcond_v1/stage2_cd}"
STUDENT="${STUDENT:-$STAGE2_OUT/ema/ema_step5000.safetensors}"     # 生成器初值 = stage2 的 **ema/**（不是 dits/）
SFT="${SFT:-$M/outputs/sft_fxcond_v1/step-10667.safetensors}"
TEACHER="${TEACHER:-$SFT}"                                          # real-score 教师 + critic 初值 = **双向** SFT
DATA_ROOT="${DATA_ROOT:-$M/data/gc25d_v10}"
TEXT_TABLE="${TEXT_TABLE:-$DATA_ROOT/text_table_gc25d.pt}"
OUT="${OUT:-$M/outputs/distill/fxcond_v1/stage3_dmd}"

# ---- 并行与端口 ----
PORT="${PORT:-29636}"
ACC_CFG="${ACC_CFG:-$M/configs/accelerate_fsdp_dmd_${DISTILL_NGPU}gpu.yaml}"

# ---- 配方（= dmd_gc.py 的 argparse 默认值 = config_stage3.json）----
VOCAB="${VOCAB:-auto}"
BASE_INIT="${BASE_INIT:-skeleton}"
DENOISE_LIST="${DENOISE_LIST:-1000,750,500,250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
KV_WINDOW="${KV_WINDOW:-17}"
SINK="${SINK:-3}"
ROPE_CAP="${ROPE_CAP:-20}"           # 必须 == SINK + KV_WINDOW
ROLLOUT_MIN="${ROLLOUT_MIN:-21}"     # ≥ sink+window+1 = 21 才练得到驱逐后的读取
ROLLOUT_MAX="${ROLLOUT_MAX:-27}"     # ≤ 27：cond 21 + cond_ext 7 = 28 个 cell 的真实条件
SCORE_FRAMES="${SCORE_FRAMES:-20}"   # 1+20 = 21 = teacher / critic 的训练片长
GRAD_HORIZON="${GRAD_HORIZON:-20}"
GRAD_PASS="${GRAD_PASS:-streamed}"
TAIL_MODE="${TAIL_MODE:-frankenstein}"
CRITIC_STEPS="${CRITIC_STEPS:-5}"
LR="${LR:-2e-6}"                     # 5e-6 是已知的高 lr min-max 崩溃
LR_CRITIC="${LR_CRITIC:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETAS="${ADAM_BETAS:-0.0,0.999}"
WARMUP="${WARMUP:-100}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
MAX_STEPS="${MAX_STEPS:-500}"
CLAMP="${CLAMP:-20,980}"
EMA_DECAY="${EMA_DECAY:-0.99}"
EMA_START="${EMA_START:-100}"
SAVE_EVERY="${SAVE_EVERY:-50}"
KEEP_DITS="${KEEP_DITS:-3}"
KEEP_EMAS="${KEEP_EMAS:-25}"
KEEP_CRITICS="${KEEP_CRITICS:-3}"
LOG_EVERY="${LOG_EVERY:-10}"
PROFILE_MEM="${PROFILE_MEM:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-0}"
SYNTHETIC="${SYNTHETIC:-0}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-auto}"
FORCE_RUN_DIR="${FORCE_RUN_DIR:-0}"

# ---- 前置检查 ----
require_abs_out "$OUT"
[ -f "$STUDENT" ] || { echo "ABORT: 没有 stage2 的 EMA: $STUDENT（STUDENT=<abs>，必须是 ema/ema_step{N}.safetensors）"; exit 1; }
case "$STUDENT" in
  */dits/*) echo "ABORT: STUDENT 指向 dits/：$STUDENT —— stage2 的部署权重是 ema/，dits/ 只为续训存在"; exit 1;;
esac
[ -f "$TEACHER" ]  || { echo "ABORT: 没有双向 SFT 教师: $TEACHER（SFT=<abs>）"; exit 1; }
if [ "$SYNTHETIC" = "0" ]; then
  [ -f "$DATA_ROOT/index/clips.jsonl" ] || { echo "ABORT: 没有数据索引: $DATA_ROOT/index/clips.jsonl"; exit 1; }
  [ -f "$TEXT_TABLE" ] || { echo "ABORT: 没有文本表: $TEXT_TABLE"; exit 1; }
  require_data_res "$DATA_ROOT" || exit 1
fi
[ -f "$ACC_CFG" ]  || { echo "ABORT: 没有 accelerate 配置: $ACC_CFG"; exit 1; }
if [ "$ROPE_CAP" -ne "$((SINK + KV_WINDOW))" ]; then
  echo "ABORT: ROPE_CAP($ROPE_CAP) != SINK($SINK) + KV_WINDOW($KV_WINDOW)"; exit 1
fi
# 等效 batch 断言：stage3 无 GA，eff = 卡数 × batch 1，配方 8
if [ "$DISTILL_NPROC" -ne 8 ] && [ "${EFF_BATCH_OVERRIDE:-0}" != "1" ]; then
  echo "ABORT: 等效 batch = $DISTILL_NPROC 卡 × batch 1 = $DISTILL_NPROC，配方是 8。冒烟：EFF_BATCH_OVERRIDE=1"
  exit 1
fi
mkdir -p "$OUT"

ARGS=(
  --student_ckpt "$STUDENT" --teacher_ckpt "$TEACHER" --base_init "$BASE_INIT"
  --data_root "$DATA_ROOT" --text_table "$TEXT_TABLE" --vocab "$VOCAB" --synthetic "$SYNTHETIC"
  --denoise_list "$DENOISE_LIST" --real_guidance_scale "$CFG_SCALE"
  --kv_window "$KV_WINDOW" --sink_size "$SINK" --rope_cap "$ROPE_CAP"
  --rollout_min "$ROLLOUT_MIN" --rollout_max "$ROLLOUT_MAX"
  --dmd_score_frames "$SCORE_FRAMES" --grad_horizon "$GRAD_HORIZON" --grad_pass "$GRAD_PASS"
  --tail_mode "$TAIL_MODE" --dmd_timestep_clamp "$CLAMP" --critic_steps "$CRITIC_STEPS"
  --lr "$LR" --lr_critic "$LR_CRITIC" --weight_decay "$WEIGHT_DECAY"
  --adam_betas "$ADAM_BETAS" --warmup "$WARMUP" --grad_clip "$GRAD_CLIP"
  --batch_size 1 --max_steps "$MAX_STEPS" --gradient_checkpointing
  --use_ema 1 --ema_decay "$EMA_DECAY" --ema_start "$EMA_START"
  --save_every "$SAVE_EVERY" --keep_last_dits "$KEEP_DITS"
  --keep_last_emas "$KEEP_EMAS" --keep_last_critics "$KEEP_CRITICS"
  --log_every "$LOG_EVERY" --profile_mem "$PROFILE_MEM"
  --num_workers "$NUM_WORKERS" --seed "$SEED"
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

echo "[stage3] student(生成器) = $STUDENT"
echo "[stage3] teacher/critic  = $TEACHER   （双向 SFT，不是 stage1 的因果 ckpt）"
echo "[stage3] out=$OUT  port=$PORT  cfg=$(basename "$ACC_CFG")  gpus=$DISTILL_GPUS  $DISTILL_NPROC 卡 FSDP1  等效 batch=$DISTILL_NPROC  res=$GC_RES"
echo "[stage3] 表 $DENOISE_LIST · sink $SINK / window $KV_WINDOW / cap $ROPE_CAP · rollout $ROLLOUT_MIN..$ROLLOUT_MAX · tail $TAIL_MODE"
echo "[stage3] K=$SCORE_FRAMES horizon=$GRAD_HORIZON pass=$GRAD_PASS critic_steps=$CRITIC_STEPS lr=$LR/$LR_CRITIC max_steps=$MAX_STEPS"

"$ACC" launch \
  --config_file "$ACC_CFG" \
  --num_processes "$DISTILL_NPROC" --main_process_port "$PORT" \
  "$HERE/dmd_gc.py" "${ARGS[@]}" "$@" 2>&1 | tee -a "$OUT/launch.log"
