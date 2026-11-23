#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# gc25d 双向 SFT 启动器：accelerate + DeepSpeed ZeRO-2，按 NGPU / GA 选（没有就生成）configs/accelerate_zero2*.yaml，
# 输出 tee -a 到 $OUT/train.log。同一个 OUT 已有训练在跑 ⇒ 拒绝启动（flock 锁 + 进程检查）。
#
# 用法（参数全走环境变量；其余 train.py 参数放 EXTRA，或直接跟在命令后面）：
#   OUT=outputs/sft_v10 bash scripts/train_sft.sh                                   # 8 卡 × micro 1 × GA 2 = 16，20000 步
#   OUT=outputs/sft_v10 RESUME=outputs/sft_v10/state bash scripts/train_sft.sh      # 崩溃续训（accelerator.load_state）
#   GC_RES=1280x704 DATA=data/gc25d_v10_704 OUT=outputs/sft_v10_704 bash scripts/train_sft.sh   # 704p，从头训
#   OUT=outputs/_smoke/sft NGPU=4 GPUS=2,3,4,5 GA=2 MAX_STEPS=10 DATA=data/gc25d_v10_mini \
#       EXTRA="--save_every 5 --val_every 5 --val_n 4 --warmup_steps 2" bash scripts/train_sft.sh
#
#   OUT        必填；相对路径相对 maliang 根目录；必须落在 $M/outputs/ 下（根盘只有 100G）
#   NGPU       进程数（默认 8；给了 GPUS 时必须与 GPUS 的个数一致）
#   GPUS       CUDA_VISIBLE_DEVICES（默认 0..NGPU-1）
#   GA         梯度累积（默认 2；全局 batch = NGPU × GA）
#   MAX_STEPS  默认 20000
#   RESUME     state 目录（续训）或 *.safetensors（热启动：gc25d 导出 / 普通 Wan 导出 ⇒ gc_sem_cond 零初始化），可空
#   VOCAB      parts（默认）/ ids
#   DATA       数据根（默认 $M/data/gc25d_v10）；它的 index/prep_args.json "res"（缺失 = 832x480）必须 = GC_RES
#   GC_RES     训练分辨率 <W>x<H>（默认 832x480；1280x704 = v10 原生分辨率），透传给 train.py（code/gc25d/geometry.py）
#   PORT       main_process_port（默认自动挑一个空闲端口，免得与同机别的 accelerate 任务撞）
#   EXTRA      其余 train.py 参数
#   ALLOW_EXISTING=1   OUT 里已有 step-*.safetensors / state 却没给 RESUME 时也照样从 step 0 开（默认拒绝，防止覆盖）
# 不要在训练运行中编辑本文件（bash 边读边执行）。
set -euo pipefail
M=$MALIANG_ROOT
PY="${PY:-$M/.venv/bin/python}"
ACC="${ACC:-$M/.venv/bin/accelerate}"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || command -v python || true)"; fi
if [ ! -x "$ACC" ]; then ACC="$(command -v accelerate || true)"; fi
[ -n "$PY" ]  || { echo "[train_sft] python not found (set PY=...)" >&2; exit 2; }
[ -n "$ACC" ] || { echo "[train_sft] accelerate not found (pip install accelerate)" >&2; exit 2; }

abspath() { case "$1" in /*) echo "$1" ;; *) echo "$M/$1" ;; esac; }

: "${OUT:?OUT 必填，例如 OUT=outputs/sft_v10}"
OUT=$(abspath "$OUT")
OUT=$("$PY" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$OUT")
case "$OUT/" in "$M/outputs/"*|"$M"/runs/*) ;; *) echo "[train_sft] note: OUT=$OUT is outside $M/outputs (allowed, but check disk space)" >&2 ;; esac
GA="${GA:-2}"
MAX_STEPS="${MAX_STEPS:-20000}"
VOCAB="${VOCAB:-parts}"
export GC_RES="${GC_RES:-832x480}"
DATA=$(abspath "${DATA:-$M/data/gc25d_v10}")
RESUME="${RESUME:-}"
[ -n "$RESUME" ] && RESUME=$(abspath "$RESUME")
EXTRA="${EXTRA:-}"
if [ -n "${GPUS:-}" ]; then
  N_LIST=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
  NGPU="${NGPU:-$N_LIST}"
  [ "$NGPU" = "$N_LIST" ] || { echo "[train_sft] NGPU=$NGPU 与 GPUS=$GPUS 的个数 $N_LIST 不一致" >&2; exit 2; }
else
  NGPU="${NGPU:-8}"
  GPUS=$(seq -s, 0 $((NGPU - 1)))
fi
PORT="${PORT:-$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
[ -f "$DATA/index/clips.jsonl" ] || { echo "[train_sft] $DATA/index/clips.jsonl 不存在（prep 没跑完？）" >&2; exit 2; }
[ -f "$DATA/READY" ] || echo "[train_sft] 注意：$DATA 没有 READY 标记（prep_v10 / verify_prep 可能没跑完）" >&2
# 分辨率：GC_RES 合法（geometry.parse_res）后规范成 RES_TAG（" 1280X704" ⇒ 1280x704），且 = 数据根的 res
# （index/prep_args.json，缺失 = 832x480），否则不启动
RES_PAIR=$(PYTHONPATH="$M/code/gc25d" "$PY" -c "import sys, geometry; print(geometry.RES_TAG, geometry.root_res(sys.argv[1]))" "$DATA") \
  || { echo "[train_sft] GC_RES=$GC_RES 不合法（见 code/gc25d/geometry.py::parse_res）" >&2; exit 2; }
read -r GC_RES DATA_RES <<< "$RES_PAIR"
export GC_RES
[ "$DATA_RES" = "$GC_RES" ] || { echo "[train_sft] DATA=$DATA 是 $DATA_RES 的数据，GC_RES=$GC_RES：设 GC_RES=$DATA_RES 或换 DATA" >&2; exit 2; }

# ---- accelerate yaml：8 卡用现成的两份，其它组合从 ga2 模板生成（只改 num_processes / gradient_accumulation_steps）----
if [ "$NGPU" = 8 ] && [ "$GA" = 2 ]; then
  CFG="$M/configs/accelerate_zero2_8gpu_ga2.yaml"
elif [ "$NGPU" = 8 ] && [ "$GA" = 1 ]; then
  CFG="$M/configs/accelerate_zero2_8gpu_ga1.yaml"
else
  CFG="$M/configs/accelerate_zero2_${NGPU}gpu_ga${GA}.yaml"
  if [ ! -f "$CFG" ]; then
    {
      echo "# ${NGPU} 卡 × GA ${GA}（全局 batch $((NGPU * GA))）：由 accelerate_zero2_8gpu_ga2.yaml 只改 num_processes / gradient_accumulation_steps"
      echo "# 生成（scripts/train_sft.sh）。train.py 的 --grad_accum 必须等于这里的 ${GA}，否则 SystemExit。"
      sed -e "s/^  gradient_accumulation_steps: 2$/  gradient_accumulation_steps: ${GA}/" \
          -e "s/^num_processes: 8$/num_processes: ${NGPU}/" "$M/configs/accelerate_zero2_8gpu_ga2.yaml"
    } > "$CFG.tmp.$$"
    grep -q "^  gradient_accumulation_steps: ${GA}$" "$CFG.tmp.$$" && grep -q "^num_processes: ${NGPU}$" "$CFG.tmp.$$" \
      || { echo "[train_sft] 生成 $CFG 失败（模板格式变了？）" >&2; rm -f "$CFG.tmp.$$"; exit 2; }
    mv "$CFG.tmp.$$" "$CFG"
    echo "[train_sft] 生成 $CFG"
  fi
fi

# ---- 同一个 OUT 只能有一个训练 ----
mkdir -p "$OUT"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$OUT/.train_sft.lock"
  flock -n 9 || { echo "[train_sft] $OUT 已有训练在跑（锁 $OUT/.train_sft.lock 被占用：$(cat "$OUT/.train_sft.pid" 2>/dev/null)），拒绝启动" >&2; exit 3; }
fi
if pgrep -af -- "train.py .*--output $OUT( |$)" >/dev/null 2>&1; then
  echo "[train_sft] 发现写 $OUT 的 train.py 进程，拒绝启动：" >&2
  pgrep -af -- "train.py .*--output $OUT( |$)" >&2
  exit 3
fi
if [ -z "$RESUME" ] && [ "${ALLOW_EXISTING:-0}" != 1 ] && { compgen -G "$OUT/step-*.safetensors" >/dev/null || [ -e "$OUT/state" ]; }; then
  echo "[train_sft] $OUT 里已有 checkpoint / state，但没给 RESUME：续训请 RESUME=$OUT/state，确要从 0 开请 ALLOW_EXISTING=1" >&2
  exit 3
fi
echo "$$ $(date '+%F %T') GPUS=$GPUS" > "$OUT/.train_sft.pid"

# ---- 环境：缓存 / 临时文件放在仓库的 .tmp/ 下 ----
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH="$M/code/gc25d:$M/code/arope"
export HF_HOME=${HF_HOME:-$MALIANG_ROOT/.cache/huggingface}
export TMPDIR="$M/.tmp" TORCHINDUCTOR_CACHE_DIR="$M/.tmp/inductor" TRITON_CACHE_DIR="$M/.tmp/triton"
export TORCH_EXTENSIONS_DIR="$M/.tmp/torch_extensions" TORCH_HOME="$M/.tmp/torch_home" XDG_CACHE_HOME="$M/.tmp/xdg_cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPUS"
mkdir -p "$TMPDIR"

CMD=("$ACC" launch --config_file "$CFG" --main_process_port "$PORT"
     "$M/code/gc25d/train.py"
     --data_root "$DATA" --vocab "$VOCAB" --grad_accum "$GA" --max_steps "$MAX_STEPS" --output "$OUT")
[ -n "$RESUME" ] && CMD+=(--resume "$RESUME")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && CMD+=($EXTRA)
CMD+=("$@")
{
  echo "[train_sft] $(date '+%F %T') host=$(hostname) GPUS=$GPUS NGPU=$NGPU GA=$GA（全局 batch $((NGPU * GA))）PORT=$PORT GC_RES=$GC_RES"
  echo "[train_sft] cfg=$CFG"
  echo "[train_sft] ${CMD[*]}"
} | tee -a "$OUT/train.log"
"${CMD[@]}" 2>&1 | tee -a "$OUT/train.log"
