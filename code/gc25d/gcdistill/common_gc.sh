#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# gc25d 三阶段蒸馏所有 launcher 的公共前置：source 它，不要执行它（ARoPE distill/common.sh 的 maliang 版）。
#   source "$(dirname "${BASH_SOURCE[0]}")/common_gc.sh"     # 从 gcdistill/run_stage{1,2,3}.sh 里
# 干五件事：
#   1) 从自身位置推 M（maliang 根）—— 不写死路径
#   2) 统一环境变量（PYTHONPATH=code/gc25d:code/arope / AROPE_ROOT / 缓存全在 local disk / 分配器 / NCCL / OMP）
#      + 分辨率 GC_RES（默认 832x480；geometry.py 解析，accelerate 拉起的子进程继承）；require_data_res 查数据根的 res
#   3) 选卡：DISTILL_NGPU（默认 8，对应配方的等效 batch）；DISTILL_GPUS 可显式指定物理卡号（并行开发期冒烟：6,7）
#   4) GPU 占用守卫：选中的卡里任意一张 >500 MiB 就拒绝启动（fail-closed）
set -euo pipefail

# ---- 1) 路径 ----
GCD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # <M>/code/gc25d/gcdistill
M="$(cd "$GCD/../../.." && pwd)"                              # <M> = $MALIANG_ROOT
export M GCD
PY="${PY:-$M/.venv/bin/python}"
ACC="${ACC:-$M/.venv/bin/accelerate}"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || command -v python || true)"; fi
if [ ! -x "$ACC" ]; then ACC="$(command -v accelerate || true)"; fi
[ -n "$PY" ]  || { echo "ABORT: python not found (set PY=...)"; exit 1; }
[ -n "$ACC" ] || { echo "ABORT: accelerate not found (pip install accelerate, or set ACC=...)"; exit 1; }
export PY ACC

# ---- 2) 环境（缓存 / 临时文件放在仓库的 .tmp/ 下）----
export PYTHONPATH="$M/code/gc25d:$M/code/arope${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$M/third_party/DiffSynth-Studio" ]; then
  export PYTHONPATH="$PYTHONPATH:$M/third_party/DiffSynth-Studio"
fi
export AROPE_ROOT="$M/code/arope"                             # gc_dit.py / infer.py 按它找 vendored ARoPE
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HOME="${HF_HOME:-$MALIANG_ROOT/.tmp/hf}"
export TMPDIR="$M/.tmp"
export TORCHINDUCTOR_CACHE_DIR="$M/.tmp/inductor"
export TRITON_CACHE_DIR="$M/.tmp/triton"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
unset GM_XCACHE GM_COMPILE GM_SAGEATTN || true
mkdir -p "$TMPDIR"

# ---- 2b) 分辨率：GC_RES 透传给所有子进程（geometry.py 是唯一出处；非法值在这里就停）----
export GC_RES="${GC_RES:-832x480}"
_res_raw="$GC_RES"
# 规范成 geometry.RES_TAG（" 1280X704" ⇒ 1280x704）：下面 require_data_res 的字符串比较、日志都用规范值
GC_RES="$("$PY" -c "import geometry; print(geometry.RES_TAG)")" \
  || { echo "ABORT: GC_RES=$_res_raw 不合法（见 code/gc25d/geometry.py::parse_res）" >&2; exit 1; }
export GC_RES
echo "[res] GC_RES=$GC_RES"

# 数据根的 res（index/prep_args.json，缺失 = 832x480）必须 = GC_RES；python 侧加载前还会再查一遍
require_data_res() {
  local r
  r="$("$PY" -c "import sys, geometry; print(geometry.root_res(sys.argv[1]))" "$1")" || return 1
  if [ "$r" != "$GC_RES" ]; then
    echo "ABORT: 数据根 $1 是 $r 的数据，GC_RES=$GC_RES（设 GC_RES=$r，或 DATA_ROOT 换成 $GC_RES 的数据根）" >&2
    return 1
  fi
}

# ---- 3) 选卡 ----
# PCI_BUS_ID：CUDA_VISIBLE_DEVICES 与守卫里的 `nvidia-smi -i` 一定指同样这几张物理卡。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DISTILL_NGPU="${DISTILL_NGPU:-8}"
case "$DISTILL_NGPU" in
  1|2|3|4|8) : ;;
  *) echo "ABORT: DISTILL_NGPU 只接受 1/2/4/8，得到 $DISTILL_NGPU" >&2; exit 1 ;;
esac
DISTILL_GPUS="${DISTILL_GPUS:-$(seq -s, 0 $((DISTILL_NGPU - 1)))}"
_n_gpus="$(printf '%s\n' "$DISTILL_GPUS" | tr ',' '\n' | grep -c .)"
if [ "$_n_gpus" != "$DISTILL_NGPU" ]; then
  echo "ABORT: DISTILL_GPUS=$DISTILL_GPUS 是 $_n_gpus 张卡，DISTILL_NGPU=$DISTILL_NGPU" >&2; exit 1
fi
DISTILL_NPROC="$DISTILL_NGPU"
export CUDA_VISIBLE_DEVICES="$DISTILL_GPUS"
export DISTILL_GPUS DISTILL_NPROC DISTILL_NGPU

# ---- 4) GPU 占用守卫（上游原样：显式收 nvidia-smi 的退出码；必须恰好回来选中的那几行）----
require_idle_gpus() {
  local thresh="${1:-500}" out rc busy seen want
  command -v nvidia-smi >/dev/null 2>&1 || { echo "ABORT: 没有 nvidia-smi" >&2; return 1; }
  out="$(timeout 30 nvidia-smi -i "$DISTILL_GPUS" --query-gpu=index,memory.used --format=csv,noheader,nounits 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "ABORT: nvidia-smi -i $DISTILL_GPUS 查询失败（退出码 $rc）：" >&2
    printf '       %s\n' "$out" >&2
    return 1
  fi
  seen="$(printf '%s\n' "$out" | awk -F', *' 'NF>=2 {printf "%s,", $1}')"
  want="$DISTILL_GPUS,"
  if [ "$seen" != "$want" ]; then
    echo "ABORT: nvidia-smi 只回了 [${seen%,}]，期望 [$DISTILL_GPUS]；有卡查不到，不启动。" >&2
    return 1
  fi
  busy="$(printf '%s\n' "$out" | awk -F', *' -v t="$thresh" \
            '$2 !~ /^[0-9]+$/ || $2+0 > t+0 {printf "GPU%s=%sMiB ", $1, $2}')"
  if [ -n "$busy" ]; then
    echo "ABORT: GPU $DISTILL_GPUS 上有占用（>${thresh} MiB）: $busy" >&2
    echo "       另一个任务还在跑？先确认是不是自己的。不要去动别人的卡。" >&2
    return 1
  fi
  echo "[guard] GPU $DISTILL_GPUS 空闲（每张 <= ${thresh} MiB），可以启动。"
  return 0
}

# 阈值只允许调**严**，不允许调松（GPU_BUSY_MIB=999999 曾经等于一个偷卡后门）
# GPU_MIN_FREE_GB set (by the auto-runner) => cards may be shared with others'
# jobs; require that much FREE memory on every selected card instead of the card being idle. Unset => original idle guard.
require_free_gpus() {
  local need="$1" out bad
  out="$(timeout 30 nvidia-smi -i "$DISTILL_GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)" || { echo "ABORT: nvidia-smi failed" >&2; return 1; }
  bad="$(printf '%s\n' "$out" | awk -F', *' -v n="$need" '($3-$2)/1024 < n {printf "GPU%s=%.0fGB ", $1, ($3-$2)/1024}')"
  [ -z "$bad" ] || { echo "ABORT: GPU $DISTILL_GPUS free memory below ${need} GB: $bad" >&2; return 1; }
  echo "[guard] GPU $DISTILL_GPUS: each has >= ${need} GB free (shared cards allowed)."
}
if [ -n "${GPU_MIN_FREE_GB:-}" ]; then
  require_free_gpus "$GPU_MIN_FREE_GB" || exit 1
else
GPU_BUSY_MIB_EFF=$(( ${GPU_BUSY_MIB:-500} < 500 ? ${GPU_BUSY_MIB:-500} : 500 ))
require_idle_gpus "$GPU_BUSY_MIB_EFF" || exit 1
fi

# OUT 用绝对路径，避免 runlib.resolve_run_dir 把输出落进代码树
require_abs_out() {
  case "$1" in
    /*) mkdir -p "$(dirname "$1")" ;;
    *) echo "ABORT: OUT=$1 must be an absolute path" >&2; exit 1 ;;
  esac
}
