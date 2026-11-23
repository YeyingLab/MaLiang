#!/usr/bin/env bash
# Pick the first GPU with at least N GB free. Source this file, then: CUDA_VISIBLE_DEVICES=$(pick 30)
# If nvidia-smi is unavailable, fall back to GPU 0.
pick() {
  local need="${1:-20}" g=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' -v n="$need" '($3-$2)/1024 >= n {print $1; exit}')
  fi
  echo "${g:-0}"
}
