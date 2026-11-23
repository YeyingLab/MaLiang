#!/usr/bin/env bash
MALIANG_ROOT="${MALIANG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# FLUX.2 klein-4B (Apache-2.0, ungated) env + weights for experiment 6's first frames (map2frame).
C=$MALIANG_ROOT/experiments
export UV_CACHE_DIR=$C/.cache/uv UV_PYTHON_INSTALL_DIR=$C/tools/python HF_HUB_CACHE=$C/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1
$C/tools/uv venv -q --python 3.11 $C/envs/flux
$C/tools/uv pip install -q --python $C/envs/flux/bin/python torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
$C/tools/uv pip install -q --python $C/envs/flux/bin/python "git+https://github.com/huggingface/diffusers" transformers accelerate \
  safetensors sentencepiece opencv-python-headless pillow hf_transfer huggingface_hub
$C/envs/flux/bin/python -c 'import diffusers; from diffusers import Flux2KleinPipeline; print("diffusers", diffusers.__version__)'
$C/envs/flux/bin/hf download black-forest-labs/FLUX.2-klein-4B --revision e7b7dc27f91deacad38e78976d1f2b499d76a294
echo FLUX_SETUP_DONE
