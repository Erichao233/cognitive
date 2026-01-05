#!/usr/bin/env bash
set -euo pipefail

# Start local vLLM server for Simulator
#
# Usage:
#   bash code/start_local_simulator.sh
#
# This script starts a vLLM server that serves as the Simulator LLM.
# The training script will connect to this server when USE_LOCAL_MODELS=1.
#
# Prerequisites:
#   - vLLM installed: pip install vllm
#   - Model downloaded or accessible from HuggingFace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ============ Configuration ============
# Prefer dedicating a GPU to the simulator to avoid stalling training.
# Default behavior:
#   - If SIMULATOR_GPU is set, use it.
#   - Else, if 4+ GPUs are detected, use GPU 3 (assuming training uses 0,1,2).
#   - Else, fall back to GPU 0.
if [[ -n "${SIMULATOR_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SIMULATOR_GPU}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    GPU_COUNT="0"
  fi
  if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT >= 4 )); then
    export CUDA_VISIBLE_DEVICES="3"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

# Model path - same as training base model by default
MODEL_PATH="${SIMULATOR_MODEL_PATH:-/root/autodl-tmp/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct}"

# Server settings
PORT="${SIMULATOR_PORT:-8000}"
HOST="${SIMULATOR_HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${SIMULATOR_GPU_MEM:-0.15}"  # Conservative to leave room for training
MAX_MODEL_LEN="${SIMULATOR_MAX_LEN:-4096}"
SERVED_MODEL_NAME="${SIMULATOR_SERVED_MODEL_NAME:-${LOCAL_LLM_MODEL_NAME:-Qwen2.5-7B-Instruct}}"

# Resolve HuggingFace snapshot path if needed
resolve_hf_snapshot() {
  local p="$1"
  if [[ -f "${p}/config.json" ]]; then
    echo "${p}"
    return 0
  fi
  if [[ -d "${p}/snapshots" ]]; then
    local snap
    snap="$(ls -1dt "${p}/snapshots/"* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${snap}" && -f "${snap}/config.json" ]]; then
      echo "${snap}"
      return 0
    fi
  fi
  echo "${p}"
  return 0
}

MODEL_PATH="$(resolve_hf_snapshot "${MODEL_PATH}")"

echo "============================================="
echo "Starting Local Simulator vLLM Server"
echo "============================================="
echo "Model: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Port: ${PORT}"
echo "GPU Memory: ${GPU_MEMORY_UTILIZATION}"
echo "============================================="
echo ""
echo "Training script should set:"
echo "  export USE_LOCAL_MODELS=1"
echo "  export LOCAL_LLM_BASE_URL=http://localhost:${PORT}/v1"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code \
  --disable-log-requests
