#!/usr/bin/env bash
set -euo pipefail

# GRPO + multi-turn simulator + RAG (Local Model Version for 3x H20 GPUs)
#
# Usage:
#   bash code/train_grpo_autodl.sh
#
# This version uses LOCAL models for both Simulator and Embedding to avoid API latency.
# Required env:
#   - For LOCAL mode (default): No API key needed
#   - For API mode: Set USE_LOCAL_MODELS=0 and SILICONFLOW_API_KEY
#
# Local model requirements:
#   - Simulator: vLLM server at LOCAL_LLM_BASE_URL (default: http://localhost:8000/v1)
#   - Embedding: bge-m3 loaded via sentence-transformers (auto-downloaded or set LOCAL_EMBEDDING_MODEL_PATH)
#
# Optional env overrides:
#   BASE_MODEL_PATH       - Training model path
#   LOCAL_LLM_BASE_URL    - Local vLLM server URL (default: http://localhost:8000/v1)
#   LOCAL_LLM_MODEL_NAME  - Model name for local vLLM (default: same as BASE_MODEL_PATH basename)
#   LOCAL_EMBEDDING_MODEL_PATH - Local embedding model (default: BAAI/bge-m3)
#   USE_LOCAL_MODELS      - 1 for local (default), 0 for API
#   SMOKE_TEST            - 1 to run minimal test

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${REPO_DIR}/code"

cd "${CODE_DIR}"

# ============ GPU Configuration (3x H20) ============
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
N_GPUS=3

# ============ Local vs API Mode ============
USE_LOCAL_MODELS="${USE_LOCAL_MODELS:-1}"

if [[ "${USE_LOCAL_MODELS}" == "1" ]]; then
  echo "=== Using LOCAL models (no API calls) ==="
  export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://localhost:8000/v1}"
  export SIMULATOR_SERVED_MODEL_NAME="${SIMULATOR_SERVED_MODEL_NAME:-${LOCAL_LLM_MODEL_NAME:-Qwen2.5-7B-Instruct}}"
  export LOCAL_LLM_MODEL_NAME="${LOCAL_LLM_MODEL_NAME:-${SIMULATOR_SERVED_MODEL_NAME}}"
  export LOCAL_EMBEDDING_MODEL_PATH="${LOCAL_EMBEDDING_MODEL_PATH:-BAAI/bge-m3}"
  export LOCAL_EMBEDDING_DEVICE="${LOCAL_EMBEDDING_DEVICE:-cpu}"
  # Point HuggingFace to your existing cache directory
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/hf}"
  export USE_LOCAL_EMBEDDING="1"
  export USE_LOCAL_SIMULATOR="1"
  # Dummy API key to pass validation (not actually used)
  export SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:-dummy_key_for_local_mode}"
else
  echo "=== Using API models (SiliconFlow) ==="
  if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
    echo "Missing env var: SILICONFLOW_API_KEY" >&2
    exit 1
  fi
  export USE_LOCAL_EMBEDDING="0"
  export USE_LOCAL_SIMULATOR="0"
fi

export SILICONFLOW_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1}"
export SILICONFLOW_EMBEDDING_MODEL="${SILICONFLOW_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
export SILICONFLOW_CHAT_MODEL="${SILICONFLOW_CHAT_MODEL:-deepseek-ai/DeepSeek-V3.2}"

# ============ General Settings ============
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RLVER_THINKING="${RLVER_THINKING:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export VERL_VLLM_CACHE_DEBUG="${VERL_VLLM_CACHE_DEBUG:-0}"
export VERL_VLLM_HARD_CLEAR="${VERL_VLLM_HARD_CLEAR:-1}"

export RAG_CACHE_DIR="${RAG_CACHE_DIR:-${REPO_DIR}/rag_cache}"
export SIMULATOR_CACHE_DIR="${SIMULATOR_CACHE_DIR:-${REPO_DIR}/simulator_cache}"
export SIMULATOR_LOG_DIR="${SIMULATOR_LOG_DIR:-${REPO_DIR}/simulator_logs}"
export ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${REPO_DIR}/rollout_logs}"

# ============ Model & Training Parameters ============
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct}"

# Batch size must be divisible by N_GPUS (3). Using 6 for balanced throughput.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
# GRPO group size. Reduced to 2 for memory efficiency (minimum for GRPO).
ROLLOUT_N="${ROLLOUT_N:-2}"
# Sequence lengths - optimized for 8 turn dialogues
# Each turn: ~200 (agent) + ~100 (user) = ~300 tokens
# 8 turns = ~2400 tokens response, plus ~1536 prompt = ~4000 total
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
PER_TURN_LENGTH="${PER_TURN_LENGTH:-200}"
MAX_TURNS="${MAX_TURNS:-8}"
# Memory settings for 3x 96GB H20 (conservative to avoid OOM)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12000}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-6}"
# Sampling
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
# Rewards
FORMAT_REWARD="${FORMAT_REWARD:-0.02}"
FORMAT_PENALTY="${FORMAT_PENALTY:-0.0}"
# Training steps
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# Checkpoint frequency (less frequent to save space)
SAVE_FREQ="${SAVE_FREQ:-100}"
SMOKE_TEST="${SMOKE_TEST:-0}"
RAG_ENABLED="${RAG_ENABLED:-true}"

if [[ "${ROLLOUT_N}" -lt 2 ]]; then
  echo "ROLLOUT_N must be >= 2 for GRPO; got ${ROLLOUT_N}" >&2
  exit 1
fi

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

BASE_MODEL_PATH="$(resolve_hf_snapshot "${BASE_MODEL_PATH}")"
if [[ ! -f "${BASE_MODEL_PATH}/config.json" ]]; then
  echo "BASE_MODEL_PATH does not look like a HF model dir (missing config.json): ${BASE_MODEL_PATH}" >&2
  echo "If you are using HF cache, point BASE_MODEL_PATH to snapshots/<hash>." >&2
  exit 1
fi

# Resolve local embedding snapshot if user points to HF cache directory (models--*/snapshots/*).
if [[ "${USE_LOCAL_MODELS}" == "1" ]]; then
  if [[ -d "${LOCAL_EMBEDDING_MODEL_PATH}" ]]; then
    LOCAL_EMBEDDING_MODEL_PATH="$(resolve_hf_snapshot "${LOCAL_EMBEDDING_MODEL_PATH}")"
    export LOCAL_EMBEDDING_MODEL_PATH
  fi
fi

RAG_PATH="${RAG_PATH:-${REPO_DIR}/data/strategy_cards.jsonl}"
if [[ ! -f "${RAG_PATH}" ]]; then
  echo "Missing strategy cards file: ${RAG_PATH}" >&2
  exit 1
fi

if [[ -z "${CKPT_DIR:-}" ]]; then
  CKPT_DIR="${REPO_DIR}/ckpts/rlver_grpo_$(date +%m%d_%H%M%S)"
fi
mkdir -p "${CKPT_DIR}"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  echo "SMOKE_TEST=1: overriding to a tiny run" >&2
  TOTAL_TRAINING_STEPS=2
  TOTAL_EPOCHS=1
  TRAIN_BATCH_SIZE=3
  ROLLOUT_N=2
  MAX_PROMPT_LENGTH=512
  MAX_RESPONSE_LENGTH=256
  PER_TURN_LENGTH=64
  MAX_TURNS=2
  GPU_MEMORY_UTILIZATION=0.45
  ENABLE_CHUNKED_PREFILL=false
  MAX_NUM_BATCHED_TOKENS=2048
  PPO_MAX_TOKEN_LEN_PER_GPU=4096
  PPO_MINI_BATCH_SIZE=3
  SAVE_FREQ=1
  RAG_ENABLED=true
fi

VIRTUAL_DATASET_SIZE="${VIRTUAL_DATASET_SIZE:-$(( TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE ))}"
VAL_VIRTUAL_DATASET_SIZE="${VAL_VIRTUAL_DATASET_SIZE:-$(( TRAIN_BATCH_SIZE * 4 ))}"

# Validate batch size divisibility
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
  echo "train_batch_size must be divisible by number of visible GPUs. train_batch_size=${TRAIN_BATCH_SIZE}, N_GPUS=${N_GPUS}" >&2
  exit 1
fi

echo "============================================="
echo "RLVER Training Configuration"
echo "============================================="
echo "Repo: ${REPO_DIR}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "BASE_MODEL_PATH: ${BASE_MODEL_PATH}"
echo "GPUs: ${N_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Batch: train=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}"
echo "Lengths: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, per_turn=${PER_TURN_LENGTH}, max_turns=${MAX_TURNS}"
echo "Local Models: USE_LOCAL_MODELS=${USE_LOCAL_MODELS}"
if [[ "${USE_LOCAL_MODELS}" == "1" ]]; then
  echo "  Simulator: ${LOCAL_LLM_BASE_URL} (model: ${LOCAL_LLM_MODEL_NAME})"
  echo "  Embedding: ${LOCAL_EMBEDDING_MODEL_PATH}"
  echo "  Embedding device: ${LOCAL_EMBEDDING_DEVICE}"
fi
echo "Save frequency: every ${SAVE_FREQ} steps"
echo "============================================="

if [[ "${USE_LOCAL_MODELS}" == "1" ]]; then
  # Fail fast if local simulator server is not reachable.
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsS --max-time 2 "${LOCAL_LLM_BASE_URL}/models" >/dev/null; then
      echo "Local simulator server is not reachable at ${LOCAL_LLM_BASE_URL}." >&2
      echo "Start it with: bash code/start_local_simulator.sh" >&2
      exit 1
    fi
  fi

  # Ensure local embedding dependency exists.
  python - <<'PY'
import sys
try:
    import sentence_transformers  # noqa: F401
except Exception as e:
    print("Missing dependency for local embeddings: sentence-transformers", file=sys.stderr)
    print("Install with: pip install sentence-transformers", file=sys.stderr)
    raise
PY
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.logger='[console]' \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  +trainer.val_before_train=False \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.save_rollout=False \
  +data.virtual_dataset_size="${VIRTUAL_DATASET_SIZE}" \
  +data.val_virtual_dataset_size="${VAL_VIRTUAL_DATASET_SIZE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
  +actor_rollout_ref.rollout.enable_prefix_caching=False \
  +actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
  +actor_rollout_ref.rollout.trust_remote_code=True \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  +actor_rollout_ref.rollout.environment.name=url_environment \
  +actor_rollout_ref.rollout.environment.per_turn_length="${PER_TURN_LENGTH}" \
  +actor_rollout_ref.rollout.environment.max_turns="${MAX_TURNS}" \
  +actor_rollout_ref.rollout.environment.format_reward="${FORMAT_REWARD}" \
  +actor_rollout_ref.rollout.environment.format_penalty="${FORMAT_PENALTY}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  +actor_rollout_ref.actor.use_loss_generation_mask=True \
  actor_rollout_ref.rollout.rag.enabled="${RAG_ENABLED}" \
  actor_rollout_ref.rollout.rag.path="${RAG_PATH}" \
  actor_rollout_ref.rollout.rag.top_k=3
