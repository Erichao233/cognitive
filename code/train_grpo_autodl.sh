#!/usr/bin/env bash
set -euo pipefail

# GRPO + multi-turn simulator + RAG (AutoDL-friendly single-node runner)
#
# Usage:
#   bash code/train_grpo_autodl.sh
#
# Required env (must be set):
#   SILICONFLOW_API_KEY
#
# Optional env overrides:
#   BASE_MODEL_PATH (default: /root/autodl-tmp/cache/hf/hub/models--Qwen--Qwen3-8B)
#   CKPT_DIR (default: <repo>/ckpts/rlver_grpo_<timestamp>)
#   TRAIN_BATCH_SIZE (default: 8)
#   ROLLOUT_N (default: 2)                 # must be >= 2 for GRPO to make sense
#   MAX_PROMPT_LENGTH (default: 2048)
#   MAX_RESPONSE_LENGTH (default: 1024)
#   PER_TURN_LENGTH (default: 256)
#   MAX_TURNS (default: 8)
#   GPU_MEMORY_UTILIZATION (default: 0.65)
#   ROLLOUT_TEMPERATURE (default: 1.0)      # set 0.0 for deterministic debug runs
#   ROLLOUT_TOP_P (default: 1.0)
#   TOTAL_TRAINING_STEPS (default: 200)
#   SMOKE_TEST (default: 0)                # set 1 to run 2 steps quickly
#
# Notes:
# - With algorithm.adv_estimator=grpo, this repo will NOT create/init the critic worker/model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${REPO_DIR}/code"

cd "${CODE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RLVER_THINKING="${RLVER_THINKING:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export VERL_VLLM_CACHE_DEBUG="${VERL_VLLM_CACHE_DEBUG:-1}"

export SILICONFLOW_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1}"
export SILICONFLOW_EMBEDDING_MODEL="${SILICONFLOW_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
export SILICONFLOW_CHAT_MODEL="${SILICONFLOW_CHAT_MODEL:-deepseek-ai/DeepSeek-V3.2}"

if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo "Missing env var: SILICONFLOW_API_KEY" >&2
  exit 1
fi

export RAG_CACHE_DIR="${RAG_CACHE_DIR:-${REPO_DIR}/rag_cache}"
export SIMULATOR_CACHE_DIR="${SIMULATOR_CACHE_DIR:-${REPO_DIR}/simulator_cache}"
export SIMULATOR_LOG_DIR="${SIMULATOR_LOG_DIR:-${REPO_DIR}/simulator_logs}"
export ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${REPO_DIR}/rollout_logs}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
PER_TURN_LENGTH="${PER_TURN_LENGTH:-256}"
MAX_TURNS="${MAX_TURNS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
FORMAT_REWARD="${FORMAT_REWARD:-0.02}"
FORMAT_PENALTY="${FORMAT_PENALTY:-0.0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
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
  TRAIN_BATCH_SIZE=4
  ROLLOUT_N=2
  MAX_PROMPT_LENGTH=1024
  MAX_RESPONSE_LENGTH=512
  PER_TURN_LENGTH=128
  MAX_TURNS=2
  GPU_MEMORY_UTILIZATION=0.5
  RAG_ENABLED=false
fi

# DataProto is split across DP world_size; generation requires train_batch_size divisible by n_gpus.
N_GPUS=4
if [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  N_GPUS=$(( $(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}') ))
elif [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  N_GPUS=1
fi
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
  echo "train_batch_size must be divisible by number of visible GPUs. train_batch_size=${TRAIN_BATCH_SIZE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

echo "Repo: ${REPO_DIR}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "BASE_MODEL_PATH: ${BASE_MODEL_PATH}"
echo "ROLLOUT_N: ${ROLLOUT_N}"
echo "Lengths: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, per_turn=${PER_TURN_LENGTH}, max_turns=${MAX_TURNS}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.logger='[console]' \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  +trainer.val_before_train=False \
  trainer.save_freq=10 \
  trainer.save_rollout=True \
  +data.virtual_dataset_size=32000 \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
  +actor_rollout_ref.rollout.enable_prefix_caching=False \
  +actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
  +actor_rollout_ref.rollout.trust_remote_code=True \
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
