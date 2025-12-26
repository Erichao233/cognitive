# RLVER Codebase Notes

This repository is a forked/customized copy of `verl` with RLVER-specific data and rollout logic layered on top.

## Overall Architecture and Execution Flow

Entry points:
- `code/train_rlver.sh` runs `python3 -m verl.trainer.main_ppo` with Hydra overrides.
- `code/verl/trainer/main_ppo.py` is the main RL training entry.

High-level flow:
1. `main_ppo.main()` loads config and starts Ray (`run_ppo` -> `main_task`).
2. `main_task` sets worker classes (FSDP vs Megatron), role mappings, resource pools, and reward handling.
3. `RayPPOTrainer` is instantiated (`code/verl/trainer/ppo/ray_trainer.py`) and `init_workers()` launches actor/rollout/ref/critic/RM workers.
4. `RayPPOTrainer.fit()` runs the PPO loop:
   - sample prompts -> rollout -> compute reward -> logprobs -> values -> advantages
   - update critic then actor
   - log metrics and checkpoint

## Data Flow (Training Pipeline)

Key data container:
- `code/verl/protocol.py` defines `DataProto` (tensor batch + non-tensor batch + meta_info).

Dataset and batching:
- `code/verl/utils/dataset/rl_dataset.py` produces dicts, then `collate_fn` stacks them into tensors + non-tensors.
- `RayPPOTrainer._create_dataloader()` always uses `VirtualRLHFDataset` (customized here) and wraps it in a `StatefulDataLoader`.

Rollout and reward:
- `ActorRolloutRefWorker.generate_sequences()` (`code/verl/workers/fsdp_workers.py`) calls vLLM rollouts.
- For RLVER, `rollout.name=vllm_multi_turn_via_chat` uses:
  - `code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` or
  - `code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_think.py`.
- These multi-turn rollouts:
  - read `raw_prompt` and `simulator` from `DataProto.non_tensor_batch`
  - run a multi-turn chat loop with the PlayerSimulator
  - return `messages`, `emo_point`, and a `generation_mask` (assistant-only tokens).
- Reward is computed by `URLEnvironment.get_reward_batched()` (`code/verl/environments/url_environment.py`), which maps `emo_point` into a token-level reward at the final response token.
- RAG retrieval is injected inside the multi-turn rollouts using a vector index (see `code/verl/utils/retrieval/strategy_cards.py`), and logs `rag_query`, `rag_card_ids`, and `rag_scores` in `non_tensor_batch`.

## Key Modules and Responsibilities

Core trainer and orchestration:
- `code/verl/trainer/main_ppo.py`: entry point, worker wiring, reward/environment selection.
- `code/verl/trainer/ppo/ray_trainer.py`: PPO loop, validation, metric logging, checkpointing.
- `code/verl/single_controller/ray/*`: Ray worker groups and resource pools.

Actor and critic:
- `code/verl/workers/fsdp_workers.py`: constructs actor/rollout/ref/critic workers and exposes RPC calls.
- `code/verl/workers/actor/dp_actor.py`: PPO policy update, logprob recompute, entropy/kl losses.
- `code/verl/workers/critic/dp_critic.py`: value estimation and critic updates.

Rollout:
- `code/verl/workers/rollout/vllm_rollout/vllm_rollout.py`: standard vLLM rollout (single-turn).
- `code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: RLVER multi-turn via chat.
- `code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_think.py`: multi-turn variant that strips `<think>` before simulator response.
- `code/verl/utils/retrieval/strategy_cards.py`: vector retrieval over `data/strategy_cards.jsonl` using SiliconFlow embeddings.

RLVER simulator and prompt:
- `code/verl/workers/rollout/vllm_rollout/hard_player_simulator_dsv3.py`: user simulator with emotion state.
- `code/verl/workers/rollout/vllm_rollout/system_prompt.py`: training system prompts.
- `data/train_profile.jsonl`, `data/test_profile.jsonl`: simulator profiles.

## Training Objectives and Losses

Policy loss:
- `DataParallelPPOActor.update_policy()` (`code/verl/workers/actor/dp_actor.py`):
  - PPO clipped policy loss + entropy regularization.
  - Optional KL loss to ref policy (`use_kl_loss`).
  - Uses `generation_mask` when `use_loss_generation_mask=True` to train only assistant tokens in multi-turn chats.

Value loss:
- `DataParallelPPOCritic.update_critic()` (`code/verl/workers/critic/dp_critic.py`):
  - PPO-style value clipping loss.

Reward shaping and advantage:
- `apply_kl_penalty()` in `code/verl/trainer/ppo/ray_trainer.py` applies token-level KL penalty if ref policy is enabled.
- `compute_advantage()` supports GAE, GRPO, REINFORCE++, REMAX, and RLOO.
- GRPO (adv_estimator=`grpo`) bypasses critic updates and uses outcome-based advantages.

## Paper-Specific vs Generic VERL Components

Paper-specific (RLVER-focused):
- `code/train_rlver.sh`: RLVER training settings (multi-turn rollout + environment reward).
- `code/verl/utils/dataset/rl_dataset.py`: custom dataset that instantiates `PlayerSimulator`, builds `raw_prompt`, and emits `simulator`.
- `code/verl/trainer/ppo/ray_trainer.py`: hard-coded `use_virtual_dataset=True` and optional rollout logging (`trainer.save_rollout`).
- `code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: multi-turn chat loop with simulator, `generation_mask`, and `emo_point`.
- `code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_think.py`: `<think>`-aware variant.
- `code/verl/environments/url_environment.py`: reward computed from `emo_point`.
- `code/verl/workers/rollout/vllm_rollout/system_prompt.py`, `code/verl/workers/rollout/vllm_rollout/hard_player_simulator_dsv3.py`, `data/*.jsonl`: simulator and prompts.

Generic VERL abstractions (framework):
- `code/verl/trainer/ppo/ray_trainer.py` (core PPO orchestration, metrics, checkpointing).
- `code/verl/workers/fsdp_workers.py`, `code/verl/workers/actor/*`, `code/verl/workers/critic/*`.
- `code/verl/protocol.py` (`DataProto`).
- `code/verl/utils/*` (tokenization, KL, advantage, dataset utilities, logging).
- `code/verl/trainer/config/ppo_trainer.yaml` (baseline PPO config, overridden by RLVER script).

## Assumptions and Placeholders to Resolve Before Running

- Environment variables are required for API calls and logging:
  - `SILICONFLOW_API_KEY`, `SILICONFLOW_BASE_URL`, `SILICONFLOW_EMBEDDING_MODEL`, `SILICONFLOW_CHAT_MODEL`
  - `RAG_CACHE_DIR`, `SIMULATOR_CACHE_DIR`, `SIMULATOR_LOG_DIR`, `ROLLOUT_LOG_DIR`
- `code/train_rlver.sh` still uses placeholders like `YOUR_DIR_TO_SAVE_CKPTS` and assumes a local Ray cluster.
