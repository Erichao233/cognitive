from verl import DataProto
import requests
import torch
import numpy as np
from verl.utils.dialogue.think_answer import parse_think_answer


class URLEnvironment():

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer


    def get_reward_batched(self, data: DataProto):  #batched
        messages_batched = []
        reward_locs = []
        format_ok = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            messages = data_item.non_tensor_batch['messages']
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            messages_batched.append(messages)

            attention_mask = data_item.batch['attention_mask']
            generation_mask = data_item.batch.get('generation_mask', None)
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = attention_mask[prompt_length:].sum()
            if generation_mask is not None:
                response_gen_mask = (attention_mask * generation_mask)[prompt_length:]
                assistant_token_locs = torch.nonzero(response_gen_mask, as_tuple=False).flatten()
                if assistant_token_locs.numel() > 0:
                    reward_locs.append(int(assistant_token_locs[-1].item()))
                else:
                    reward_locs.append(int(valid_response_length - 1))
            else:
                reward_locs.append(int(valid_response_length - 1))

            last_assistant = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    last_assistant = msg.get("content", "") or ""
                    break
            format_ok.append(float(parse_think_answer(last_assistant).ok))

        # reward_batched = requests.post(url, json=payload).json()
        reward_batched = data.non_tensor_batch['emo_point']/100
        reward_batched = np.maximum(reward_batched, 0)
        original_reward_batched = reward_batched.copy()

        # Optional format reward for enforcing <think>/<answer> closed tags.
        fmt_bonus = float(getattr(self.config.actor_rollout_ref.rollout.environment, "format_reward", 0.02))
        fmt_penalty = float(getattr(self.config.actor_rollout_ref.rollout.environment, "format_penalty", 0.0))
        reward_batched = reward_batched + np.where(np.array(format_ok) > 0.5, fmt_bonus, -fmt_penalty)


        original_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        penalized_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            original_reward_tensor[i, reward_locs[i]] = original_reward_batched[i]
            penalized_reward_tensor[i, reward_locs[i]] = reward_batched[i]
        
        return original_reward_tensor, penalized_reward_tensor
