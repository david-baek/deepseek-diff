# %%

import argparse

parser = argparse.ArgumentParser(description="Run ablation experiments on DeepSeek models.")
parser.add_argument("--half_ablate", type=int, required=True)
parser.add_argument("--frac", type=float, required=True)
parser.add_argument("--target_name", type=str, required=True, choices=["wait", "deductive", "alternative", "contrastive"])

args = parser.parse_args()
frac = args.frac

import torch
import einops
import sys

name_to_model_map = {
    "qwen-1.5b": ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "qwen-7b": ["Qwen/Qwen2.5-Math-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    "qwen-14b": ["Qwen/Qwen2.5-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
}

target_name = args.target_name
half_ablate = False if args.half_ablate == 0 else True
amplify = False

target_token_map = {
    "wait" : [" Wait"],
    "deductive": [" Therefore", " Thus"],
    "alternative": [" Alternatively"],
    "contrastive": [" However", " But"]
}


weight_path = "../checkpoints/version_1/qwen-7b_13.pt"
weights = torch.load(weight_path)

base_dec = weights["W_dec"][:, 0, :]
reasoning_dec = weights["W_dec"][:, 1, :]
base_norms = torch.norm(base_dec, p=1, dim=1)
reasoning_norms = torch.norm(reasoning_dec, p=1, dim=1)

relative_norms = reasoning_norms / base_norms
normalized_relative_norms = relative_norms / (1 + relative_norms)


# %%

new_filename = "../results/" + weight_path[3:-3].replace("/","__") + f"_{target_name}.json"

import json
with open(new_filename, 'r') as f:
    data = json.load(f)


active_features = {int(key): data[key] for key in data.keys() if data[key] > 0}

import numpy as np
# Compute the frequency threshold such that only the top frac quantile of active features are selected.
freq_values = list(active_features.values())
quantile_threshold = np.quantile(freq_values, 1 - frac)

all_active_features = list(active_features.keys())
ablate_features = []

for feature in all_active_features:
    # Only consider features in the top frac quantile by firing frequency.
    if active_features[feature] >= quantile_threshold:
        rel_norm = reasoning_norms[feature] / base_norms[feature]
        normalized_relative_norm = rel_norm / (1 + rel_norm)
        if normalized_relative_norm > 0.5:
            ablate_features.append(feature)

print(f"{len(ablate_features)} features ablated")
print(ablate_features)

import random
import numpy as np
random.seed(49)
np.random.seed(49)
random_ablate_features = random.sample(list(range(32768)), len(ablate_features))
print(random_ablate_features)


# %%
import sys
sys.path.append("../")

from crosscoder_diff.utils import load_open_reasoning_tokens
all_tokens = load_open_reasoning_tokens()


# %%
from transformer_lens import HookedTransformer, ActivationCache

import os
os.environ['HF_HOME'] = '/om2/user/dbaek/.cache/'

device = "cuda" if torch.cuda.is_available() else "cpu"

save_name = "qwen-7b"

name_to_model_map = {
    "qwen-1.5b": ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "qwen-7b": ["Qwen/Qwen2.5-Math-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    "qwen-14b": ["Qwen/Qwen2.5-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
}


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(name_to_model_map[save_name][0])


base_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][0],
    device=device, 
    n_devices=2,
)

chat_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][1],
    device=device,
    n_devices=2,
)
hook_point = f"blocks.{base_model.cfg.n_layers // 2}.hook_resid_pre"


# %%
wait_token_ids = set()
for target_token in target_token_map[target_name]:
    token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    wait_token_ids.update(token_ids)
print("Wait token IDs:", wait_token_ids)

# %%
amplify_scale = 2.0               # multiply these features by 3.0

# For quick lookup
ablate_features = set(ablate_features)

def make_hook_fn(act_A, act_B, pos, model_idx):
    act_all = torch.stack([act_A, act_B], dim=1)
    act_all = act_all[:, :, :pos, :]
    new_act_all = act_all.clone()

    act_all = act_all[:, :, 1:, :] # Discard BOS
    
    # Rearrange so that each token becomes an individual “batch” element.
    # New shape: [ (batch * (seq_len-1)), n_models, d_model ]
    act_all = einops.rearrange(act_all, "batch n_models seq_len d_model -> (batch seq_len) n_models d_model")
    act_all = act_all.to(device)
    act_all = act_all.to(weights["W_enc"].dtype)
    
    # -------- Encode --------
    # x_enc: shape [d_hidden]
    # (batch dimension is "1" here, effectively)
    x_enc = einops.einsum(
        act_all,  # shape [1, n_models, d_model]
        weights["W_enc"],                # shape [n_models, d_model, d_hidden]
        "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden"
    )  # shape [d_hidden]
    x_enc = torch.nn.functional.relu(x_enc + weights["b_enc"])
    
    # -------- Intervene (ablate/amplify) --------
    # Option 1: Zero out some features
    if amplify:
        x_enc[:, list(ablate_features)] *= amplify_scale
    else:
        if half_ablate:
            x_enc[:, list(ablate_features)] = 0.0
        else:
            x_enc[:, list(random_ablate_features)] = 0.0
    # Option 2: Multiply some features
#    x_enc[list(amplify_features)] *= amplify_scale

    # -------- Decode --------
    # x_dec: shape [n_models, d_model]
    x_dec = einops.einsum(
        x_enc,  # shape [1, d_hidden]
        weights["W_dec"],               # shape [d_hidden, n_models, d_model]
        "b dh, dh nm dm -> b nm dm"
    )  # shape [n_models, d_model]
    x_dec = x_dec + weights["b_dec"]

    recon_act = einops.rearrange(x_dec, "seq_len n_models d_model -> 1 n_models seq_len d_model")
    new_act_all[:, :, 1:, :] = recon_act

    def hook_fn(value, hook):
        return new_act_all[:, model_idx, :, :]
    return hook_fn


total_wait_tokens = 0

orig_base_logit_list = []
orig_chat_logit_list = []
new_base_logit_list = []
new_chat_logit_list = []
for idx, tokens in enumerate(all_tokens[:100]):
    torch.cuda.empty_cache()
    # Convert tokens to a Python list for easy comparison.
    tokens_list = tokens.tolist()
    
    # Find positions in the original sequence where the token is one of our wait tokens.
    wait_positions = [i for i, token_id in enumerate(tokens_list) if token_id in wait_token_ids]
    
    if not wait_positions:
        continue  # Skip sequences with no wait token.
    
    total_wait_tokens += len(wait_positions)
    
    # Ensure tokens has a batch dimension.
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)  # Now shape [1, seq_length]

    with torch.no_grad():
        # Run the base model to get the original logits.
        original_logits = base_model(tokens)

    with torch.no_grad():
        # Get the chat model activations for the same tokens.
        chat_logits = chat_model(tokens)

    print(original_logits.shape, chat_logits.shape)

    with torch.no_grad():
        # === Run the models to obtain cached activations at the chosen hook point ===
        _, cache_A = base_model.run_with_cache(tokens, names_filter=hook_point)
        _, cache_B = chat_model.run_with_cache(tokens, names_filter=hook_point)

    act_A = cache_A[hook_point]
    act_B = cache_B[hook_point]

    for pos in wait_positions:
        torch.cuda.empty_cache()
        token_id = tokens[0, pos].item()
        print(tokenizer.decode(tokens[0,max(pos-30,0):(pos+1)]))
        sys.stdout.flush()
        with torch.no_grad():
            logits_A = base_model.run_with_hooks(tokens[:, :pos], fwd_hooks=[(hook_point, make_hook_fn(act_A, act_B, pos, 0))])
            logits_B = chat_model.run_with_hooks(tokens[:, :pos], fwd_hooks=[(hook_point, make_hook_fn(act_A, act_B, pos, 1))])

            orig_base_logit_list.append(original_logits[0][pos-1][token_id].item())
            orig_chat_logit_list.append(chat_logits[0][pos-1][token_id].item())
            new_base_logit_list.append(logits_A[0][-1][token_id].item())
            new_chat_logit_list.append(logits_B[0][-1][token_id].item())

        total_wait_tokens += 1
        if total_wait_tokens >= 100:
            break
    if total_wait_tokens >= 100:
        break

final_result_dict = {
    "orig_base_logit": orig_base_logit_list,
    "orig_chat_logit": orig_chat_logit_list,
    "new_base_logit": new_base_logit_list,
    "new_chat_logit": new_chat_logit_list,
    "ablate_features": list(ablate_features) if half_ablate else list(random_ablate_features),
}

import json
with open(f"../results/{save_name}_{target_name}_{"half" if half_ablate else "full"}_{"amp" if amplify else "no"}_{str(frac)}_logits.json", 'w') as f:
    json.dump(final_result_dict, f)
    
