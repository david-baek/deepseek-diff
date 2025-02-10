# %%

import torch
import einops
import sys

name_to_model_map = {
    "qwen-1.5b": ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "qwen-7b": ["Qwen/Qwen2.5-Math-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    "qwen-14b": ["Qwen/Qwen2.5-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
}

weight_path = "./checkpoints/version_1/qwen-7b_6.pt"
weights = torch.load(weight_path)

base_dec = weights["W_dec"][:, 0, :]
reasoning_dec = weights["W_dec"][:, 1, :]
base_norms = torch.norm(base_dec, p=1, dim=1)
reasoning_norms = torch.norm(reasoning_dec, p=1, dim=1)

relative_norms = reasoning_norms / base_norms
normalized_relative_norms = relative_norms / (1 + relative_norms)

# %%
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

target_name = "contrastive"

target_token_map = {
    "wait" : [" wait", "Wait"],
    "deductive": ["Therefore", "Thus"],
    "alternative": ["Alternatively"],
    "contrastive": ["However", "But"]
}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(name_to_model_map[save_name][0])


base_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][0],
    device=device, 
    n_devices=4,
)

chat_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][1],
    device=device, 
    n_devices=4,
)
hook_point = f"blocks.{base_model.cfg.n_layers // 2}.hook_resid_pre"

# %%
weights = torch.load(weight_path)

import torch
import einops

# === STEP 1. Identify the token IDs for "wait" (or "Wait", or general target tokens) ===
# (Many tokenizers prepend a space for non-BOS words; adjust if needed.)

wait_token_ids = set()
for target_token in target_token_map[target_name]:
    token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    wait_token_ids.update(token_ids)
print("Wait token IDs:", wait_token_ids)

# === STEP 2. Initialize a counter for all features ===
# We need to know the dimensionality of the encoded representation (d_hidden).
# One way is to use the weight matrix; here weights["W_enc"] is assumed to have shape [n_models, d_model, d_hidden].
d_hidden = weights["W_enc"].shape[2]
feature_counts = torch.zeros(d_hidden, dtype=torch.int64)
feature_counts = feature_counts.to(device)

# Optionally track total number of wait tokens found
total_wait_tokens = 0

# === STEP 3. Iterate over each sequence, locate wait tokens, and accumulate feature activations ===
# We process the first 100 sequences in all_tokens.
for batch_idx, tokens in enumerate(all_tokens[:1000]):
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
    
    # === Run the models to obtain cached activations at the chosen hook point ===
    _, cache_A = base_model.run_with_cache(tokens, names_filter=hook_point)
    _, cache_B = chat_model.run_with_cache(tokens, names_filter=hook_point)
    
    # Stack activations from both models along a new dimension ("n_models").
    # Assume each cache entry has shape: [batch, seq_len, d_model].
    acts = torch.stack([cache_A[hook_point], cache_B[hook_point]], dim=1)
    
    # Drop the BOS token (position 0 along the sequence dimension).
    # New shape: [batch, n_models, seq_len-1, d_model]
    acts = acts[:, :, 1:, :]
    
    # Rearrange so that each token becomes an individual “batch” element.
    # New shape: [ (batch * (seq_len-1)), n_models, d_model ]
    acts = einops.rearrange(acts, "batch n_models seq_len d_model -> (batch seq_len) n_models d_model")
    acts = acts.to(device)
    acts = acts.to(weights["W_enc"].dtype)
    
    # === Compute the encoded (sparse) representation using the weight matrix ===
    # x_enc will have shape [ (seq_len-1), d_hidden ].
    x_enc = einops.einsum(
        acts,
        weights["W_enc"],
        "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden"
    )
    
    # === STEP 4. For each wait token in this sequence, count the active features ===
    # Note: because we dropped the BOS token, a token at original position i (with i > 0)
    # corresponds to row (i-1) in x_enc.
    for pos in wait_positions:
        if pos == 0:
            continue  # Skip the BOS token if flagged.
        token_idx = pos - 1  # Adjust for dropped BOS.
        
        # Get the encoded representation for this token.
        token_encoding = x_enc[token_idx]  # Shape: [d_hidden]
        
        # Create a binary mask: 1 if active (>0), else 0.
        active_mask = (token_encoding > 0).long()  # Shape: [d_hidden]
        
        # Accumulate counts per feature.
        feature_counts += active_mask

print(f"\nFound a total of {total_wait_tokens} target tokens in the processed sequences.\n")

all_result_dict = dict()
for i, count in enumerate(feature_counts):
    all_result_dict[i] = count.item()

import json
with open("./results/" + weight_path[2:-3].replace("/","__") + f"_{target_name}.json", 'w') as file:
    json.dump(all_result_dict, file, indent=4)