# %%

import torch
import einops
import sys

name_to_model_map = {
    "qwen-1.5b": ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "qwen-7b": ["Qwen/Qwen2.5-Math-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    "qwen-14b": ["Qwen/Qwen2.5-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
}

weight_path = "./checkpoints/version_2/qwen-14b_6.pt"
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

save_name = "qwen-14b"

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
    n_devices=6,
)

chat_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][1],
    device=device, 
    n_devices=6,
)
hook_point = f"blocks.{base_model.cfg.n_layers // 2}.hook_resid_pre"

# %%

top_feat_list = torch.topk(normalized_relative_norms, 20)[1]
reasoning_result_dict = dict()

## Check unique features of distilled model
for feat_idx in top_feat_list.tolist():
    torch.cuda.empty_cache()
    # List to store the top 10 activations per sequence.
    # Each entry is a tuple: (activation_value, batch_index, token_index)
    results = []

    # Process each sequence (i.e. each row) individually.
    for batch_idx, tokens in enumerate(all_tokens[:100]):
        # If tokens is 1D (shape [seq_length]), add a batch dimension.
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)  # Now shape becomes [1, seq_length]
        
        # Run the models to obtain cached activations at the hook point.
        _, cache_A = base_model.run_with_cache(tokens, names_filter=hook_point)
        _, cache_B = chat_model.run_with_cache(tokens, names_filter=hook_point)
        
        # Stack the two models’ activations along a new dimension "n_models".
        # Assume each cache entry has shape: [batch, seq_len, d_model]  
        # → stacked shape: [batch, n_models, seq_len, d_model]
        acts = torch.stack([cache_A[hook_point], cache_B[hook_point]], dim=1)
        
        # Drop the BOS token (position 0) along the sequence dimension.
        # New shape becomes: [batch, n_models, seq_len-1, d_model]
        acts = acts[:, :, 1:, :]
        
        # Rearrange so that each token becomes an individual “batch” element.
        # Now shape: [ (batch * (seq_len-1)), n_models, d_model ]
        acts = einops.rearrange(acts, "batch n_models seq_len d_model -> (batch seq_len) n_models d_model")
        acts = acts.to(device)
        acts = acts.to(weights["W_enc"].dtype)
        
        # Compute the encoded representation using the weight matrix.
        # The einsum combines the activations from the n_models dimension.
        # Resulting shape: [ (seq_len-1), d_hidden ]
        x_enc = einops.einsum(
            acts,
            weights["W_enc"],
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden"
        )
        
        # Extract the activation values for the feature of interest.
        # feat_values has shape: [ (seq_len-1) ], where each index corresponds to token index (j+1)
        # in the original sequence (since BOS was dropped).
        feat_values = x_enc[:, feat_idx]
        
        # Instead of storing all token activations from this sequence,
        # use torch.topk to keep only the top 10 activations.
        k = min(10, feat_values.size(0))  # In case the sequence is very short.
        row_topk_values, row_topk_indices = torch.topk(feat_values, k=k)
        
        # Save these top activations along with their positions.
        # (Add 1 to token indices to map back to the original tokens, because BOS was dropped.)
        for token_idx_offset, activation in zip(row_topk_indices.tolist(), row_topk_values.tolist()):
            results.append((activation, batch_idx, token_idx_offset + 1))
        
        if batch_idx % 20 == 0:
            print(batch_idx)

    # After processing all sequences, `results` contains at most 10 entries per row.
    # Now sort them in descending order by activation to obtain the overall top 10.
    top10 = sorted(results, key=lambda x: x[0], reverse=True)[:10]

    reasoning_result_dict[feat_idx] = []
    # Print the overall top 10 token positions with context from the original tokens.
    for rank, (activation, batch_idx, token_idx) in enumerate(top10, start=1):
        # Define a context window, e.g., 20 tokens before the target token.
        context_start = max(token_idx - 100, 0)
        context_tokens = all_tokens[batch_idx, context_start: token_idx + 1]
        decoded_context = tokenizer.decode(context_tokens.tolist())
        
        print(f"Rank {rank}: Batch {batch_idx}, Token index {token_idx}, Activation {activation:.4f}\n"
            f"Context: {decoded_context}\n")
        reasoning_result_dict[feat_idx].append({
            "rank": rank,
            "batch_idx": batch_idx,
            "activation": activation,
            "token_index": token_idx,
            "context": decoded_context
        })
    
    sys.stdout.flush()

# %%
top_feat_list = torch.topk(1-normalized_relative_norms, 20)[1]

base_result_dict = dict()
## Check unique features of base model
for feat_idx in top_feat_list.tolist():
    torch.cuda.empty_cache()
    # List to store the top 10 activations per sequence.
    # Each entry is a tuple: (activation_value, batch_index, token_index)
    results = []

    # Process each sequence (i.e. each row) individually.
    for batch_idx, tokens in enumerate(all_tokens[:100]):
        # If tokens is 1D (shape [seq_length]), add a batch dimension.
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)  # Now shape becomes [1, seq_length]
        
        # Run the models to obtain cached activations at the hook point.
        _, cache_A = base_model.run_with_cache(tokens, names_filter=hook_point)
        _, cache_B = chat_model.run_with_cache(tokens, names_filter=hook_point)
        
        # Stack the two models’ activations along a new dimension "n_models".
        # Assume each cache entry has shape: [batch, seq_len, d_model]  
        # → stacked shape: [batch, n_models, seq_len, d_model]
        acts = torch.stack([cache_A[hook_point], cache_B[hook_point]], dim=1)
        
        # Drop the BOS token (position 0) along the sequence dimension.
        # New shape becomes: [batch, n_models, seq_len-1, d_model]
        acts = acts[:, :, 1:, :]
        
        # Rearrange so that each token becomes an individual “batch” element.
        # Now shape: [ (batch * (seq_len-1)), n_models, d_model ]
        acts = einops.rearrange(acts, "batch n_models seq_len d_model -> (batch seq_len) n_models d_model")
        acts = acts.to(device)
        acts = acts.to(weights["W_enc"].dtype)
        
        # Compute the encoded representation using the weight matrix.
        # The einsum combines the activations from the n_models dimension.
        # Resulting shape: [ (seq_len-1), d_hidden ]
        x_enc = einops.einsum(
            acts,
            weights["W_enc"],
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden"
        )
        
        # Extract the activation values for the feature of interest.
        # feat_values has shape: [ (seq_len-1) ], where each index corresponds to token index (j+1)
        # in the original sequence (since BOS was dropped).
        feat_values = x_enc[:, feat_idx]
        
        # Instead of storing all token activations from this sequence,
        # use torch.topk to keep only the top 10 activations.
        k = min(10, feat_values.size(0))  # In case the sequence is very short.
        row_topk_values, row_topk_indices = torch.topk(feat_values, k=k)
        
        # Save these top activations along with their positions.
        # (Add 1 to token indices to map back to the original tokens, because BOS was dropped.)
        for token_idx_offset, activation in zip(row_topk_indices.tolist(), row_topk_values.tolist()):
            results.append((activation, batch_idx, token_idx_offset + 1))
        
        if batch_idx % 20 == 0:
            print(batch_idx)

    # After processing all sequences, `results` contains at most 10 entries per row.
    # Now sort them in descending order by activation to obtain the overall top 10.
    top10 = sorted(results, key=lambda x: x[0], reverse=True)[:10]

    base_result_dict[feat_idx] = []
    # Print the overall top 10 token positions with context from the original tokens.
    for rank, (activation, batch_idx, token_idx) in enumerate(top10, start=1):
        # Define a context window, e.g., 20 tokens before the target token.
        context_start = max(token_idx - 100, 0)
        context_tokens = all_tokens[batch_idx, context_start: token_idx + 1]
        decoded_context = tokenizer.decode(context_tokens.tolist())
        
        print(f"Rank {rank}: Batch {batch_idx}, Token index {token_idx}, Activation {activation:.4f}\n"
            f"Context: {decoded_context}\n")
        base_result_dict[feat_idx].append({
            "rank": rank,
            "batch_idx": batch_idx,
            "activation": activation,
            "token_index": token_idx,
            "context": decoded_context
        })

    sys.stdout.flush()

all_result_dict = {
    'base': base_result_dict,
    'reasoning': reasoning_result_dict
}

import json
with open("./results/" + weight_path[2:-3].replace("/","__") + ".json", 'w') as file:
    json.dump(all_result_dict, file, indent=4)