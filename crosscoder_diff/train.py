# %%
import os
import torch
import torch.distributed as dist
from utils import *
from trainer import Trainer

# --- Distributed Setup ---
# Check if we're in a distributed run by looking for environment variables set by torchrun.
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"[Rank {rank}] Initializing distributed training (world size: {world_size}, local rank: {local_rank})")
    # Initialize the process group with the NCCL backend (best for multi-GPU on a single node)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
else:
    rank = 0
    world_size = 1
    local_rank = 0
    device = "cuda"
    
# %%

save_name = "qwen-1.5b"

name_to_model_map = {
    "qwen-1.5b": ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "qwen-7b": ["Qwen/Qwen2.5-Math-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    "qwen-14b": ["Qwen/Qwen2.5-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
    "qwen-32b": ["Qwen/Qwen2.5-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"],
}

# Load the models onto the proper device
base_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][0],
    device=device,
    n_devices=4
)

chat_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][1],
    device=device,
    n_devices=4
)

d_in = base_model.cfg.d_model
layer = base_model.cfg.n_layers // 2

# --- Wrap models with DistributedDataParallel if using more than one GPU ---
if world_size > 1:
    from torch.nn.parallel import DistributedDataParallel as DDP
    base_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)
    chat_model = DDP(chat_model, device_ids=[local_rank], output_device=local_rank)


# %%
all_tokens = load_open_reasoning_tokens()
print(all_tokens.shape)

import sys
print("Size of all_tokens: ", sys.getsizeof(all_tokens) / 1024**3, "GB")
sys.stdout.flush()

# %%
import wandb
#wandb.login(key="53d2399bd1fe394b1c3fcbfcbea36897145e459f")
default_cfg = {
    "seed": 49,
    "batch_size": 1024,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 200_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": d_in,
    "dict_size": 2**15,
    "seq_len": 4096,
    "enc_dtype": "bf16",
    "model_name": name_to_model_map[save_name][0],
    "site": "resid_pre",
    "device": device,
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": f"blocks.{layer}.hook_resid_pre",
    "wandb_project": "crosscoder-reasoning",
    "save_name": save_name,
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
# %%