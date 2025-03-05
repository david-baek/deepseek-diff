# %%

import torch
import einops
import sys
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_comp", type=int, default=2)
parser.add_argument("--save_name", type=str, default="qwen-1.5b")
args = parser.parse_args()
n_comp = args.n_comp
save_name = args.save_name

# %%
from transformer_lens import HookedTransformer, ActivationCache

import os
os.environ['HF_HOME'] = '/om2/user/dbaek/.cache/'

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    n_devices=2
)

chat_model = HookedTransformer.from_pretrained(
    name_to_model_map[save_name][1],
    device=device,
    n_devices=2
)
hook_point = f"blocks.{base_model.cfg.n_layers // 2}.hook_resid_pre"

function_list = ["english-spanish", "english-german", "english-french", "present-past", "singular-plural"]
from sklearn.decomposition import PCA
for func_name in function_list:
    filename = f"../abstractive/{func_name}.json"
    with open(filename, "r") as f:
        data = json.load(f)
        
    filtered_data = []
    for i in range(len(data)):
        if len(tokenizer.encode(data[i]["input"])) == 1 and len(tokenizer.encode(data[i]["output"])) == 1:
            filtered_data.append(data[i]["input"])
            filtered_data.append(data[i]["output"])

    torch.cuda.empty_cache()
    with torch.no_grad():
        _, cache_A = base_model.run_with_cache(filtered_data, names_filter=hook_point)
        _, cache_B = chat_model.run_with_cache(filtered_data, names_filter=hook_point)

    act_A = cache_A[hook_point][:, 0, :]
    act_B = cache_B[hook_point][:, 0, :]

    pca = PCA(n_components=n_comp)
    act_A_pca = pca.fit_transform(act_A.cpu().numpy())

    pca = PCA(n_components=n_comp)
    act_B_pca = pca.fit_transform(act_B.cpu().numpy())


    print(act_A_pca.shape, act_B_pca.shape)

    orig_norm_list = []
    chat_norm_list = []
    for i in range(len(filtered_data)//2):
        for j in range(i+1, len(filtered_data)//2):
            norm1 = np.linalg.norm(act_A_pca[2*i] - act_A_pca[2*i+1] - act_A_pca[2*j] + act_A_pca[2*j+1], ord=2) / np.sqrt(np.linalg.norm(act_A_pca[2*i], ord=2)**2 + np.linalg.norm(act_A_pca[2*i+1], ord=2)**2 + np.linalg.norm(act_A_pca[2*j], ord=2)**2 + np.linalg.norm(act_A_pca[2*j+1], ord=2)**2)

            norm2 = np.linalg.norm(act_B_pca[2*i] - act_B_pca[2*i+1] - act_B_pca[2*j] + act_B_pca[2*j+1], ord=2) / np.sqrt(np.linalg.norm(act_B_pca[2*i], ord=2)**2 + np.linalg.norm(act_B_pca[2*i+1], ord=2)**2 + np.linalg.norm(act_B_pca[2*j], ord=2)**2 + np.linalg.norm(act_B_pca[2*j+1], ord=2)**2)
            
            orig_norm_list.append(str(norm1))
            chat_norm_list.append(str(norm2))

    result_dict = {
        "base": orig_norm_list,
        "reasoning": chat_norm_list,
    }

    with open(f"../results/{func_name}_{save_name}_{n_comp}.json", "w") as f:
        json.dump(result_dict, f)



