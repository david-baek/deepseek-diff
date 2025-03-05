# %%
import os
from IPython import get_ipython

ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import plotly.io as pio
pio.renderers.default = "jupyterlab"

# Import stuff
import einops
import json
import argparse

from datasets import load_dataset
from pathlib import Path
import plotly.express as px
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch
import numpy as np
from transformer_lens import HookedTransformer
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

from functools import partial

from IPython.display import HTML

from transformer_lens.utils import to_numpy
import pandas as pd
import random

from html import escape
import colorsys


import wandb

import plotly.graph_objects as go

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis",
     "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid",
     "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth"
}

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    fig = px.imshow(to_numpy(tensor), color_continuous_midpoint=0.0,labels={"x":xaxis, "y":yaxis}, **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label

    fig.show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(y=to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, return_fig=False, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    if return_fig:
        return fig
    fig.show(renderer)

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def bar(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.bar(
        y=to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        template="simple_white",
        **kwargs).show(renderer)

def create_html(strings, values, saturation=0.5, allow_different_length=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))

# crosscoder stuff

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg    

def load_pile_lmsys_mixed_tokens():
    try:
        print("Loading data from disk")
        all_tokens = torch.load("./data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
    except:
        print("Data is not cached. Loading data from HF")
        data = load_dataset(
            "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2", 
            split="train", 
            cache_dir="/om/user/dbaek/.cache/"
        )
        data.save_to_disk("./data/pile-lmsys-mix-1m-tokenized-gemma-2.hf")
        data.set_format(type="torch", columns=["input_ids"])
        all_tokens = data["input_ids"]
        torch.save(all_tokens, "./data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
        print(f"Saved tokens to disk")
    return all_tokens
from transformers import AutoTokenizer
def load_reasoning_tokens():
    cache_path = "./data/s1k-reasoning-tokens.pt"
    try:
        print("Loading reasoning tokens from disk")
        tokenized_lines = torch.load(cache_path)
        return tokenized_lines
    except Exception as e:
        print("Cached reasoning tokens not found. Loading dataset from HF")
        data = load_dataset(
            "simplescaling/s1K", 
            split="train", 
            cache_dir="/om/user/dbaek/.cache/"
        )
        
        # Initialize the tokenizer.
        # Here we use Qwen/Qwen2.5-Math-1.5B; replace with your model if needed.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        
        tokenized_lines = []
        batch_lines = []
        for row in data:
            # Concatenate the question and thinking trajectory.
            text_line = (
                "Question:\n" 
                + row["question"] 
                + "\n<think>" 
                + row["thinking_trajectories"][0] 
                + "\n</think>"
            )
            # Tokenize without adding special tokens.
            tokens = tokenizer(text_line, add_special_tokens=False)["input_ids"]
            if len(tokens) >= 2048:
                batch_lines.append(text_line)
                
        tokenized_lines = tokenizer(batch_lines, truncation=True, max_length=2048, return_tensors="pt")["input_ids"]
        
        # Save the padded tensor to disk.
        torch.save(tokenized_lines, cache_path)
        print("Saved reasoning tokens to disk")
        return tokenized_lines


def load_open_reasoning_tokens():
    cache_path = "../data/open-reasoning-tokens.pt"
    try:
        print("Loading reasoning tokens from disk")
        tokenized_lines = torch.load(cache_path)
        return tokenized_lines
    except Exception as e:
        print("Cached reasoning tokens not found. Loading dataset from HF")
        data = load_dataset(
            "open-thoughts/OpenThoughts-114k", 
            split="train", 
            cache_dir="/om/user/dbaek/.cache/"
        )
        data = data.shuffle(seed=49)
        
        # Initialize the tokenizer.
        # Here we use Qwen/Qwen2.5-Math-1.5B; replace with your model if needed.
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        
        tokenized_lines = []
        batch_lines = []
        for row in data:
            # Concatenate the question and thinking trajectory.
            question = row['conversations'][0]['value']
            thinking_trajectories = row['conversations'][1]['value']
            thinking_trajectories = thinking_trajectories.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>").replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
            text_line = question + "\n" + thinking_trajectories
            # Tokenize without adding special tokens.
            tokens = tokenizer(text_line, add_special_tokens=False)["input_ids"]
            if len(tokens) >= 2048:
                batch_lines.append(text_line)

        data_text = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            split="train",
            cache_dir="/om/user/dbaek/.cache/"
        )
        data_text = data_text.shuffle(seed=49)
        data_text = data_text.take(100000)
        for row in data_text:
            # Concatenate the question and thinking trajectory.
            text_line = row['text']
            # Tokenize without adding special tokens.
            tokens = tokenizer(text_line, add_special_tokens=False)["input_ids"]
            if len(tokens) >= 2048:
                batch_lines.append(text_line)

        random.seed(49)
        random.shuffle(batch_lines)
                
        tokenized_lines = tokenizer(batch_lines, truncation=True, max_length=2048, return_tensors="pt")["input_ids"]
        
        # Save the padded tensor to disk.
        torch.save(tokenized_lines, cache_path)
        print("Saved reasoning tokens to disk")
        return tokenized_lines
