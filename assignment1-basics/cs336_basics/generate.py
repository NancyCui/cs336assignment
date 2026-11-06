# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""you may not use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
    • torch.nn.Parameter
    • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.).1
    • The torch.optim.Optimizer base class
    preferring the lexicographically greater pair
    """

import os
import math
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import typing
import yaml
from cs336_basics.module import *
from cs336_basics.optimizer import *
from cs336_basics.bpe import *
from cs336_basics.tokenizer import *
from cs336_basics.train import *

def top_p_sampling(logits, temperature=1.0, top_p=0.9):
    # print (f"top_p_sampling, temperature: {temperature:2f}, top_p: {top_p:2f}")
    probs = softmax(logits, -1, temperature)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative_probs - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_index)

def decode(lm, prompt, tokenizer, end_token, context_length, temperature, top_p, device):
    input_seqs = tokenizer.encode(prompt)
    inputs_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device).unsqueeze(0)
    end_token_id = tokenizer.encode(end_token)[0]

    with torch.no_grad():
        for _ in range(context_length - inputs_tensor.shape[1]):
            logits = lm(inputs_tensor)[:, -1, :]
            next_token_id = top_p_sampling(logits, temperature, top_p)
            
            if next_token_id.item() == end_token_id:
                            break
            inputs_tensor = torch.cat([inputs_tensor, next_token_id], dim=1)

    output = tokenizer.decode(inputs_tensor[0].tolist())
    return output


if __name__ == "__main__":
    config_file = "./config/model_config.yaml"
    tokenizer = Tokenizer.from_files(
        "./data/bpe_vocab.pkl",
        "./data/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    config = load_config(config_file)
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_20000.txt")
    print (checkpoint_path)

    transformer_lm = TransformerLM(vocab_size=config.vocab_size, 
                  context_length=config.context_length, 
                  d_model=config.d_model, 
                  num_layers=config.num_layers, 
                  num_heads=config.num_heads, 
                  d_ff=config.d_ff, 
                  rope_theta=config.rope_theta, 
                  device=config.device, 
                  dtype=config.dtype)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    transformer_lm.load_state_dict(checkpoint["model"])
    transformer_lm.eval()
    prompt = "Once upon a time"
    end_token = "<|endoftext|>"
    generated_text = decode(transformer_lm, prompt, tokenizer, end_token,  context_length=256, temperature=0.2, top_p=0.9, device=config.device)
    print(generated_text)
