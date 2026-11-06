# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import statistics
import torch
import torch.nn as nn
import argparse
import timeit
import cs336_basics.model
from cs336_basics.model import *
from cs336_basics.optimizer import *
import yaml
import pandas as pd
import torch.cuda.nvtx as nvtx
import wandb
from dataclasses import dataclass, asdict

@dataclass
class TrainConfig:
    device: torch.device
    dtype: torch.dtype
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: int
    lr: float
    lr_min: float
    weight_decay: float
    betas: tuple
    eps: float
    max_grad_norm: float
    warmup_ratio: float
    token_ids_path: str
    checkpoint_dir: str
    eval_token_ids_path: str
    batch_size: int
    total_tokens: int
    validation_interval: int
    checkpoint_interval: int
    eval_steps: int
    wandb_project: str
    wandb_name: str

def load_config(src):
    with open(src) as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = torch.device(cfg["device"])
    cfg["dtype"] = getattr(torch, cfg["dtype"])
    cfg["betas"] = tuple(cfg["betas"])
    return TrainConfig(**cfg) 

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        res = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return res

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def create_random_batch(batch_size, context_length, vocab_size, device):
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def run_benchmark(lm_model, optimizer, input, warmup_steps, total_steps, mode, config):
    for step in range(warmup_steps):
        new_lr = get_cosine_lr(step, max_learning_rate=config.lr, min_learning_rate=config.lr_min, warmup_iters=warmup_steps, cosine_cycle_iters=total_steps)
        for group in optimizer.param_groups:
            group['lr'] = new_lr
        logits = lm_model(input)
        if mode != "forward-only":
            loss = logits.mean()
            optimizer.zero_grad()
            loss.backward()
            if mode == "full":
                optimizer.step() 
        synchronize(config.device)

    forward_times = []
    backward_times = []
    optimizer_times = []
    with nvtx.range_push("benchmark"):
        for step in range(warmup_steps, total_steps):
            with nvtx.range("get_consine_lr"):
                new_lr = get_cosine_lr(step, max_learning_rate=config.lr, min_learning_rate=config.lr_min, warmup_iters=warmup_steps, cosine_cycle_iters=total_steps)
            for group in optimizer.param_groups:
                group['lr'] = new_lr
            start_time = timeit.default_timer()
            with nvtx.range("Forward"):
                logits = lm_model(input)
            synchronize(config.device) 
            forward_time = timeit.default_timer() - start_time
            forward_times.append(forward_time)

            if mode != "forward-only":
                backward_start_time = timeit.default_timer() 
                loss = logits.mean()
                optimizer.zero_grad()
                with nvtx.range("backward"):
                    loss.backward()
                synchronize(config.device) 
                backward_time = timeit.default_timer() - backward_start_time
                backward_times.append(backward_time) 
                if mode == "full":
                    optimizer_start_time = timeit.default_timer()  
                    with nvtx.range("optimizer step"):
                        optimizer.step()
                    synchronize(config.device)
                    optimizer_time = timeit.default_timer() - optimizer_start_time
                    optimizer_times.append(optimizer_time)
            
    def average(times):
        if not times:
            return 0.0, 0.0
        return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0
    
    return average(forward_times), average(backward_times), average(optimizer_times)


def benchmark(args):
    config = load_config(args.config_path)
    with nvtx.range("define model"):
        transformer_lm = BasicsTransformerLM(vocab_size=config.vocab_size,
                                context_length=config.context_length,
                                d_model=args.d_model,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                d_ff=args.d_ff,
                                rope_theta=10000.0).to(config.device)
    num_params = sum(p.numel() for p in transformer_lm.parameters())
    
    adamw = AdamW(transformer_lm.parameters(), 
                  lr=config.lr, 
                  betas=config.betas, 
                  eps=config.eps,
                  weight_decay=config.weight_decay)
    input = create_random_batch(config.batch_size, config.context_length, config.vocab_size, config.device)
    (f_mean, f_std), (b_mean, b_std), (o_mean, o_std) = run_benchmark(transformer_lm, adamw, input, args.warmup, args.total_steps, args.mode, config)
    csv_path = "./data/benchmark_results.csv"
    df = pd.DataFrame([{
        "Mode": args.mode,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "Layers": args.num_layers,
        "Warmup": args.warmup,
        "Heads": args.num_heads,
        "Batch": config.batch_size,
        "Context": config.context_length,
        "Params (B)": num_params / 1e9,
        "Forward (ms)": f"{f_mean * 1000:.2f} ± {f_std * 1000:.2f}",
        "Backward (ms)": f"{b_mean * 1000:.2f} ± {b_std * 1000:.2f}",
        "Optimizer (ms)": f"{o_mean * 1000:.2f} ± {o_std * 1000:.2f}",
        "Total (ms)": (f_mean + b_mean + o_mean) * 1000,
    }])
    print(df.to_latex(index=False, float_format="%.2f"))
    df.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--total_steps", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--mode", choices=["forward-only", "forward-backward", "full"], default="full")

    args = parser.parse_args()
    benchmark(args)
