# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import statistics
import torch
import torch.nn as nn
import argparse
import timeit
from cs336_basics.model import *
from cs336_basics.optimizer import *
import yaml
import pandas as pd
import wandb
from contextlib import nullcontext
from dataclasses import dataclass, asdict

use_cast = 0

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

def create_random_batch(batch_size, context_length, vocab_size, device):
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def run_benchmark(lm_model, optimizer, input, warmup_steps, total_steps, mode, config, profile_memory=False):
    use_cast = config.dtype != torch.float32
    ctx = (
        torch.autocast(device_type=config.device.type, dtype=config.dtype)
        if use_cast
        else nullcontext()
    )
    if args.use_compile:
        lm_model = torch.compile(lm_model, fullgraph=True)
    with ctx:
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
    if profile_memory:
        if config.device.type != "cuda":
            raise ValueError(
                "CUDA memory profiling requires a CUDA device"
            )

        torch.cuda.memory._record_memory_history(
            max_entries=1000000
        )

    with ctx:
        for step in range(warmup_steps, total_steps):
            new_lr = get_cosine_lr(step, max_learning_rate=config.lr, min_learning_rate=config.lr_min, warmup_iters=warmup_steps, cosine_cycle_iters=total_steps)
            for group in optimizer.param_groups:
                group['lr'] = new_lr
            start_time = timeit.default_timer()
            logits = lm_model(input)
            synchronize(config.device) 
            forward_time = timeit.default_timer() - start_time
            forward_times.append(forward_time)

            if mode != "forward-only":
                backward_start_time = timeit.default_timer() 
                loss = logits.mean()
                optimizer.zero_grad()
                loss.backward()
                synchronize(config.device) 
                backward_time = timeit.default_timer() - backward_start_time
                backward_times.append(backward_time) 
                if mode == "full":
                    optimizer_start_time = timeit.default_timer()  
                    optimizer.step()
                    synchronize(config.device)
                    optimizer_time = timeit.default_timer() - optimizer_start_time
                    optimizer_times.append(optimizer_time)

    if profile_memory and config.device.type == "cuda":
        try:
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)


    def average(times):
        if not times:
            return 0.0, 0.0
        return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0
    
    return average(forward_times), average(backward_times), average(optimizer_times)


def benchmark(args):
    config = load_config(args.config_path)
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
    (f_mean, f_std), (b_mean, b_std), (o_mean, o_std) = run_benchmark(transformer_lm, adamw, input, args.warmup, args.total_steps, args.mode, config, args.profile_memory)
    csv_path = "./data/benchmark_results_precision.csv"
    df = pd.DataFrame([{
        "isCompile": args.use_compile,
        "Mode": args.mode,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "dtype": config.dtype,
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
    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--mode", choices=["forward-only", "forward-backward", "full"], default="full")
    parser.add_argument("--profile_memory", action="store_true", help="Enable CUDA memory profiling")

    args = parser.parse_args()
    benchmark(args)
