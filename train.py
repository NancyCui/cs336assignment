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
import time
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import typing
import yaml
import wandb
from dataclasses import dataclass, asdict
from cs336_basics.module import *
from cs336_basics.optimizer import *

def data_loader(x, batch_size, context_length, device):
    # 1. 随机采样 batch_size 个起点
    ix = np.random.randint(0, x.shape[0] - context_length, size=(batch_size,))
    input_seqs = np.stack([x[i: i + context_length] for i in ix])
    next_token_seq = np.stack([x[i + 1: i + context_length + 1] for i in ix])
    inputs_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
    next_token_tensor = torch.tensor(next_token_seq, dtype=torch.long, device=device)
    return inputs_tensor, next_token_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    return iteration

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

@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    total_loss = 0.0
    for inputs, targets in val_batches:
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / len(val_batches)

def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def train(config_file, resume_from=None):
    config = load_config(config_file)
    transformer_lm = TransformerLM(vocab_size=config.vocab_size, 
                  context_length=config.context_length, 
                  d_model=config.d_model, 
                  num_layers=config.num_layers, 
                  num_heads=config.num_heads, 
                  d_ff=config.d_ff, 
                  rope_theta=config.rope_theta, 
                  device=config.device, 
                  dtype=config.dtype)
    
    adamw = AdamW(transformer_lm.parameters(),
                  lr=config.lr,
                  weight_decay=config.weight_decay,
                  betas=config.betas,
                  eps=config.eps,
    )

    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from, transformer_lm, adamw)
    transformer_lm_compiled = None
    if config.device.type == "cpu":
        transformer_lm_compiled = torch.compile(transformer_lm)

    if config.device.type == "mps":
        transformer_lm_compiled = torch.compile(transformer_lm, backend="aot_eager")
    
    token_ids = np.load(config.token_ids_path, mmap_mode='r')
    token_ids_eval = np.load(config.eval_token_ids_path, mmap_mode='r')
    val_batches = [data_loader(token_ids_eval, config.batch_size, config.context_length, config.device) for _ in range(config.eval_steps)]
    total_steps = (config.total_tokens // (config.batch_size * config.context_length))
    wandb.init(project=config.wandb_project, name=config.wandb_name, config={**asdict(config), "total_steps": total_steps})
    num_warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    print(f"warmup_step {num_warmup_steps:6d} | total_steps {total_steps:.2f}")
    start_time = time.perf_counter()
    eval_time_total = 0.0
    for step in range(start_step, total_steps):
        input, target = data_loader(token_ids, config.batch_size, config.context_length, config.device)
        if transformer_lm_compiled is not None:
           logits = transformer_lm_compiled(input) 
        else:
            logits = transformer_lm(input)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        adamw.zero_grad()
        loss.backward()
        gradient_clipping(transformer_lm.parameters(), config.max_grad_norm) 
        new_lr = lr_cosine(step, lr_max=config.lr, lr_min=config.lr_min, T_warmup=num_warmup_steps, T_c=total_steps)
        for group in adamw.param_groups:
            group['lr'] = new_lr
        adamw.step()
  
        if (step + 1) % config.validation_interval == 0:
            synchronize(config.device)
            elapsed = time.perf_counter() - start_time
            train_wall_time = elapsed - eval_time_total
            eval_start = time.perf_counter()
            eval_loss = evaluate(transformer_lm, val_batches)
            synchronize(config.device)
            eval_time_total += time.perf_counter() - eval_start

            print(f"Step {step+1:6d} | train_loss {loss.item():.4f} | eval_loss {eval_loss:.4f} "
                  f"| lr {new_lr:.2e} | train_wall_time {train_wall_time:.2f}s")
            wandb.log({"loss": loss, "eval_loss": eval_loss, "lr": new_lr, "train_wall_time": train_wall_time, "total_wall_time": time.perf_counter() - start_time,}, step=step+1)
                  
        if (step + 1) % config.checkpoint_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step+1}.txt")
            save_checkpoint(transformer_lm, adamw, step + 1, checkpoint_path)
            print("Saved checkpoint to:,", {checkpoint_path})

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_final.txt")
    save_checkpoint(transformer_lm, adamw, total_steps, checkpoint_path)


if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    # train("./config/model_config.yaml")
    train("./config/model_config_2.yaml")
    # train(
    #     "./config/model_config_2.yaml",
    #     resume_from="./data/checkpoints/tiny_stories/checkpoint_step_5000.txt"
    # )