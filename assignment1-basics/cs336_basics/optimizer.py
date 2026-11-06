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


def cross_entropy(pred_logits, targets):
    x = pred_logits - pred_logits.max(dim=-1, keepdim=True).values
    x = x - x.exp().sum(dim=-1, keepdim=True).log()
    loss = - x[torch.arange(x.shape[0]), targets]
    loss = loss.mean() 
    return loss

def perplexity(pred_logits, targets):
    loss = cross_entropy(pred_logits, targets)
    return torch.exp(loss)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps} 
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1) 
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                g = p.grad.data
                m = betas[0] * m + (1- betas[0]) * g
                v = betas[1] * v + (1- betas[1]) * g.pow(2)
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr * weight_decay * p.data
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def lr_cosine(t, lr_max, lr_min, T_warmup, T_c):
    if t < T_warmup:
        return t/T_warmup * lr_max
    if t <= T_c:
        return lr_min + 0.5 * (math.cos((t - T_warmup)/(T_c - T_warmup) * math.pi) + 1) * (lr_max - lr_min)
    return lr_min

def gradient_clipping(parameters, max_l2_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += (p.grad.data**2).sum().item()
    total_norm = math.sqrt(total_norm)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= clip_coef


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward() 
        opt.step()






