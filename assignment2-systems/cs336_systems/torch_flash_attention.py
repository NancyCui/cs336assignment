# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import math
import torch
from einops import rearrange, einsum
from math import ceil
from typing import Tuple, Optional
from torch.autograd import Function


def flash_attention_backward(Q, K, V, O, L, grad_output, scale, is_causal):
    batch, num_querys, dim = Q.shape
    num_keys = K.shape[1]
    Sij = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") * scale
    if is_causal:
        q_range = torch.arange(num_querys, device=Q.device)
        k_range = torch.arange(num_keys, device=Q.device)
        mask = q_range[:, None] < k_range[None, :]
        Sij = Sij + torch.where(mask, -1e6, 0.0)

    # rowsum(O * dO), 元素乘，不是矩阵
    D = (O * grad_output).sum(dim=-1)
    Pij = torch.exp(Sij - L[:, :, None])
    dV =  einsum(Pij, grad_output, "... query key, ... query d_v -> ... key d_v")
    dP = einsum(grad_output, V, "... query d_v, ... key d_v -> ... query key")
    dS = Pij * (dP - D[:, :, None]) 
    dQ = einsum(dS, K, "... query key, ... key d_k -> ... query d_k") * scale
    dK = einsum(dS, Q, "... query key, ... query d_k -> ... key d_k") * scale
    return dQ, dK, dV 

compiled_flash_backward = torch.compile(flash_attention_backward)


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, num_querys, D = Q.shape
        num_keys = K.shape[1]
        ctx.is_causal = is_causal
        ctx.Q_ROWS_TILE_SIZE = 16
        ctx.K_ROWS_TILE_SIZE = 2 ** math.ceil(math.log2(D)) // 16 #next power of D
        O = torch.empty_like(Q)
        L = torch.empty((batch, num_querys), device=Q.device, dtype=torch.float32) 
        scale = D ** -0.5
        ctx.scale = scale

        for b in range(0, batch):
            for i in range(0, num_querys, ctx.Q_ROWS_TILE_SIZE):
                # 当前 Query Block
                Qi = Q[b, i:i+ctx.Q_ROWS_TILE_SIZE].float()

                # max, l, o
                mi = torch.full((Qi.shape[0],), float("-inf"), device=Q.device, dtype=torch.float32,)
                li = torch.zeros_like(mi)
                Oi = torch.zeros((Qi.shape[0], D), device=Q.device, dtype=torch.float32)

                for j in range(0, num_keys, ctx.K_ROWS_TILE_SIZE):
                    Kj = K[b, j:j+ctx.K_ROWS_TILE_SIZE, :].float()
                    Vj = V[b, j:j+ctx.K_ROWS_TILE_SIZE, :].float()
                    Sij = einsum(Qi, Kj, "... query d_k, ... key d_k -> ... query key") * scale
                    if is_causal:
                        q_range = torch.arange(i, i + Qi.shape[0], device=Q.device)
                        k_range = torch.arange(j, j + Kj.shape[0], device=Q.device)
                        mask = q_range[:, None] < k_range[None, :]
                        Sij = Sij + torch.where(mask, -1e6, 0.0)

                    mij = Sij.max(dim=-1).values
                    mi_new = torch.maximum(mi, mij)
                    
                    alpha = torch.exp(mi - mi_new)
                    Pij = torch.exp(Sij - mi_new[:, None])
                    li = alpha * li + Pij.sum(dim=-1)
                    PV = einsum(Pij, Vj, "... query key, ... key d_v ->  ... query d_v")
                    Oi = alpha[:, None] * Oi + PV
                    mi = mi_new

                O[b, i:i+ctx.Q_ROWS_TILE_SIZE] = (Oi / li[:, None]).to(Q.dtype)
                # logsumexp: mi + torch.log(li)
                L[b, i:i+ctx.Q_ROWS_TILE_SIZE] = mi + torch.log(li)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = compiled_flash_backward(
            Q, K, V, O, L,
            grad_output,
            ctx.scale,
            ctx.is_causal
        )
        return dQ, dK, dV, None







