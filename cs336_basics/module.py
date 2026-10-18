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

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        linear transformation module.
        """
        super().__init__()
        self.W = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, 0, std, -3*std, 3*std) 

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")
  
        
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        """
        Int[torch.Tensor, "..."] -> Float[torch.Tensor, "... embedding_dim"]
        """
        return self.embedding[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        RMSNorm is “position-wise”, only occurs on d_model
        """
        # 混合精度, 归一化、loss、softmax要用fp32, to prevent overflow 
        in_dtype = x.dtype          # 记录输入类型
        x = x.to(torch.float32)     # 转到 float32
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x / rms * self.gain
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        # down(gate * up)
        return self.w2.forward(self.SiLU(self.w1.forward(x)) * self.w3.forward(x))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        thetas = einsum(
            torch.arange(max_seq_len),
            torch.pow(theta, -torch.arange(0, d_k, 2) / d_k),
            "index, theta -> index theta",
        )
        cos = torch.cos(thetas).to(device)
        sin = torch.sin(thetas).to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2] 
        x_odd = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        result = torch.empty_like(x)
        result[..., ::2] = x_even * cos - x_odd * sin
        result[..., 1::2] = x_even * sin + x_odd * cos
        return result

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature zero error")
    x = x / temperature
    x = x - x.max(dim, keepdim=True).values
    exp = x.exp()
    return exp / exp.sum(dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask:None):
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    return einsum(softmax(scores/ math.sqrt(K.shape[-1]), -1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, pos_encoder=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pos_encoder = pos_encoder
        self.device = device
        self.dtype = dtype
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype) 
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype) 
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype) 
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1) == 0, persistent=False)

    def scaled_dot_product_attention(self, Q, K, V, mask:None):
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))
        return einsum(softmax(scores/ math.sqrt(K.shape[-1]), -1), V, "... queries keys, ... keys d_v -> ... queries d_v") 

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Qh = rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        Kh = rearrange(K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        Vh = rearrange(V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads) 
        if self.pos_encoder is not None and token_positions is not None:
            Qh = self.pos_encoder(Qh, token_positions)
            Kh = self.pos_encoder(Kh, token_positions)
        mask = self.causal_mask[:seq_len, :seq_len]
        out = rearrange(self.scaled_dot_product_attention(Qh, Kh, Vh, mask), "... h seq_len d -> ... seq_len (h d)")
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta, device=None, dtype=None):
        super().__init__()
        self.rope = RoPE(rope_theta, d_model//num_heads, max_seq_len, device)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, pos_encoder=self.rope, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.token_positions = torch.arange(max_seq_len, device=device)

    def forward(self, x):
        # mha
        x_residual = x
        seq_len = x.shape[-2]
        x = self.mha(self.norm1(x), self.token_positions[: seq_len])
        x = x_residual + x

        #ffn
        x_residual = x
        x = self.swiglu(self.norm2(x))
        x = x_residual + x
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) for i in range(num_layers)])
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        for transformerLayer in self.layers:
            x = transformerLayer(x)
        x = self.output_embedding(self.norm(x))
        return x






