# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import gc
import torch
from einops import rearrange, einsum
from math import ceil
from typing import Tuple, Optional
from torch.autograd import Function
import triton
import triton.language as tl
import pandas as pd
from cs336_systems.triton_flash_attention import *
from cs336_basics.model import *

csv_path = "./data/flash_attention_vs_torch_attention.csv"
device = "cuda"
DEVICE = torch.device(device)
seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
dims = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]

def generate_inputs(batch_size, seq_len, dim, dtype):
    torch.manual_seed(0)
    q = torch.randn(batch_size, seq_len, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    do = torch.randn(batch_size, seq_len, dim, device=DEVICE, dtype=dtype)
    return q, k, v, do

def triton_flash_attn(q, k, v, is_causal=True) :
    return TritonFlashAttention2.apply(q, k, v, is_causal)

def pytorch_vanilla_attn(q, k, v, is_causal=True):
    mask = None
    if is_causal:
        mask = ~torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=DEVICE), diagonal=1)
    return scaled_dot_product_attention(q, k, v, mask)

def run_benchmark(dtype, seq_len, dim):
    Q_ROWS_TILE_SIZE, K_ROWS_TILE_SIZE = get_tile_sizes(seq_len, dim)
    q, k, v, do = generate_inputs(1, seq_len, dim, dtype)
    # pytorch time
    pytorch_fwd = triton.testing.do_bench(lambda: pytorch_vanilla_attn(q, k, v, True))
    out = pytorch_vanilla_attn(q, k, v, True)
    pytorch_bwd = triton.testing.do_bench(
        lambda: out.backward(do, retain_graph=True),
        grad_to_none=[q, k, v],
    )
    del out
    q.grad = k.grad = v.grad = None


    def pytorch_end_to_end():
        out = pytorch_vanilla_attn(q, k, v, True)
        out.backward(do)

    pytorch_total = triton.testing.do_bench(
        pytorch_end_to_end,
        grad_to_none=[q, k, v],
    )
    q.grad = k.grad = v.grad = None

    # triton
    triton_fwd = triton.testing.do_bench(lambda: triton_flash_attn(q, k, v, True))
    out_triton = triton_flash_attn(q, k, v, True)
    triton_bwd = triton.testing.do_bench(
        lambda: out_triton.backward(do, retain_graph=True),
        grad_to_none=[q, k, v],
    ) 
    del out_triton
    q.grad = k.grad = v.grad = None

    def triton_end_to_end():
        out = triton_flash_attn(q, k, v, True)
        out.backward(do)

    triton_total = triton.testing.do_bench(
        triton_end_to_end,
        grad_to_none=[q, k, v],
    )

    df = pd.DataFrame([{
        "d_model": dim,
        "seq_len": seq_len,
        "d_type": dtype,
        "Q_ROWS_TILE_SIZE": Q_ROWS_TILE_SIZE,
        "K_ROWS_TILE_SIZE": K_ROWS_TILE_SIZE,
        "triton_fwd(ms)": triton_fwd,
        "triton_bwd(ms)": triton_bwd,
        "triton_total(ms)": triton_total,
        "pytorch_fwd(ms)": pytorch_fwd,
        "pytorch_bwd(ms)": pytorch_bwd,
        "pytorch_total(ms)": pytorch_total,
    }])
    print(df.to_latex(index=False, float_format="%.2f"))
    df.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
    )



if __name__ == "__main__":
    for dtype in dtypes:
        for seq_len in seq_lengths:
            for dim in dims:
                try:
                    run_benchmark(dtype, seq_len, dim)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "mps backend out of memory" in msg:
                        print(f"OOM: d_model={dim}, seq_len={seq_len}")
                    elif "int_max" in msg:
                        print(f"MPSGraph limit: d_model={dim}, seq_len={seq_len}")
                    else:
                        raise
                finally:
                    gc.collect()
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
                    elif DEVICE.type == "mps":
                        torch.mps.empty_cache()
    df_all = pd.read_csv(csv_path)

    print(df_all.to_latex(
        index=False,
        float_format="%.2f",
    ))
