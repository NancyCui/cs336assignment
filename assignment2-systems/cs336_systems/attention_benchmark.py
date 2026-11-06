# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import gc
import torch
import timeit
import itertools
from cs336_basics.model import scaled_dot_product_attention

#device = "cuda"
device = "mps"
BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]

WARMUP = 10
REPEAT = 100

DEVICE = torch.device(device)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()

def benchmark(d_model, seq_len):
    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)

    compiled_attention = torch.compile(scaled_dot_product_attention, fullgraph=True)

    def forward():
        output = compiled_attention(Q, K, V)
        synchronize()
        return output

    def backward(output):
        output.sum().backward()
        synchronize() 


    # warmup
    for _ in range(WARMUP):
        output = compiled_attention(Q, K, V)
        output.sum().backward()

        Q.grad = K.grad = V.grad = None
    synchronize()
    del output

    forward_times = []
    backward_times = []
    memory_mibs = []
    for _ in range(REPEAT):
        Q.grad = K.grad = V.grad = None
        start_time = timeit.default_timer()
        output = forward()
        forward_time = timeit.default_timer() - start_time
        forward_times.append(forward_time * 1000)

        if DEVICE.type == "cuda":
            memory_mibs.append(torch.cuda.memory_allocated() / 1024**2)

        start_time = timeit.default_timer()
        backward(output)
        backward_time = timeit.default_timer() - start_time 
        backward_times.append(backward_time * 1000)
        del output

    return {"d_model": d_model, "seq_len": seq_len,
            "forward_ms": sum(forward_times) / REPEAT,
            "backward_ms": sum(backward_times) / REPEAT,
            "memory_mib": ( sum(memory_mibs) / len(memory_mibs) if memory_mibs else float("nan")),
            }



if __name__ == "__main__":
    for d_model, seq_len in itertools.product(HEAD_DIMS, SEQ_LENS):
        try:
            res = benchmark(d_model, seq_len)
            print(
                f"d_model={res['d_model']:3d} | "
                f"seq_len={res['seq_len']:5d} | "
                f"forward={res['forward_ms']:.2f}ms | "
                f"backward={res['backward_ms']:.2f}ms | "
                f"memory={res['memory_mib']:.2f}MiB",
                flush=True,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps backend out of memory" in msg:
                print(f"OOM: d_model={d_model}, seq_len={seq_len}")
            elif "int_max" in msg:
                print(f"MPSGraph limit: d_model={d_model}, seq_len={seq_len}")
            else:
                raise
        finally:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE.type == "mps":
                torch.mps.empty_cache()
