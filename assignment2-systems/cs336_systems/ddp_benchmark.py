import os
import torch
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import statistics
import argparse
import yaml
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.nn_utils import *
from cs336_systems.parallel.naive_ddp import *
from cs336_systems.parallel.minimal_ddp import Minimal_DDP
from cs336_systems.parallel.ddp import DDP
from cs336_systems.parallel.optimizer_state_sharding import OptimizerStateSharding
from dataclasses import dataclass, asdict
from cs336_systems.parallel.fsdp import FSDP
import cs336_systems.parallel.fsdp as fsdp_impl

# backend = "nccl"
# device_type = "gpu"

d_model = 1024
d_ff = 4096
num_layers = 24
num_heads = 16
warmup_steps = 5

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
    backend: str


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

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cuda_memory_stats(device):
    torch.cuda.synchronize(device)
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "peak_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def train_step(model, optimizer, x, y, device, dtype, profile_memory=False):
    synchronize(device)
    start = timeit.default_timer()
    if profile_memory:
        torch.cuda.reset_peak_memory_stats(device)

    with nvtx.range("train_step"):
        optimizer.zero_grad()

        # Model weights stay FP32; BF16 autocast can reduce activation memory.
        with nvtx.range("forward"):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
                logits = model(x)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        with nvtx.range("backward"):            
            loss.backward()

        with nvtx.range("gradient_sync"):
            comm_start = timeit.default_timer()
            model.finish_gradient_synchronization()
            synchronize(device)
            comm_seconds = timeit.default_timer() - comm_start

        before_step_memory = None
        if profile_memory:
            before_step_memory = cuda_memory_stats(device)
            # 重置峰值，单独统计 optimizer.step() 阶段
            torch.cuda.reset_peak_memory_stats(device)


        with nvtx.range("optimizer"): 
            optimizer.step()

        after_step_memory = None
        if profile_memory:
            after_step_memory = cuda_memory_stats(device)
    
    synchronize(device)
    step_seconds = timeit.default_timer() - start
    return step_seconds, comm_seconds,before_step_memory, after_step_memory 

def tensor_memory_mb(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None) / 1024**2

def optimizer_state_memory_mb(optimizer):
    inner = getattr(optimizer, "optimizer", optimizer)
    if inner is None:
        return 0.0
    return tensor_memory_mb(
        value
        for state in inner.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )

def memory_breakdown(model, optimizer, device):
    # 将当前 GPU 显存拆成 Parameters、Gradients、Optimizer States、Other 四部分
    params = tensor_memory_mb(model.parameters())
    grads = tensor_memory_mb(p.grad for p in model.parameters())
    states = optimizer_state_memory_mb(optimizer)
    allocated = cuda_memory_stats(device)["allocated_mb"]
    return [params, grads, states, allocated - params - grads - states]

def enable_fsdp_all_gather_nvtx():
    # 运行时把 fsdp 模块中的 all_gather_weight 替换为带 NVTX 标记的函数。
    if getattr(fsdp_impl.all_gather_weight, "_nvtx_wrapped", False):
        return

    original = fsdp_impl.all_gather_weight

    def traced_all_gather(*args, **kwargs):
        with nvtx.range("FSDP::all_gather_weight"):
            return original(*args, **kwargs)

    traced_all_gather._nvtx_wrapped = True
    fsdp_impl.all_gather_weight = traced_all_gather

def ddp_benchmark(rank, world_size, data, target, total_steps, mode, optimizer_mode, profile_memory, config):
    setup(rank, world_size, config.backend)
    try:
        batch_size = data.shape[0]
        assert batch_size % world_size == 0
        local_batch_size = batch_size // world_size

        start_index = rank * local_batch_size
        end_index = start_index + local_batch_size

        device = (torch.device(f"cuda:{rank}") if config.backend == "nccl" 
                  else config.device)
        data = data[start_index:end_index].to(device)
        target = target[start_index:end_index].to(device)
        if profile_memory:
            torch.cuda.reset_peak_memory_stats(device)
        model = BasicsTransformerLM(vocab_size=config.vocab_size,
                                    context_length=config.context_length,
                                    d_model=d_model,
                                    num_layers=num_layers,
                                    num_heads=num_heads,
                                    d_ff=d_ff,
                                    rope_theta=10000.0).to(device)
        if mode == 0:
            model = Naive_DDP(model)
        elif mode == 1:
            model = Minimal_DDP(model)
        elif mode == 2:
            model = DDP(model)
        elif mode == 3:
            enable_fsdp_all_gather_nvtx()
            model = FSDP(model, compute_dtype=None)
        else:
            raise ValueError(f"Unknown DDP mode: {mode}")

        optimizer_kwargs = dict(lr=config.lr, betas=config.betas,
                                eps=config.eps, weight_decay=config.weight_decay)

        if optimizer_mode == "zero1":
            optimizer = OptimizerStateSharding(model.parameters(), AdamW, **optimizer_kwargs)
        else:
            optimizer = AdamW(model.parameters(), **optimizer_kwargs)

        records = []
        memory_records = []
        if profile_memory:
            init_memory = cuda_memory_stats(device)
            torch.cuda.reset_peak_memory_stats(device)
        for i in range(total_steps):
            step_seconds, comm_seconds, before_step_memory, after_step_memory = train_step(model, optimizer, data, target, device, config.dtype, profile_memory)

            if i >= warmup_steps:
                records.append((step_seconds, comm_seconds))

            if profile_memory:
                memory_records.append({
                    "step": i,
                    "before_allocated_mb": before_step_memory["allocated_mb"],
                    "before_peak_mb": before_step_memory["peak_mb"],
                    "after_allocated_mb": after_step_memory["allocated_mb"],
                    "after_peak_mb": after_step_memory["peak_mb"],
                })

        stat_names = ["step_seconds_mean", "comm_seconds_mean"]
        stat_values = [
            statistics.mean(r[0] for r in records),
            statistics.mean(r[1] for r in records),
        ]

        if profile_memory:
            first = memory_records[0]
            steady = memory_records[warmup_steps:] or memory_records
            breakdown = memory_breakdown(model, optimizer, device)

            memory_stats = {
                # 初始化
                "init_allocated_mib": init_memory["allocated_mb"],
                "init_peak_mib": init_memory["peak_mb"],

                # 第一次 Step
                "first_before_step_allocated_mib": first["before_allocated_mb"],
                "first_before_step_peak_mib": first["before_peak_mb"],
                "first_after_step_allocated_mib": first["after_allocated_mb"],
                "first_after_step_peak_mib": first["after_peak_mb"],

                # 全部 Step
                "all_before_step_allocated_mean_mib": statistics.mean(
                    r["before_allocated_mb"] for r in memory_records
                ),
                "all_forward_backward_peak_max_mib": max(
                    r["before_peak_mb"] for r in memory_records
                ),
                "all_after_step_allocated_mean_mib": statistics.mean(
                    r["after_allocated_mb"] for r in memory_records
                ),
                "all_optimizer_step_peak_max_mib": max(
                    r["after_peak_mb"] for r in memory_records
                ),
                "overall_training_peak_max_mib": max(
                    max(r["before_peak_mb"], r["after_peak_mb"])
                    for r in memory_records
                ),

                # 稳态 Step
                "steady_after_step_allocated_mean_mib": statistics.mean(
                    r["after_allocated_mb"] for r in steady
                ),
                "steady_optimizer_step_peak_max_mib": max(
                    r["after_peak_mb"] for r in steady
                ),

                # 显存组成
                "parameters_mib": breakdown[0],
                "gradients_mib": breakdown[1],
                "optimizer_states_mib": breakdown[2],
                "other_mib": breakdown[3],
            }
            stat_names.extend(memory_stats.keys())
            stat_values.extend(memory_stats.values())

        local_stats = torch.tensor(
            stat_values,
            device=device,
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]

        dist.all_gather(gathered, local_stats)

        if rank == 0:
            for r, stats in enumerate(gathered):
                step_ms = stats[0].item() * 1000
                comm_ms = stats[1].item() * 1000

                print(
                    f"rank={r} | "
                    f"step={step_ms:.2f}ms | "
                    f"comm={comm_ms:.2f}ms | "
                    f"ratio={comm_ms / step_ms * 100:.2f}%"
                )

            # 汇总为一个整体 Benchmark 结果
            mean_step = statistics.mean(s[0].item() for s in gathered)
            mean_comm = statistics.mean(s[1].item() for s in gathered)

            print(
                f"\nXL | GPUs={world_size} | "
                f"step={mean_step * 1000:.2f}ms | "
                f"comm={mean_comm * 1000:.2f}ms | "
                f"comm_ratio={mean_comm / mean_step * 100:.2f}%"
            )
            if profile_memory:
                print(
                    f"\n=== GPU Memory Summary | "
                    f"optimizer={optimizer_mode} | MiB ==="
                )

                for r, stats in enumerate(gathered):
                    print(f"\n[Rank {r} / cuda:{r}]")

                    for idx, name in enumerate(stat_names[2:], start=2):
                        print(
                            f"  {name:<42} "
                            f"{stats[idx].item():12.2f}"
                        )

                

    finally:
        dist.destroy_process_group()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--optimizer_mode", type=str, choices=["no", "zero1"], default="no")
    parser.add_argument("--profile_memory", action="store_true")
    args = parser.parse_args()

    total_steps = 15
    num_procs = 2
    mode = args.mode
    optimizer_mode = args.optimizer_mode
    profile_memory = args.profile_memory
    config = load_config("./conf/model_config_2.yaml")
    if profile_memory:
        if not torch.cuda.is_available():
            raise RuntimeError("Memory profiling requires sCUDA GPUs")
        config.backend = "nccl"

    input = create_random_batch(config.batch_size, config.context_length, config.vocab_size, config.device)
    target = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=config.device)
    mp.spawn(fn=ddp_benchmark, args=(num_procs, input, target, total_steps, mode, optimizer_mode, profile_memory, config,  ), nprocs=num_procs, join=True)

