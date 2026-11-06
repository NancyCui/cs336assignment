import os
import torch
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import statistics

# backend = "nccl"
# device_type = "gpu"
backend = "gloo"
device_type = "cpu"
device = torch.device(device_type)
data_size = [1,10,100,1024]
num_procs = [2, 4, 6]
warmup_step = 5
total_steps = 15
csv_path = "./data/all_reduce.csv"

def write_csv(df):
    print(df.to_latex(index=False, float_format="%.2f"))
    df.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
    )

def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def all_reduce_benchmark(rank, world_size, size_mb):
    setup(rank, world_size)
    try:
        num_elements = size_mb * 1000000 // 4
        times = []
        data = torch.randint(0, 10, (num_elements, ), dtype=torch.float32, device=device)
        for _ in range(warmup_step):
            dist.all_reduce(data, async_op=False)

        synchronize()
        for _ in range(warmup_step, total_steps):
            data = torch.randint(0, 10, (num_elements, ), dtype=torch.float32, device=device) 
            synchronize()
            start_time = timeit.default_timer()
            dist.all_reduce(data, async_op=False)
            synchronize()
            run_time = timeit.default_timer() - start_time
            times.append(run_time)

        avg_time = sum(times) / len(times)
        gathered_times = [None] * world_size
        dist.all_gather_object(gathered_times, avg_time)

        if rank == 0:
            final_avg, final_std = statistics.mean(gathered_times), statistics.stdev(gathered_times) 
            df = pd.DataFrame([{
                "world_size": world_size,
                "size_mb": size_mb,
                "Average (ms)": f"{final_avg * 1000:.2f} ± {final_std * 1000:.2f}",
            }])
            write_csv(df)
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    for world_size in num_procs:
        for size_mb in data_size:
            mp.spawn(fn=all_reduce_benchmark, args=(world_size, size_mb, ), nprocs=world_size, join=True)