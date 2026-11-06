import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def alternate_ring_all_reduce(x):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    # 保存最终的 All-Reduce 结果
    y = x.clone()

    send_buf = x.clone()
    for t in range(1, world_size):
        recv_buf = torch.empty_like(x)
        # 同时发送和接收，避免阻塞式 send/recv 导致死锁
        send_req = dist.isend(send_buf, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)
        send_req.wait()
        recv_req.wait()
        y += recv_buf
        send_buf = recv_buf
    return y

def run_ring_all_reduce(rank, world_size, backend, device):
    setup(rank, world_size, backend)
    x = torch.full((4,), fill_value=float(rank + 1), device=device, dtype=torch.float32)
    result = alternate_ring_all_reduce(x)

if __name__ == "__main__":
    num_procs = 4
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.spawn(fn=run_ring_all_reduce, args=(num_procs, backend, device), nprocs=num_procs, join=True)