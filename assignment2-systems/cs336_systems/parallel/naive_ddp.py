import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import torch.nn as nn

def broadcast_model(model, src=0):
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=src)
        for buffer in model.buffers():
            dist.broadcast(buffer, src=src)
            
class Naive_DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        broadcast_model(self.module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is None:
                continue
            with torch.cuda.nvtx.range("sync_all_reduce"):
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    async_op=False,
                )
                param.grad.div_(self.world_size)


