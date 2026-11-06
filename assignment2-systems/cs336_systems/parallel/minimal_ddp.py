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
            
class Minimal_DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        broadcast_model(self.module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        grads = [param.grad for param in self.module.parameters() if param.grad is not None]

        if not grads:
            return
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        with torch.cuda.nvtx.range("sync_all_reduce"):
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False,)
        flat_grads.div_(self.world_size)
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for grad, unflat_grad in zip(grads, unflat_grads):
            grad.copy_(unflat_grad)



