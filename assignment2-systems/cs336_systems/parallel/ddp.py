import torch
import torch.distributed as dist
import torch.nn as nn

def broadcast_model(model, src=0):
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=src)
        for buffer in model.buffers():
            dist.broadcast(buffer, src=src)
            
class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        broadcast_model(self.module)
        self.handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _grad_hook(self, param):
        if param.grad is None:
            return
        # Asynchronously start All-Reduce
        with torch.cuda.nvtx.range("async_all_reduce"):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True,)
        self.handles.append(handle)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        for param in self.module.parameters():
            if param.grad is None:
                continue
            param.grad.div_(self.world_size)

        self.handles.clear()

