import torch
from torch.optim import Optimizer
import torch.distributed as dist
import torch.nn as nn
import math
from cs336_basics.model import Linear, Embedding
from torch.autograd.graph import saved_tensors_hooks
import types

def shard_parameters(param, rank, world_size):
    flat = param.detach().reshape(-1).float()
    shard_size = (flat.numel() + world_size - 1) // world_size
    # all_gather 需要每个GPU tensor 大小一样，所以需要padding不期
    padded_size = shard_size * world_size
    padded = torch.zeros(padded_size, dtype=flat.dtype, device=flat.device)
    padded[:flat.numel()] = flat
    start = rank * shard_size
    end = start + shard_size
    return nn.Parameter(
        padded[start:end].clone(),
        requires_grad=param.requires_grad,
    )

def all_gather_weight(local_shard, original_shape, world_size, compute_dtype=None):
    dtype = compute_dtype or local_shard.dtype
    local_compute = local_shard.detach().to(dtype).contiguous()
    gathered = [torch.empty_like(local_compute) for _ in range(world_size)]
    dist.all_gather(gathered, local_compute)
    full_flat = torch.cat(gathered)
    full_weight = full_flat[:math.prod(original_shape)]
    return full_weight.view(original_shape)

def reduce_scatter_gradient(full_grad, world_size):
    flat_grad = full_grad.flatten().float()
    shard_size = (flat_grad.numel() + world_size - 1) // world_size
    padded_size = shard_size * world_size
    padded = torch.zeros(padded_size, dtype=flat_grad.dtype, device=flat_grad.device)
    padded[:flat_grad.numel()] = flat_grad
    chunks = list(padded.chunk(world_size))
    local_grad = torch.zeros_like(chunks[0])
    dist.reduce_scatter(local_grad, chunks, op=dist.ReduceOp.SUM)
    local_grad.div_(world_size)
    return local_grad

class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.module = module
        self.compute_dtype = compute_dtype

        self.handles = []
        self._sharded = {}
        self._wrap_modules()

    def _wrap_modules(self):
        for name, child in self.module.named_modules():
            if not isinstance(child, (Linear, Embedding)):
                continue
            shape = tuple(child.weight.shape)
            child.weight = shard_parameters(child.weight, self.rank, self.world_size)
            child._fsdp_original_shape = shape
            self._sharded[name] = (child, shape)
            child.register_forward_pre_hook(self._pre_forward_hook)
            child.register_forward_hook(self._post_forward_hook)
            self._wrap_forward(child)

        # Non-sharded trainable parameters are replicated; synchronize gradients.
        shard_ids = {id(child.weight) for child, _ in self._sharded.values()}
        for param in self.module.parameters():
            if id(param) not in shard_ids and param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _restore_full(self, state):
        full = all_gather_weight(state['shard'], state['shape'], self.world_size, self.compute_dtype)
        with torch.no_grad():
            state['full_param'].data = full

    def _wrap_forward(self, module):
        original_forward = module.forward

        def wrapped_forward(layer, *args, **kwargs):
            state = layer._fsdp_state
            full_param = state['full_param']
            base_ptr = full_param.untyped_storage().data_ptr()
            base_device = full_param.device
            def pack(tensor):
                if tensor.untyped_storage().data_ptr() == base_ptr and tensor.device == base_device:
                    return ('weight', tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
                return tensor
            def unpack(packed):
                if not (isinstance(packed, tuple) and len(packed) == 4 and packed[0] == 'weight'):
                    return packed
                _, size, stride, offset = packed
                if state['full_param'].numel() == 0:
                    self._restore_full(state)
                return state['full_param'].detach().as_strided(size, stride, offset)

            if isinstance(layer, Linear) and self.compute_dtype is not None:
                args = tuple(x.to(self.compute_dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in args)
            # 给 Autograd 注册一套保存和恢复 Tensor 的规则
            with saved_tensors_hooks(pack, unpack):
                return original_forward(*args, **kwargs)
            
        module.forward = types.MethodType(wrapped_forward, module)

    def _grad_hook(self, param):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, param))

    def _pre_forward_hook(self, module, inputs):
        shard = module.weight
        shape = module._fsdp_original_shape
        full = all_gather_weight(shard, shape, self.world_size, self.compute_dtype)
        full_param = nn.Parameter(full.detach(), requires_grad=shard.requires_grad)
        
        state = {'shard': shard, 'shape': shape, 'full_param': full_param}
        module._fsdp_state = state
        module._parameters['weight'] = full_param

        def on_full_grad(param):
            if param.grad is not None:
                local_grad = reduce_scatter_gradient(param.grad, self.world_size)
                with torch.no_grad():
                    if shard.grad is None:
                        shard.grad = local_grad
                    else:
                        shard.grad.add_(local_grad)
                    param.grad = None
            with torch.no_grad():
                param.data = torch.empty(0, device=param.device, dtype=param.dtype)
        
        if full_param.requires_grad:
            full_param.register_post_accumulate_grad_hook(on_full_grad)

    def _post_forward_hook(self, module, inputs, output):
        state = module._fsdp_state
        module._parameters['weight'] = state['shard']
        def restore_before_backward(grad):
            if state["full_param"].numel() == 0:
                self._restore_full(state)
            return grad
        
        # 反向传播计算到 output 的梯度时，请调用 restore_before_backward() 
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(restore_before_backward)

        with torch.no_grad():
            state['full_param'].data = torch.empty(0, device=state['shard'].device, dtype=self.compute_dtype or torch.float32)
        del module._fsdp_state

    def forward(self, *inputs, **kwargs): 
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self): 
        for handle, param in self.handles:
            handle.wait()
            param.grad.div_(self.world_size)

        self.handles.clear()



