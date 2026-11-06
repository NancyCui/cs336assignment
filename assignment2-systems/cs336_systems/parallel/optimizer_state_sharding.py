import torch
from torch.optim import Optimizer
import torch.distributed as dist

class OptimizerStateSharding(Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.param_to_rank = {}
        self.param_count = 0
        self.local_param_groups = []
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        super().__init__(params, defaults=kwargs)

        if self.local_param_groups:
            self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)
        else:
            self.optimizer =  None
       

    def step(self, closure=None, **kwargs):
        loss = None
        if self.optimizer is not None:
           loss = self.optimizer.step(closure=closure, **kwargs) 
        handles = []
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    handle = dist.broadcast(param.data, src=self.param_to_rank[param], async_op=True)
                    handles.append(handle)
            for handle in handles:
                handle.wait()
        return loss

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        local_params = []
        group = self.param_groups[-1]
        for param in group["params"]:
            which_rank = self.param_count % self.world_size
            self.param_to_rank[param] = which_rank
            if which_rank == self.rank:
                local_params.append(param)
            self.param_count += 1

        if local_params:
            local_param_group = {k: v for k, v in group.items() if k != 'params'}
            local_param_group['params'] = local_params
            
            if not hasattr(self, "optimizer"):
                self.local_param_groups.append(local_param_group)
            elif self.optimizer is None:
                self.optimizer = self.optimizer_cls([local_param_group], **self.optimizer_kwargs) 
            else:
                self.optimizer.add_param_group(local_param_group)   