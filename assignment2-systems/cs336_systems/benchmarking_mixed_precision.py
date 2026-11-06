# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("input:", x.dtype)
        x = self.fc1(x)
        print("after fc1:", x.dtype)
        x = self.relu(x)
        print("after relu:", x.dtype)
        x = self.ln(x)
        print("after layernorm:", x.dtype)
        x = self.fc2(x)
        print("after fc2:", x.dtype)
        return x

def make_backward_hook(name):
    def hook(module, grad_input, grad_output):
        print("Backward called!")
        print(f"\nBackward: {name}")
        print("grad_input:",
              [g.dtype if g is not None else None
               for g in grad_input])
        print("grad_output:",
              [g.dtype if g is not None else None
               for g in grad_output])
    return hook

def print_param(model):
    for name, param in model.named_parameters():
        if param.grad is not None: 
            print(f"Layer: {name} | Size: {param.size()} | Dtype: {param.dtype} | gradient of {name}: {param.grad.dtype}")
        else:
            print(f"Layer: {name} | Size: {param.size()} | Dtype: {param.dtype}")


if __name__ == "__main__":
    device_type = "mps"
    device = torch.device(device_type)
    dtype = torch.bfloat16
    
    model = ToyModel(30, 3).to(device)
    print ("model generation")
    print_param(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-2)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.ReLU)):
            module.register_full_backward_hook(make_backward_hook(name))

    with torch.autocast(device_type=device_type, dtype=dtype):
        print ("after autocast")
        print_param(model)
        input = torch.rand((10, 30), device=device)
        result = model(input)
        print ("result", result.dtype)
        target = torch.rand((10, 3), device=device)
        criterion = nn.MSELoss()
        loss = criterion(result, target)
        print ("loss", loss.dtype)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print ("after optimization")
        print_param(model)


