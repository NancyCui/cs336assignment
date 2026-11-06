# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import statistics
import torch
import torch.nn as nn
import argparse
import timeit
from cs336_basics.model import *
from cs336_basics.optimizer import *
import yaml
import pandas as pd
import wandb
from contextlib import nullcontext
from dataclasses import dataclass, asdict


def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


if __name__ == "__main__":
    x = torch.randn((4, 512, 2560), requires_grad=True)
    #ln = RMSNorm(x.shape[-1])
    ln = torch.compile(RMSNorm(x.shape[-1]))
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()
