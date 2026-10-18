# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""you may not use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
    • torch.nn.Parameter
    • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.).1
    • The torch.optim.Optimizer base class
    preferring the lexicographically greater pair
    """

import os
import math
import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import typing
import yaml
from cs336_basics.module import *
from cs336_basics.optimizer import *
from cs336_basics.bpe import *
from cs336_basics.tokenizer import *


if __name__ == "__main__":
    print ("generate")