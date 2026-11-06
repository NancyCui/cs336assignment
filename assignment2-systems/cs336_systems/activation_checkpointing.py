import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def four_blocks(x, block):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x

def two_blocks(x, block):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x, block):
    x = checkpoint(two_blocks, x, block, use_reentrant=False)
    x = checkpoint(two_blocks, x, block, use_reentrant=False)
    return x

def recursive_forward(blocks, x):
    n = len(blocks)

    if n == 1:
        return blocks[0](x)

    mid = n // 2

    def left_fn(x):
        return recursive_forward(blocks[:mid], x)

    def right_fn(x):
        return recursive_forward(blocks[mid:], x)

    x = checkpoint(left_fn, x, use_reentrant=False)
    x = checkpoint(right_fn, x, use_reentrant=False)

    return x

if __name__ == "__main__":
    d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
    block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                             positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))
    block = torch.compile(block, fullgraph=True)
    x = torch.randn((4, context_length, d_model), requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        # y = four_blocks(x, block)
        y = four_blocks_checkpoint(x, block)
    print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes /(1024**2):.2f} MiB")
