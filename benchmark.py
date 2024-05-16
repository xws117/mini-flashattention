import torch
import os
import math
from einops import rearrange, repeat
from mini_flashattention import fwd

def attention_ref(qkv):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum('bhts,bshd->bthd', attention , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)

def cuda_attention_ref(qkv):
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q,'b s h d -> t h d').detach().requires_grad_()
    k = rearrange(q,'b s h d -> t h d').detach().requires_grad_()
    v = rearrange(q,'b s h d -> t h d').detach().requires_grad_()
    fwd(q,k,v)


repeats = 30
batch_size = 64
nheads = 16
seqlen = 1024
n = 512
d = n // nheads
device = 'cuda'
dtype = torch.float16


x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

## 传统的transformer

qkv = rearrange(Wqkv(x), 'b s (t h d) -> b s t h d', t=3, h=nheads).detach().requires_grad_()

fn = lambda qkv: attention_ref(qkv)
# benchmark_all(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')
# attention_ref(qkv)
print("finish classic transformer!")
cuda_attention_ref(qkv)


