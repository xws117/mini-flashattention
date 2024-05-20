import torch
import mini_flashattention

batch_size = 64
seqlen = 16
n = 16
device = 'cuda'
dtype = torch.float16

x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)


mini_flashattention.fwd(x,x,x)