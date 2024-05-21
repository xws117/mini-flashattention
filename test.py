import torch
import mini_flashattention

batch_size = 64 * 1024
seqlen = 16
n = 32
device = 'cuda'
dtype = torch.float16

x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype)
cnt = 0
index = 0
for i in range(16):
    for j in range(32):
        x[0,i,j] = cnt

mini_flashattention.fwd(x,x,x)