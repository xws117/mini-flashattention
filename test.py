import torch
import mini_flashattention

batch_size = 1
seqlen = 16     # 把这个seqlen定死了
head_num = 1
n = 32
device = 'cuda'
dtype = torch.float16

x = torch.randn(batch_size,seqlen, head_num, n, device='cuda', dtype=dtype)
cnt = 0
index = 0
for i in range(16):
    for j in range(32):
        x[0,i,0,j] = cnt
        cnt += 1

mini_flashattention.fwd(x,x,x)
