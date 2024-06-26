//
// Created by mark on 2024-05-14.
//

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <stdio.h>
#include <iostream>
#include "kernel_traits.h"
#include "params.h"


void run_fmha_fp16_sm80(Params params);

void fwd(
        const at::Tensor &q,  // q,k,v's shape is [batch * seqlen ,heads, dim]
        const at::Tensor &k,  // q,k,v's size is [64 * 1024, 16 64]
        const at::Tensor &v
        ){
    // print shape of q
    torch::IntArrayRef q_size = q.sizes();
    // 打印大小
    for (int i = 0; i < q_size.size(); ++i) {
        std::cout << "Q Dimension " << i << ": " << q_size[i] << std::endl;
    }

    // tile shape
    //using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;

    // total warp is 2, threads num is 2x32=64
    int warps_M = 1;
    int warps_N = 2;
    int THREADS = 128;

    auto batch_size = q_size[0];
    auto seqlen = q_size[1];
    auto num_heads = q_size[2];
    auto head_size = q_size[3];

    //TODO
    auto tile_q = Tile{16, 16, 32, warps_M, warps_N, 1};
    auto tile_k = Tile{16, 32,256, warps_M, warps_N, 1};
    auto tile_v = Tile{16,256, 32, warps_M, warps_N, 1};
    

    auto param = Params{
        batch_size, seqlen, num_heads, head_size,
        tile_q, tile_k, tile_v,
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        q.stride(1), q.stride(2)
    };

    run_fmha_fp16_sm80(param);

}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Multi-head Self-attention";
    m.def("fwd", &fwd, "fwd");
//    m.def("bwd", &mha_bwd, "Backward pass");
//    m.def("fwd_block", &mha_fwd_block, "Forward pass (blocksparse)");
//    m.def("bwd_block", &mha_bwd_block, "Backward pass (blocksparse)");
}
