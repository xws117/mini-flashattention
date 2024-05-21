
#include "params.h"
#include "gemm_tile.h"
#include "smem_tile.h"
#include <stdio.h>

inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, int steps, const int loop_step_idx) {

//    inline __device__ Gmem_tile_qkv(void *ptr_, const uint32_t row_stride_in_elts, int bidh,int bidb,Params param,
//                                    const uint32_t head_stride_in_elts, const int headdim, const int tidx)
    const int tidx = threadIdx.x;
    __shared__ char smem[16 * 32 *2];
    Gmem_tile_qkv q = {params.q_ptr, params.row_stride_in_elts,bidb,bidh,params.head_stride_in_elts,params.d,tidx};
//    if (tidx==0){
//        printf("hhhh");
//    }
    Gmem_tile_qkv k = {params.k_ptr, params.row_stride_in_elts,bidb,bidh,params.head_stride_in_elts,params.d,tidx};
    Gmem_tile_qkv v = {params.v_ptr, params.row_stride_in_elts,bidb,bidh,params.head_stride_in_elts,params.d,tidx};

    q.load();
    __syncthreads();

    if (bidb==0 && bidh==0 && tidx==0){
        printf("uint4: (%u, %u, %u, %u)\n", q.fetch_.x, q.fetch_.y, q.fetch_.z, q.fetch_.w);
    }

    Smem_tile_row_a smem_q = {smem,tidx};
    smem_q.store(q.fetch_);
    __syncthreads();
    half* half_data = reinterpret_cast<half*>(smem);
    if (bidb==0 && bidh==0 && tidx==0){
        for(int i=0;i<4;i++){
            for(int j=0; j<8*8; j++){
                printf("%f ", __half2float(half_data[i*8*8 + j] ));
            }
            printf("\n");
        }
    }

}

inline __device__ void device_1xN_loop(const Params &params){

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    //const int tidx_global = (bidb * params.h + bidh) * blockDim.x * 2 + tidx;

    int M = params.tile_q.m; // 16
    int STEPS = (params.s + M -1) / M;  // (1024+16-1) / 16  第二层循环，就是对Q矩阵的循环，每一次读取 16 * 64 的tile，steps就是q循环的次数

    int blocksize_c = params.tile_q.n; // 256
    int max_loop_steps = (params.s + blocksize_c - 1) / blocksize_c;  // 第一层循环，读取KV的数据，每次的tile大小为64 *256 ，
                                                                      // 取值为256，为的是减少一下循环的次数，和论文里面的说法好像不一样，不知道后面的版本有没有修改

    device_1xN_(params, bidb, bidh, STEPS,  0);
    for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
        device_1xN_(params, bidb, bidh, STEPS,  loop_step_idx);
    }
    device_1xN_(params, bidb, bidh, STEPS,  max_loop_steps - 1);
}

__global__ void fmha_fprop_fp16_sm80_loop_kernel(Params params) {
    printf("Begin of fmha_fprop_fp16_sm80_loop_kernel \n");
    device_1xN_loop(params);
    printf("Begin of fmha_fprop_fp16_sm80_loop_kernel \n");
}
__global__ void fmha_test() {
    printf("wthat the fuck");
}

void run_fmha_fp16_sm80(Params params) {
    printf("Begin of run_fmha_fp16_sm80 \n");
    auto batch_size = params.b;
    auto num_heads = params.h;
    // 这里面的block的size是 batch * heads ，也就是每一个block里面，处理完整的一个q*k^*v的运算 ，每一个block中q为 seqlem * head_size
    dim3 grid(batch_size, num_heads, 1);
    // 每一个block中使用128个线程进行处理和计算，分为4个warp，
    fmha_fprop_fp16_sm80_loop_kernel<<<grid,64>>>(params);
    fmha_test<<<1,2>>>();
    cudaDeviceSynchronize();
    printf("End of run_fmha_fp16_sm80 \n");
}