
#include "params.h"
#include "gemm_tile.h"

inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, int steps, const int loop_step_idx) {

//    inline __device__ Gmem_tile_qkv(void *ptr_, const uint32_t row_stride_in_elts, int bidh,int bidb,Params param,
//                                    const uint32_t head_stride_in_elts, const int headdim, const int tidx)

    const int tidx = threadIdx.x;
    Gmem_tile_qkv q = {params.q, params.row_stride_in_elts,bidb,bidh,params.head_stride_in_elts,params.d,tidx};
    if (tidx==0){
        printf("hhhh");
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