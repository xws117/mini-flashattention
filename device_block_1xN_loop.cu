
#include "params.h"

inline __device__ void device_block_1xN_(const Params &params, const int bidb, const int bidh, int steps, Prng &ph0, Prng &ph1, const int loop_step_idx) {

}

inline __device__ void device_block_1xN_loop(const Params &params){

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    //const int tidx_global = (bidb * params.h + bidh) * blockDim.x * 2 + tidx;

    int M = params.tile_q.m; // 16
    int STEPS = (params.s + M -1) / M;  // (1024+16-1) / 16

    int blocksize_c = params.tile_q.n; // 256

    int max_loop_steps = (params.s + blocksize_c - 1) / blocksize_c;
    device_block_1xN_(params, bidb, bidh, STEPS,  0);
    for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
        device_block_1xN_(params, bidb, bidh, STEPS,  loop_step_idx);
    }
    device_block_1xN_(params, bidb, bidh, STEPS,  max_loop_steps - 1);
}