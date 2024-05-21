//
// Created by mark on 2024-05-16.
//
#include <cuda_fp16.h>

#include "params.h"
#include "utils.h"


#define THREADS_PER_ROW 4
#define BYTES_PER_LDG 16
#define LDGS 1
#define BYTES_PER_ELEMENT 2

struct Gmem_tile_qkv {

    inline __device__ Gmem_tile_qkv(void *ptr_, const uint32_t row_stride_in_elts, int bidh,int bidb,
                                    const uint32_t head_stride_in_elts, const int headdim, const int tidx)
            : row_stride_in_bytes(row_stride_in_elts * BYTES_PER_ELEMENT)
            , ptr(reinterpret_cast<char *>(ptr_))
            , tidx_(tidx)
            , col_predicate((tidx % THREADS_PER_ROW) * (BYTES_PER_LDG / BYTES_PER_ELEMENT) < headdim) {

        //  来计算在当前的tile中位于哪一行
        int row = tidx / THREADS_PER_ROW;
        // 计算当前tile中位于哪一列
        int col = tidx % THREADS_PER_ROW;


        // offset之前所有的 行 * 行的大小
        uint32_t row_offset = (uint32_t)(( bidb * 1024  + row) * row_stride_in_bytes);

        // offset 本行之前所有的 head * head_size
        row_offset += (uint32_t)(bidh * head_stride_in_elts * BYTES_PER_ELEMENT);

        // 然后offset自己的col*BYTES_PER_LDG 就得到了最后的要开始读取数据的位置
        ptr += row_offset + col * BYTES_PER_LDG;
    };
    const uint32_t row_stride_in_bytes;
    char *ptr;
    // The fetch registers.
    uint4 fetch_;
    // Keep track of the row the thread is processing as we move the tile.
    // int row_;
    const int tidx_;

    const bool col_predicate;

    inline __device__ void load() {
        int row_ = tidx_ / THREADS_PER_ROW;
        const void *ptrs;
        ptrs = ptr;
        fetch_ = make_uint4(0, 0, 0, 0);
        // not packing predicates removes restrictions (e.g. FP16 384, 4 warps)
        Ldg_functor<uint4, 1> fct(fetch_, ptrs);
        fct.load(0, 1);
    }



};