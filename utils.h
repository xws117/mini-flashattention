

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void sts(uint32_t ptr, uint4 val) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "r"(ptr)
    , "r"(val.x)
    , "r"(val.y)
    , "r"(val.z)
    , "r"(val.w));
}

////////////////////////////////////////////////////////////////////////////////////////////////////


inline __device__ void clear(uint4 &dst) {
    dst = make_uint4(0u, 0u, 0u, 0u);
}

inline __device__ void ldg(uint4 &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint4*>(ptr);
}

template< typename Data_type, int N >
struct Ldg_functor {
    // Ctor.
    inline __device__ Ldg_functor(Data_type fetch, const void* ptrs)
    : fetch_(fetch), ptrs_(ptrs) {
    }

    // Clear the element.
    inline __device__ void clear(int ii) {
        clear_(fetch_);
    }

    // Trigger the loads.
    inline __device__ void load(int ii, bool p) {
        if( p ) {
            ldg(fetch_, ptrs_);
        }
    }

    // The fetch registers.
    Data_type fetch_;
    // The pointers.
    const void* ptrs_;
};