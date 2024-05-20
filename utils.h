

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

inline __device__ void clear(uint4 &dst) {
    dst = make_uint4(0u, 0u, 0u, 0u);
}

inline __device__ void ldg(uint4 &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint4*>(ptr);
}

template< typename Data_type, int N >
struct Ldg_functor {
    // Ctor.
    inline __device__ Ldg_functor(Data_type (&fetch)[N], const void* (&ptrs)[N])
    : fetch_(fetch), ptrs_(ptrs) {
    }

    // Clear the element.
    inline __device__ void clear(int ii) {
        clear_(fetch_[ii]);
    }

    // Trigger the loads.
    inline __device__ void load(int ii, bool p) {
        if( p ) {
            ldg(fetch_[ii], ptrs_[ii]);
        }
    }

    // The fetch registers.
    Data_type (&fetch_)[N];
    // The pointers.
    const void* (&ptrs_)[N];
};