#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "clean_sar_cuda.h"

struct PeakPair {
    float   mag;
    int32_t idx;
};

/**
 * Warp-level parallel argmax reduction using __shfl_down_sync.
 */
__device__ __forceinline__ PeakPair warp_reduce_max(PeakPair val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float   other_mag = __shfl_down_sync(0xffffffff, val.mag, offset);
        int32_t other_idx = __shfl_down_sync(0xffffffff, val.idx, offset);
        if (other_mag > val.mag) {
            val.mag = other_mag;
            val.idx = other_idx;
        }
    }
    return val;
}

/**
 * Block-level parallel argmax reduction across 256 or 512 threads using shared memory.
 */
template <int BLOCK_SIZE>
__device__ __forceinline__ PeakPair block_reduce_max(PeakPair val) {
    __shared__ PeakPair shared_peaks[32]; // Max 32 warps per block (1024 threads)
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared_peaks[wid] = val;
    }
    __syncthreads();

    // First warp reduces the per-warp maximums
    int num_warps = BLOCK_SIZE / 32;
    PeakPair block_val = { -1.0f, -1 };
    if (threadIdx.x < num_warps) {
        block_val = shared_peaks[lane];
    }
    if (wid == 0) {
        block_val = warp_reduce_max(block_val);
    }
    return block_val;
}

/**
 * Global Grid Reduction Kernel: Computes per-block maximum peak magnitude and flat index.
 */
template <int BLOCK_SIZE>
__global__ void find_block_peaks_kernel(
    const clean_complex_t* __restrict__ d_residual,
    int32_t height,
    int32_t width,
    int32_t guard_margin,
    PeakPair* __restrict__ d_block_peaks
) {
    int total_elements = height * width;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    PeakPair local_max = { -1.0f, -1 };

    for (int idx = tid; idx < total_elements; idx += stride) {
        int r = idx / width;
        int c = idx % width;

        if (guard_margin > 0) {
            if (r < guard_margin || r >= height - guard_margin ||
                c < guard_margin || c >= width - guard_margin) {
                continue;
            }
        }

        clean_complex_t val = d_residual[idx];
        float mag = sqrtf(val.real * val.real + val.imag * val.imag);

        if (mag > local_max.mag) {
            local_max.mag = mag;
            local_max.idx = idx;
        }
    }

    PeakPair block_max = block_reduce_max<BLOCK_SIZE>(local_max);

    if (threadIdx.x == 0) {
        d_block_peaks[blockIdx.x] = block_max;
    }
}

/**
 * Final Reduction Kernel: Reduces per-block peaks to single global maximum.
 */
template <int BLOCK_SIZE>
__global__ void find_global_peak_kernel(
    const PeakPair* __restrict__ d_block_peaks,
    int32_t num_blocks,
    PeakPair* __restrict__ d_global_peak
) {
    PeakPair local_max = { -1.0f, -1 };

    for (int i = threadIdx.x; i < num_blocks; i += BLOCK_SIZE) {
        PeakPair p = d_block_peaks[i];
        if (p.mag > local_max.mag) {
            local_max = p;
        }
    }

    PeakPair global_max = block_reduce_max<BLOCK_SIZE>(local_max);

    if (threadIdx.x == 0) {
        *d_global_peak = global_max;
    }
}

#endif /* REDUCTION_CUH */
