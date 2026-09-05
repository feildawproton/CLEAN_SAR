#define PI 3.14159265358979323846f

typedef struct {
    float real;
    float imag;
} clean_complex_t;

typedef struct {
    float row_ss;
    float col_ss;
    float row_bw;
    float col_bw;
    float row_wid;
    float col_wid;
    float scp_slant_range;
    float scp_row;
    float scp_col;
    int   chip_start_row;
    int   chip_start_col;
    int   row_wgt_type;  // 0: Uniform, 1: Taylor, 2: Hamming, 3: Hann
    int   col_wgt_type;
} clean_physics_config_t;

// Fast in-register normalized sinc
__device__ __forceinline__ float dev_sinc(float x) {
    if (fabsf(x) < 1e-7f) {
        return 1.0f;
    }
    float px = PI * x;
    return sinf(px) / px;
}

// Exact 1D modulated sinc pattern with aperture windowing
__device__ __forceinline__ float eval_1d_window_sinc(float pos, float bw, int wgt_type) {
    float x = bw * pos;
    if (wgt_type == 0) { // UNIFORM
        return dev_sinc(x);
    } else if (wgt_type == 1) { // TAYLOR (nbar=4, SLL=-30dB)
        const float f1 = 0.29265601f;
        const float f2 = -0.01578375f;
        const float f3 = 0.00218104f;
        return dev_sinc(x)
             + f1 * (dev_sinc(x - 1.0f) + dev_sinc(x + 1.0f))
             + f2 * (dev_sinc(x - 2.0f) + dev_sinc(x + 2.0f))
             + f3 * (dev_sinc(x - 3.0f) + dev_sinc(x + 3.0f));
    } else if (wgt_type == 2) { // HAMMING (normalized to 1.0 at center: / 0.54)
        return (0.54f * dev_sinc(x) + 0.23f * (dev_sinc(x - 1.0f) + dev_sinc(x + 1.0f))) * (1.0f / 0.54f);
    } else if (wgt_type == 3) { // HANN (normalized to 1.0 at center: / 0.50)
        return (0.50f * dev_sinc(x) + 0.25f * (dev_sinc(x - 1.0f) + dev_sinc(x + 1.0f))) * 2.0f;
    }
    return dev_sinc(x);
}

// Matched -3dB resolution Gaussian restoring beam
__device__ __forceinline__ float eval_clean_beam_gaussian(float u_prime, float v_prime, float sigma_r, float sigma_c) {
    float r_term = u_prime / fmaxf(sigma_r, 1e-6f);
    float c_term = v_prime / fmaxf(sigma_c, 1e-6f);
    return expf(-0.5f * (r_term * r_term + c_term * c_term));
}

// --------------------------------------------------------------------------
// 1. Parallel Argmax Reduction Kernels
// --------------------------------------------------------------------------

struct ValIdx {
    float val;
    int   idx;
};

__device__ __forceinline__ ValIdx warp_reduce_max(ValIdx v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        if (other_val > v.val) {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

extern "C" __global__ void reduce_max_pass1_kernel(
    const clean_complex_t* __restrict__ d_residual,
    float* __restrict__ d_block_vals,
    int*   __restrict__ d_block_idxs,
    int total_elements,
    int height,
    int width,
    int guard_margin
) {
    __shared__ ValIdx shared_val_idx[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    ValIdx local_max = {-1.0f, -1};

    for (int idx = blockIdx.x * blockDim.x + tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        if (guard_margin > 0) {
            int r = idx / width;
            int c = idx % width;
            if (r < guard_margin || r >= height - guard_margin || c < guard_margin || c >= width - guard_margin) {
                continue;
            }
        }
        float re = d_residual[idx].real;
        float im = d_residual[idx].imag;
        float mag = sqrtf(re * re + im * im);
        if (mag > local_max.val) {
            local_max.val = mag;
            local_max.idx = idx;
        }
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) {
        shared_val_idx[wid] = local_max;
    }
    __syncthreads();

    if (wid == 0) {
        ValIdx block_max = (lane < (blockDim.x >> 5)) ? shared_val_idx[lane] : ValIdx{-1.0f, -1};
        block_max = warp_reduce_max(block_max);
        if (lane == 0) {
            d_block_vals[blockIdx.x] = block_max.val;
            d_block_idxs[blockIdx.x] = block_max.idx;
        }
    }
}

extern "C" __global__ void reduce_max_pass2_kernel(
    const float* __restrict__ d_block_vals,
    const int*   __restrict__ d_block_idxs,
    float* __restrict__ d_out_val,
    int*   __restrict__ d_out_idx,
    int num_blocks
) {
    __shared__ ValIdx shared_val_idx[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    ValIdx local_max = {-1.0f, -1};

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = d_block_vals[i];
        if (val > local_max.val) {
            local_max.val = val;
            local_max.idx = d_block_idxs[i];
        }
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) {
        shared_val_idx[wid] = local_max;
    }
    __syncthreads();

    if (wid == 0) {
        ValIdx global_max = (lane < (blockDim.x >> 5)) ? shared_val_idx[lane] : ValIdx{-1.0f, -1};
        global_max = warp_reduce_max(global_max);
        if (lane == 0) {
            *d_out_val = global_max.val;
            *d_out_idx = global_max.idx;
        }
    }
}

// --------------------------------------------------------------------------
// 2. Fused Subtraction & Restoration Kernel
// --------------------------------------------------------------------------

extern "C" __global__ void fused_clean_sub_add_kernel(
    clean_complex_t* __restrict__ d_residual,
    clean_complex_t* __restrict__ d_model,
    clean_complex_t* __restrict__ d_components,
    int height,
    int width,
    int r0,
    int c0,
    float comp_re,
    float comp_im,
    float cos_t,
    float sin_t,
    float sigma_r,
    float sigma_c,
    float row_ss,
    float col_ss,
    float row_bw,
    float col_bw,
    int   row_wgt_type,
    int   col_wgt_type,
    int   psf_size
) {
    int kh = psf_size / 2;
    int kw = psf_size / 2;

    int dr = (int)(blockIdx.x * blockDim.x + threadIdx.x) - kh;
    int dc = (int)(blockIdx.y * blockDim.y + threadIdx.y) - kw;

    if (dr < -kh || dr > kh || dc < -kw || dc > kw) {
        return;
    }

    int img_r = r0 + dr;
    int img_c = c0 + dc;

    if (img_r < 0 || img_r >= height || img_c < 0 || img_c >= width) {
        return;
    }

    // Metric spatial coordinates (meters)
    float u = (float)dr * row_ss;
    float v = (float)dc * col_ss;

    // Rotate by local polar shear angle theta
    float u_prime =  u * cos_t + v * sin_t;
    float v_prime = -u * sin_t + v * cos_t;

    // Exact dirty PSF evaluation
    float psf_r = eval_1d_window_sinc(u_prime, row_bw, row_wgt_type);
    float psf_c = eval_1d_window_sinc(v_prime, col_bw, col_wgt_type);
    float h_dirty = psf_r * psf_c;

    // Exact clean restoring beam evaluation
    float h_clean = eval_clean_beam_gaussian(u_prime, v_prime, sigma_r, sigma_c);

    int idx = img_r * width + img_c;

    // Update residual and model in-place
    d_residual[idx].real -= comp_re * h_dirty;
    d_residual[idx].imag -= comp_im * h_dirty;

    d_model[idx].real += comp_re * h_clean;
    d_model[idx].imag += comp_im * h_clean;

    if (dr == 0 && dc == 0 && d_components != 0) {
        d_components[idx].real += comp_re;
        d_components[idx].imag += comp_im;
    }
}

// --------------------------------------------------------------------------
// 3. Final Synthesis Kernel (Clean = Model + Residual)
// --------------------------------------------------------------------------

extern "C" __global__ void synthesize_clean_image_kernel(
    const clean_complex_t* __restrict__ d_residual,
    const clean_complex_t* __restrict__ d_model,
    clean_complex_t*       __restrict__ d_clean,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        d_clean[idx].real = d_model[idx].real + d_residual[idx].real;
        d_clean[idx].imag = d_model[idx].imag + d_residual[idx].imag;
    }
}
