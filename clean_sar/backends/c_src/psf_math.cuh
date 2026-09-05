#ifndef PSF_MATH_CUH
#define PSF_MATH_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "clean_sar_cuda.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define FWHM_GAUSS_FACTOR 1.6651092221997054f /* 2.0 * sqrt(ln(2.0)) */

/**
 * Evaluates normalized sinc(x) = sin(pi * x) / (pi * x).
 */
__device__ __forceinline__ float eval_sinc(float x) {
    float abs_x = fabsf(x);
    if (abs_x < 1e-7f) {
        return 1.0f;
    }
    float px = (float)M_PI * x;
    return sinf(px) / px;
}

/**
 * Evaluates 1D spatial-domain pattern with window weighting (modulated sinc series).
 */
__device__ __forceinline__ float eval_1d_window_sinc(float pos, float bw, int32_t wgt_type) {
    float u = bw * pos;
    if (wgt_type == CLEAN_WGT_UNIFORM) {
        return eval_sinc(u);
    } else if (wgt_type == CLEAN_WGT_TAYLOR) {
        /* Taylor window (nbar=4, SLL=-30dB) */
        const float f1 = 0.29265601f;
        const float f2 = -0.01578375f;
        const float f3 = 0.00218104f;
        float pat = eval_sinc(u);
        pat += f1 * (eval_sinc(u - 1.0f) + eval_sinc(u + 1.0f));
        pat += f2 * (eval_sinc(u - 2.0f) + eval_sinc(u + 2.0f));
        pat += f3 * (eval_sinc(u - 3.0f) + eval_sinc(u + 3.0f));
        return pat;
    } else if (wgt_type == CLEAN_WGT_HAMMING) {
        return 0.54f * eval_sinc(u) + 0.23f * (eval_sinc(u - 1.0f) + eval_sinc(u + 1.0f));
    } else if (wgt_type == CLEAN_WGT_HANN) {
        return 0.50f * eval_sinc(u) + 0.25f * (eval_sinc(u - 1.0f) + eval_sinc(u + 1.0f));
    }
    return eval_sinc(u);
}

/**
 * Computes local polar shear angle theta and its trigonometric components (cos_t, sin_t).
 */
__device__ __forceinline__ void compute_local_shear(
    int32_t r0,
    int32_t c0,
    const clean_physics_config_t& config,
    float* out_cos_t,
    float* out_sin_t
) {
    float r_g = (float)(r0 + config.chip_start_row);
    float c_g = (float)(c0 + config.chip_start_col);

    float x_row = (r_g - config.scp_row) * config.row_ss;
    float y_col = (c_g - config.scp_col) * config.col_ss;

    float theta = atan2f(y_col, config.scp_slant_range + x_row);
    __sincosf(theta, out_sin_t, out_cos_t);
}

/**
 * Evaluates the Gaussian restoring clean beam matched to -3dB resolution width.
 */
__device__ __forceinline__ float eval_clean_beam_gaussian(
    float u_prime,
    float v_prime,
    float sigma_r,
    float sigma_c
) {
    float term_r = u_prime / (sigma_r > 1e-6f ? sigma_r : 1e-6f);
    float term_c = v_prime / (sigma_c > 1e-6f ? sigma_c : 1e-6f);
    return expf(-0.5f * (term_r * term_r + term_c * term_c));
}

#endif /* PSF_MATH_CUH */
