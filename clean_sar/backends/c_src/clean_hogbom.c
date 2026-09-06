#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static inline float c_sinc(float x) {
    if (fabsf(x) < 1e-7f) {
        return 1.0f;
    }
    float px = PI * x;
    return sinf(px) / px;
}

static inline float eval_1d_window_sinc(float pos, float bw, int wgt_type) {
    float x = bw * pos;
    if (wgt_type == 0) { // UNIFORM
        return c_sinc(x);
    } else if (wgt_type == 1) { // TAYLOR (nbar=4, SLL=-30dB)
        const float f1 = 0.29265601f;
        const float f2 = -0.01578375f;
        const float f3 = 0.00218104f;
        return c_sinc(x)
             + f1 * (c_sinc(x - 1.0f) + c_sinc(x + 1.0f))
             + f2 * (c_sinc(x - 2.0f) + c_sinc(x + 2.0f))
             + f3 * (c_sinc(x - 3.0f) + c_sinc(x + 3.0f));
    } else if (wgt_type == 2) { // HAMMING (normalized to 1.0 at center: / 0.54)
        return (0.54f * c_sinc(x) + 0.23f * (c_sinc(x - 1.0f) + c_sinc(x + 1.0f))) * (1.0f / 0.54f);
    } else if (wgt_type == 3) { // HANN (normalized to 1.0 at center: / 0.50)
        return (0.50f * c_sinc(x) + 0.25f * (c_sinc(x - 1.0f) + c_sinc(x + 1.0f))) * 2.0f;
    }
    return c_sinc(x);
}

static inline float eval_clean_beam_gaussian(float u_prime, float v_prime, float sigma_r, float sigma_c) {
    float r_term = u_prime / (sigma_r > 1e-6f ? sigma_r : 1e-6f);
    float c_term = v_prime / (sigma_c > 1e-6f ? sigma_c : 1e-6f);
    return expf(-0.5f * (r_term * r_term + c_term * c_term));
}

#if defined(_WIN32)
#define CLEAN_EXPORT __declspec(dllexport)
#else
#define CLEAN_EXPORT __attribute__((visibility("default")))
#endif

CLEAN_EXPORT int run_hogbom_c(
    const clean_complex_t* dirty_image,
    int height,
    int width,
    const clean_physics_config_t* config,
    int psf_size,
    float gain,
    float threshold,
    int max_iters,
    int guard_margin,
    clean_complex_t* out_clean,
    clean_complex_t* out_residual,
    clean_complex_t* out_model,
    clean_complex_t* out_components,
    int* out_iters,
    float* history_peaks,
    int* history_coords_r,
    int* history_coords_c
) {
    if (!dirty_image || !config || !out_clean || !out_residual || !out_model || !out_components) {
        return -1;
    }

    int total_elements = height * width;
    memcpy(out_residual, dirty_image, total_elements * sizeof(clean_complex_t));
    memset(out_model, 0, total_elements * sizeof(clean_complex_t));
    memset(out_components, 0, total_elements * sizeof(clean_complex_t));

    const float fwhm_const = 2.0f * sqrtf(logf(2.0f));
    float sigma_r = config->row_wid / fwhm_const;
    float sigma_c = config->col_wid / fwhm_const;

    int kh = psf_size / 2;
    int kw = psf_size / 2;

    // Initial peak finding
    float max_val = -1.0f;
    int max_idx = -1;

    for (int idx = 0; idx < total_elements; ++idx) {
        if (guard_margin > 0) {
            int r = idx / width;
            int c = idx % width;
            if (r < guard_margin || r >= height - guard_margin ||
                c < guard_margin || c >= width - guard_margin) {
                continue;
            }
        }
        float re = out_residual[idx].real;
        float im = out_residual[idx].imag;
        float mag = sqrtf(re * re + im * im);
        if (mag > max_val) {
            max_val = mag;
            max_idx = idx;
        }
    }

    float init_peak = max_val;
    float stop_thresh = (threshold < 1.0f) ? (threshold * init_peak) : threshold;

    int it = 0;
    while (it < max_iters) {
        if (history_peaks) {
            history_peaks[it] = max_val;
        }

        if (max_val <= stop_thresh || max_val == 0.0f || max_idx < 0) {
            break;
        }

        int r0 = max_idx / width;
        int c0 = max_idx % width;

        if (history_coords_r && history_coords_c) {
            history_coords_r[it] = r0;
            history_coords_c[it] = c0;
        }

        float amp_re = out_residual[max_idx].real;
        float amp_im = out_residual[max_idx].imag;

        float comp_re = gain * amp_re;
        float comp_im = gain * amp_im;

        float rg = (float)r0 + (float)config->chip_start_row;
        float cg = (float)c0 + (float)config->chip_start_col;
        float xr = (rg - config->scp_row) * config->row_ss;
        float yc = (cg - config->scp_col) * config->col_ss;
        float theta = atan2f(yc, config->scp_slant_range + xr);
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        for (int dr = -kh; dr <= kh; ++dr) {
            int img_r = r0 + dr;
            if (img_r < 0 || img_r >= height) continue;

            for (int dc = -kw; dc <= kw; ++dc) {
                int img_c = c0 + dc;
                if (img_c < 0 || img_c >= width) continue;

                float u = (float)dr * config->row_ss;
                float v = (float)dc * config->col_ss;

                float u_prime =  u * cos_t + v * sin_t;
                float v_prime = -u * sin_t + v * cos_t;

                float psf_r = eval_1d_window_sinc(u_prime, config->row_bw, config->row_wgt_type);
                float psf_c = eval_1d_window_sinc(v_prime, config->col_bw, config->col_wgt_type);
                float h_dirty = psf_r * psf_c;

                float h_clean = eval_clean_beam_gaussian(u_prime, v_prime, sigma_r, sigma_c);

                int idx = img_r * width + img_c;
                out_residual[idx].real -= comp_re * h_dirty;
                out_residual[idx].imag -= comp_im * h_dirty;

                out_model[idx].real += comp_re * h_clean;
                out_model[idx].imag += comp_im * h_clean;

                if (dr == 0 && dc == 0) {
                    out_components[idx].real += comp_re;
                    out_components[idx].imag += comp_im;
                }
            }
        }

        it++;

        // Next peak finding
        max_val = -1.0f;
        max_idx = -1;
        for (int idx = 0; idx < total_elements; ++idx) {
            if (guard_margin > 0) {
                int r = idx / width;
                int c = idx % width;
                if (r < guard_margin || r >= height - guard_margin ||
                    c < guard_margin || c >= width - guard_margin) {
                    continue;
                }
            }
            float re = out_residual[idx].real;
            float im = out_residual[idx].imag;
            float mag = sqrtf(re * re + im * im);
            if (mag > max_val) {
                max_val = mag;
                max_idx = idx;
            }
        }
    }

    if (out_iters) {
        *out_iters = it;
    }

    // Synthesize clean image: clean = model + residual
    for (int idx = 0; idx < total_elements; ++idx) {
        out_clean[idx].real = out_model[idx].real + out_residual[idx].real;
        out_clean[idx].imag = out_model[idx].imag + out_residual[idx].imag;
    }

    return 0;
}
