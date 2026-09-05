#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "clean_sar_cuda.h"
#include "psf_math.cuh"
#include "reduction.cuh"

#define REDUCTION_BLOCK_SIZE 256
#define NUM_REDUCTION_BLOCKS 64

/**
 * Fused On-the-Fly PSF Evaluation & Residual/Model Update Kernel.
 *
 * For the peak at (r0, c0), this kernel evaluates the exact rotated PSF and clean restoring beam
 * across the (psf_size x psf_size) window and updates d_residual and d_model in a single pass.
 */
__global__ void fused_sub_add_kernel(
    clean_complex_t* __restrict__ d_residual,
    clean_complex_t* __restrict__ d_model,
    int32_t height,
    int32_t width,
    int32_t r0,
    int32_t c0,
    clean_complex_t comp,
    float cos_t,
    float sin_t,
    float sigma_r,
    float sigma_c,
    clean_physics_config_t config,
    int32_t psf_size,
    int32_t beam_type
) {
    int kh = psf_size / 2;
    int kw = psf_size / 2;

    int dr = (int)(blockIdx.x * blockDim.x + threadIdx.x) - kh;
    int dc = (int)(blockIdx.y * blockDim.y + threadIdx.y) - kw;

    if (abs(dr) > kh || abs(dc) > kw) {
        return;
    }

    int img_r = r0 + dr;
    int img_c = c0 + dc;

    if (img_r < 0 || img_r >= height || img_c < 0 || img_c >= width) {
        return;
    }

    // Metric coordinate offsets (meters)
    float u = (float)dr * config.row_ss;
    float v = (float)dc * config.col_ss;

    // Rotate coordinates by local polar shear angle theta
    float u_prime =  u * cos_t + v * sin_t;
    float v_prime = -u * sin_t + v * cos_t;

    // Evaluate exact dirty PSF (modulated sincs with windowing)
    float psf_r = eval_1d_window_sinc(u_prime, config.row_bw, config.row_wgt_type);
    float psf_c = eval_1d_window_sinc(v_prime, config.col_bw, config.col_wgt_type);
    float h_dirty = psf_r * psf_c;

    // Evaluate exact clean restoring beam (matched 3dB Gaussian)
    float h_clean = eval_clean_beam_gaussian(u_prime, v_prime, sigma_r, sigma_c);

    // Subtract from residual: Residual -= comp * h_dirty
    int idx = img_r * width + img_c;
    d_residual[idx].real -= comp.real * h_dirty;
    d_residual[idx].imag -= comp.imag * h_dirty;

    // Add to restored model: Model += comp * h_clean
    d_model[idx].real += comp.real * h_clean;
    d_model[idx].imag += comp.imag * h_clean;
}

/**
 * Elementwise Image Synthesis Kernel: CleanImage = Model + Residual.
 */
__global__ void synthesize_clean_image_kernel(
    const clean_complex_t* __restrict__ d_residual,
    const clean_complex_t* __restrict__ d_model,
    clean_complex_t* __restrict__ d_clean,
    int32_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        d_clean[idx].real = d_model[idx].real + d_residual[idx].real;
        d_clean[idx].imag = d_model[idx].imag + d_residual[idx].imag;
    }
}

/**
 * Main C/CUDA Entrypoint.
 */
int run_hogbom_clean_cuda(
    const clean_complex_t*  h_dirty_image,
    clean_complex_t*        h_clean_image_out,
    clean_complex_t*        h_residual_out,
    clean_complex_t*        h_model_out,
    int32_t                 height,
    int32_t                 width,
    clean_physics_config_t  config,
    clean_loop_options_t    options,
    clean_metrics_t*        metrics_out
) {
    if (!h_dirty_image || !h_clean_image_out || height <= 0 || width <= 0) {
        return -1;
    }

    size_t total_elements = (size_t)height * width;
    size_t img_bytes = total_elements * sizeof(clean_complex_t);

    // Device memory pointers
    clean_complex_t* d_residual = NULL;
    clean_complex_t* d_model = NULL;
    clean_complex_t* d_clean = NULL;
    PeakPair* d_block_peaks = NULL;
    PeakPair* d_global_peak = NULL;

    cudaError_t err;

    err = cudaMalloc((void**)&d_residual, img_bytes);
    if (err != cudaSuccess) return (int)err;

    err = cudaMalloc((void**)&d_model, img_bytes);
    if (err != cudaSuccess) { cudaFree(d_residual); return (int)err; }

    err = cudaMalloc((void**)&d_clean, img_bytes);
    if (err != cudaSuccess) { cudaFree(d_residual); cudaFree(d_model); return (int)err; }

    err = cudaMalloc((void**)&d_block_peaks, NUM_REDUCTION_BLOCKS * sizeof(PeakPair));
    if (err != cudaSuccess) { cudaFree(d_residual); cudaFree(d_model); cudaFree(d_clean); return (int)err; }

    err = cudaMalloc((void**)&d_global_peak, sizeof(PeakPair));
    if (err != cudaSuccess) {
        cudaFree(d_residual); cudaFree(d_model); cudaFree(d_clean); cudaFree(d_block_peaks);
        return (int)err;
    }

    // Initialize device memory
    cudaMemcpy(d_residual, h_dirty_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_model, 0, img_bytes);

    // CUDA timing events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);

    // Precompute Gaussian beam parameters
    float sigma_r = config.row_wid / FWHM_GAUSS_FACTOR;
    float sigma_c = config.col_wid / FWHM_GAUSS_FACTOR;

    int psf_size = options.psf_size > 0 ? options.psf_size : 65;
    if (psf_size % 2 == 0) psf_size += 1;

    dim3 psf_block(16, 16);
    dim3 psf_grid((psf_size + psf_block.x - 1) / psf_block.x,
                  (psf_size + psf_block.y - 1) / psf_block.y);

    // Find initial peak
    find_block_peaks_kernel<REDUCTION_BLOCK_SIZE><<<NUM_REDUCTION_BLOCKS, REDUCTION_BLOCK_SIZE>>>(
        d_residual, height, width, options.guard_margin, d_block_peaks
    );
    find_global_peak_kernel<REDUCTION_BLOCK_SIZE><<<1, REDUCTION_BLOCK_SIZE>>>(
        d_block_peaks, NUM_REDUCTION_BLOCKS, d_global_peak
    );

    PeakPair h_peak;
    cudaMemcpy(&h_peak, d_global_peak, sizeof(PeakPair), cudaMemcpyDeviceToHost);

    float initial_peak = h_peak.mag;
    float stop_thresh = (options.threshold < 1.0f) ? (options.threshold * initial_peak) : options.threshold;

    if (options.verbose) {
        printf("[CUDA CLEAN] Initial peak: %.4e, Stopping threshold: %.4e, Max iters: %d\n",
               initial_peak, stop_thresh, options.max_iters);
    }

    int iter = 0;
    int converged = 0;

    // Main Hogbom CLEAN loop
    while (iter < options.max_iters) {
        // Run parallel peak reduction
        find_block_peaks_kernel<REDUCTION_BLOCK_SIZE><<<NUM_REDUCTION_BLOCKS, REDUCTION_BLOCK_SIZE>>>(
            d_residual, height, width, options.guard_margin, d_block_peaks
        );
        find_global_peak_kernel<REDUCTION_BLOCK_SIZE><<<1, REDUCTION_BLOCK_SIZE>>>(
            d_block_peaks, NUM_REDUCTION_BLOCKS, d_global_peak
        );

        cudaMemcpy(&h_peak, d_global_peak, sizeof(PeakPair), cudaMemcpyDeviceToHost);

        if (h_peak.mag <= stop_thresh || h_peak.mag <= 0.0f) {
            converged = 1;
            if (options.verbose) {
                printf("[CUDA CLEAN] Converged at iteration %d: peak %.4e <= %.4e\n",
                       iter, h_peak.mag, stop_thresh);
            }
            break;
        }

        int r0 = h_peak.idx / width;
        int c0 = h_peak.idx % width;

        // Retrieve complex amplitude at peak
        clean_complex_t amp;
        cudaMemcpy(&amp, &d_residual[h_peak.idx], sizeof(clean_complex_t), cudaMemcpyDeviceToHost);

        clean_complex_t comp;
        comp.real = options.gain * amp.real;
        comp.imag = options.gain * amp.imag;

        // Compute local polar shear angle
        float r_g = (float)(r0 + config.chip_start_row);
        float c_g = (float)(c0 + config.chip_start_col);
        float x_row = (r_g - config.scp_row) * config.row_ss;
        float y_col = (c_g - config.scp_col) * config.col_ss;
        float theta = atan2f(y_col, config.scp_slant_range + x_row);
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        // Fused PSF evaluation & subtraction/addition kernel
        fused_sub_add_kernel<<<psf_grid, psf_block>>>(
            d_residual,
            d_model,
            height,
            width,
            r0,
            c0,
            comp,
            cos_t,
            sin_t,
            sigma_r,
            sigma_c,
            config,
            psf_size,
            options.beam_type
        );

        iter++;
        if (options.verbose && iter % 250 == 0) {
            printf("  Iter %5d: Current max residual = %.4e (Peak at [%d, %d])\n",
                   iter, h_peak.mag, r0, c0);
        }
    }

    // Synthesize final clean image: Clean = Model + Residual
    int block_sz = 256;
    int grid_sz = (int)((total_elements + block_sz - 1) / block_sz);
    synthesize_clean_image_kernel<<<grid_sz, block_sz>>>(d_residual, d_model, d_clean, (int32_t)total_elements);

    // Stop timer
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);

    // Copy results back to host
    cudaMemcpy(h_clean_image_out, d_clean, img_bytes, cudaMemcpyDeviceToHost);
    if (h_residual_out) {
        cudaMemcpy(h_residual_out, d_residual, img_bytes, cudaMemcpyDeviceToHost);
    }
    if (h_model_out) {
        cudaMemcpy(h_model_out, d_model, img_bytes, cudaMemcpyDeviceToHost);
    }

    // Populate metrics
    if (metrics_out) {
        metrics_out->iterations = iter;
        metrics_out->execution_time_ms = elapsed_ms;
        metrics_out->initial_peak = initial_peak;
        metrics_out->final_peak = h_peak.mag;
        metrics_out->suppression_db = (initial_peak > 0 && h_peak.mag > 0)
            ? (20.0f * log10f(initial_peak / h_peak.mag)) : 0.0f;
        metrics_out->num_components = iter;
        metrics_out->converged = converged;
    }

    // Free device memory
    cudaFree(d_residual);
    cudaFree(d_model);
    cudaFree(d_clean);
    cudaFree(d_block_peaks);
    cudaFree(d_global_peak);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
