#ifndef CLEAN_SAR_CUDA_H
#define CLEAN_SAR_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * 2D Single-Precision Complex Number representation.
 * Binary compatible with PyTorch / NumPy complex64 and CUDA cuComplex / float2.
 */
typedef struct {
    float real;
    float imag;
} clean_complex_t;

/**
 * Window weighting enum for SAR spectral aperture.
 */
typedef enum {
    CLEAN_WGT_UNIFORM = 0,
    CLEAN_WGT_TAYLOR  = 1,
    CLEAN_WGT_HAMMING = 2,
    CLEAN_WGT_HANN    = 3
} clean_window_type_t;

/**
 * Clean restoring beam type.
 */
typedef enum {
    CLEAN_BEAM_GAUSSIAN = 0,
    CLEAN_BEAM_MAINLOBE = 1
} clean_beam_type_t;

/**
 * Scalar SAR physics configuration defining resolution, geometry, and coordinates.
 * Binary compatible with Python CleanPhysicsConfig.
 */
typedef struct {
    float row_ss;             /* Row sample spacing in meters */
    float col_ss;             /* Column sample spacing in meters */
    float row_bw;             /* Row impulse response bandwidth in cycles/meter */
    float col_bw;             /* Column impulse response bandwidth in cycles/meter */
    float row_wid;            /* Row -3dB resolution width in meters */
    float col_wid;            /* Column -3dB resolution width in meters */
    float scp_slant_range;    /* Slant range R0 from sensor to SCP in meters */
    float scp_row;            /* Global Scene Center Point row pixel index */
    float scp_col;            /* Global Scene Center Point column pixel index */
    int32_t chip_start_row;   /* Sub-image chip start row offset (global) */
    int32_t chip_start_col;   /* Sub-image chip start col offset (global) */
    int32_t row_wgt_type;     /* clean_window_type_t */
    int32_t col_wgt_type;     /* clean_window_type_t */
} clean_physics_config_t;

/**
 * Execution parameters and stopping criteria for Complex Hogbom CLEAN.
 */
typedef struct {
    float   gain;             /* Loop damping factor gamma (e.g. 0.1) */
    float   threshold;        /* Stopping threshold fraction (e.g. 0.02) */
    int32_t max_iters;        /* Maximum number of iterations (e.g. 2500) */
    int32_t psf_size;         /* PSF kernel dimensions (must be odd, e.g. 65) */
    int32_t guard_margin;     /* Border margin excluded from peak search */
    int32_t beam_type;        /* clean_beam_type_t */
    int32_t verbose;          /* 1: Print periodic progress, 0: Quiet */
} clean_loop_options_t;

/**
 * Execution results and convergence metrics.
 */
typedef struct {
    int32_t iterations;       /* Total iterations executed */
    float   execution_time_ms;/* Kernel execution time in milliseconds */
    float   initial_peak;     /* Maximum magnitude of dirty image at start */
    float   final_peak;       /* Maximum residual magnitude at termination */
    float   suppression_db;   /* Total peak sidelobe suppression in dB */
    int32_t num_components;   /* Number of point scatterers extracted */
    int32_t converged;        /* 1 if stopped by threshold, 0 if hit max_iters */
} clean_metrics_t;

/**
 * Main C/CUDA entrypoint for Complex Hogbom CLEAN deconvolution.
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
);

#ifdef __cplusplus
}
#endif

#endif /* CLEAN_SAR_CUDA_H */
