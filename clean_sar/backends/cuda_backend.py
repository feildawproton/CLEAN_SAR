"""
cuda_backend.py: Native CUDA implementation of Complex Hogbom CLEAN deconvolution
compiled directly to PTX/CUBIN via NVIDIA Runtime Compiler (NVRTC) and launched via CUDA Driver API.
"""

import os
import sys
import math
import time
import ctypes
from typing import Optional, Union, Tuple, List, Literal
import numpy as np

from ..config import CleanPhysicsConfig


# ---------------------------------------------------------------------------
# NVRTC & CUDA Driver Singleton Manager
# ---------------------------------------------------------------------------

_NVRTC_LIB = None
_CUDA_LIB = None
_CUDA_CTX = None
_CUDA_MODULE = None

_CUDA_KERNELS = {}

NVRTC_CANDIDATE_PATHS = [
    "/home/feildaw/mypyenv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
    "libnvrtc.so.12",
    "libnvrtc.so",
]


def _init_cuda_driver():
    global _NVRTC_LIB, _CUDA_LIB, _CUDA_CTX, _CUDA_MODULE, _CUDA_KERNELS

    if _CUDA_MODULE is not None:
        _CUDA_LIB.cuCtxSetCurrent(_CUDA_CTX)
        return True

    # 1. Load NVRTC
    for p in NVRTC_CANDIDATE_PATHS:
        try:
            _NVRTC_LIB = ctypes.CDLL(p)
            break
        except Exception:
            continue

    if _NVRTC_LIB is None:
        return False

    # 2. Load CUDA Driver
    try:
        _CUDA_LIB = ctypes.CDLL("libcuda.so.1")
    except Exception:
        try:
            _CUDA_LIB = ctypes.CDLL("libcuda.so")
        except Exception:
            return False

    # Initialize Driver & Context
    if _CUDA_LIB.cuInit(0) != 0:
        return False

    dev = ctypes.c_int()
    if _CUDA_LIB.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        return False

    ctx = ctypes.c_void_p()
    # Check if existing context exists
    if _CUDA_LIB.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 and ctx.value is not None:
        _CUDA_CTX = ctx
    else:
        if _CUDA_LIB.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
            return False
        _CUDA_CTX = ctx

    _CUDA_LIB.cuCtxSetCurrent(_CUDA_CTX)

    # 3. Read CUDA Source and Compile via NVRTC
    cu_file = os.path.join(os.path.dirname(__file__), "c_src", "clean_hogbom.cu")
    if not os.path.isfile(cu_file):
        return False

    with open(cu_file, "rb") as f:
        cuda_src = f.read()

    prog = ctypes.c_void_p()
    if _NVRTC_LIB.nvrtcCreateProgram(ctypes.byref(prog), cuda_src, b"clean_hogbom.cu", 0, None, None) != 0:
        return False

    opts = [b"--std=c++14", b"--gpu-architecture=compute_86"]
    opts_arr = (ctypes.c_char_p * len(opts))(*opts)
    compile_res = _NVRTC_LIB.nvrtcCompileProgram(prog, len(opts), opts_arr)
    if compile_res != 0:
        return False

    ptx_size = ctypes.c_size_t()
    _NVRTC_LIB.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    _NVRTC_LIB.nvrtcGetPTX(prog, ptx)

    # 4. Load PTX Module
    module = ctypes.c_void_p()
    if _CUDA_LIB.cuModuleLoadData(ctypes.byref(module), ptx) != 0:
        return False
    _CUDA_MODULE = module

    # 5. Extract Kernel Functions
    for name in [
        b"reduce_max_pass1_kernel",
        b"reduce_max_pass2_kernel",
        b"fused_clean_sub_add_kernel",
        b"synthesize_clean_image_kernel",
    ]:
        k = ctypes.c_void_p()
        if _CUDA_LIB.cuModuleGetFunction(ctypes.byref(k), _CUDA_MODULE, name) != 0:
            return False
        _CUDA_KERNELS[name.decode("utf-8")] = k

    return True


def is_cuda_lib_available() -> bool:
    """Returns True if the native CUDA NVRTC driver and module are initialized and ready."""
    return _init_cuda_driver()


# ---------------------------------------------------------------------------
# Native Execution Entrypoint
# ---------------------------------------------------------------------------

def run_hogbom_cuda_native(
    dirty_image: np.ndarray,
    config: Optional[CleanPhysicsConfig] = None,
    psf_generator=None,
    beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    clean_mask: Optional[np.ndarray] = None,
    guard_margin: int = 0,
    device=None,
    verbose: bool = False,
):
    """
    Executes Complex Hogbom CLEAN deconvolution on GPU using native CUDA C++ kernels.
    """
    from ..algorithm import CleanResult

    if not _init_cuda_driver():
        raise NotImplementedError(
            "Native CUDA backend could not be initialized via NVRTC / CUDA Driver API."
        )

    # Explicitly bind the CUDA driver context to current thread
    _CUDA_LIB.cuCtxSetCurrent(_CUDA_CTX)

    t_pipeline_start = time.perf_counter()

    dirty_arr = np.ascontiguousarray(dirty_image, dtype=np.complex64)
    H, W = dirty_arr.shape
    total_elements = H * W
    bytes_complex = total_elements * 8

    # 1. Device Memory Allocations
    d_residual = ctypes.c_void_p()
    d_model = ctypes.c_void_p()
    d_components = ctypes.c_void_p()
    d_clean = ctypes.c_void_p()

    NUM_BLOCKS_PASS1 = 64
    d_block_vals = ctypes.c_void_p()
    d_block_idxs = ctypes.c_void_p()
    d_global_val = ctypes.c_void_p()
    d_global_idx = ctypes.c_void_p()

    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_residual), bytes_complex)
    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_model), bytes_complex)
    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_components), bytes_complex)
    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_clean), bytes_complex)

    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_block_vals), NUM_BLOCKS_PASS1 * 4)
    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_block_idxs), NUM_BLOCKS_PASS1 * 4)
    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_global_val), 4)
    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_global_idx), 4)

    _CUDA_LIB.cuMemsetD8_v2(d_model, 0, bytes_complex)
    _CUDA_LIB.cuMemsetD8_v2(d_components, 0, bytes_complex)

    # 2. Host to Device Copy (H2D)
    t0_h2d = time.perf_counter()
    _CUDA_LIB.cuMemcpyHtoD_v2(d_residual, dirty_arr.ctypes.data_as(ctypes.c_void_p), bytes_complex)
    t_h2d = time.perf_counter() - t0_h2d

    # 3. Setup Kernel Arguments
    _WINDOW_MAP = {"UNIFORM": 0, "RECT": 0, "TAYLOR": 1, "HAMMING": 2, "HANN": 3}
    row_wgt_code = _WINDOW_MAP.get(str(config.row_wgt).upper(), 0) if config else 0
    col_wgt_code = _WINDOW_MAP.get(str(config.col_wgt).upper(), 0) if config else 0

    fwhm_const = 2.0 * math.sqrt(math.log(2.0))
    sigma_r = (config.row_wid / fwhm_const) if config else 0.5
    sigma_c = (config.col_wid / fwhm_const) if config else 0.5

    c_total_elements = ctypes.c_int(total_elements)
    c_height = ctypes.c_int(H)
    c_width = ctypes.c_int(W)
    c_guard = ctypes.c_int(guard_margin)
    c_num_blocks1 = ctypes.c_int(NUM_BLOCKS_PASS1)

    args_red1 = (ctypes.c_void_p * 7)(
        ctypes.cast(ctypes.byref(d_residual), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_block_vals), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_block_idxs), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_total_elements), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_height), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_width), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_guard), ctypes.c_void_p),
    )

    args_red2 = (ctypes.c_void_p * 5)(
        ctypes.cast(ctypes.byref(d_block_vals), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_block_idxs), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_global_val), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_global_idx), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_num_blocks1), ctypes.c_void_p),
    )

    h_max_val = ctypes.c_float()
    h_max_idx = ctypes.c_int()

    k_reduce1 = _CUDA_KERNELS["reduce_max_pass1_kernel"]
    k_reduce2 = _CUDA_KERNELS["reduce_max_pass2_kernel"]
    k_fused = _CUDA_KERNELS["fused_clean_sub_add_kernel"]
    k_synth = _CUDA_KERNELS["synthesize_clean_image_kernel"]

    # 4. Pure GPU Deconvolution Loop
    t0_compute = time.perf_counter()

    # Find initial peak
    _CUDA_LIB.cuLaunchKernel(k_reduce1, NUM_BLOCKS_PASS1, 1, 1, 256, 1, 1, 0, None, args_red1, None)
    _CUDA_LIB.cuLaunchKernel(k_reduce2, 1, 1, 1, 256, 1, 1, 0, None, args_red2, None)
    _CUDA_LIB.cuMemcpyDtoH_v2(ctypes.byref(h_max_val), d_global_val, 4)
    _CUDA_LIB.cuMemcpyDtoH_v2(ctypes.byref(h_max_idx), d_global_idx, 4)

    init_peak = float(h_max_val.value)
    stop_thresh = (threshold * init_peak) if threshold < 1.0 else threshold

    if verbose:
        origin_str = f"origin ({config.chip_start_row}, {config.chip_start_col})" if config else ""
        print(f"[CLEAN] Initial peak: {init_peak:.4e}, Stopping threshold: {stop_thresh:.4e}, Max iters: {max_iters} ({origin_str})")
        print(f"        Backend: CUDA (Native C++/NVRTC) | Beam: {beam_type.upper()} | PSF size: {psf_size}x{psf_size}")

    h_peak_complex = (ctypes.c_float * 2)()
    history_peaks: List[float] = []
    history_coords: List[Tuple[int, int]] = []

    block_dim = 16
    grid_dim = (psf_size + block_dim - 1) // block_dim

    c_r0 = ctypes.c_int(0)
    c_c0 = ctypes.c_int(0)
    c_comp_re = ctypes.c_float(0.0)
    c_comp_im = ctypes.c_float(0.0)
    c_cos_t = ctypes.c_float(1.0)
    c_sin_t = ctypes.c_float(0.0)
    c_sigma_r = ctypes.c_float(sigma_r)
    c_sigma_c = ctypes.c_float(sigma_c)
    c_row_ss = ctypes.c_float(config.row_ss if config else 0.5)
    c_col_ss = ctypes.c_float(config.col_ss if config else 0.5)
    c_row_bw = ctypes.c_float(config.row_bw if config else 1.5)
    c_col_bw = ctypes.c_float(config.col_bw if config else 1.5)
    c_row_wgt = ctypes.c_int(row_wgt_code)
    c_col_wgt = ctypes.c_int(col_wgt_code)
    c_psf_size = ctypes.c_int(psf_size)

    args_fused = (ctypes.c_void_p * 20)(
        ctypes.cast(ctypes.byref(d_residual), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_model), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_components), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_height), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_width), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_r0), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_c0), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_comp_re), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_comp_im), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_cos_t), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_sin_t), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_sigma_r), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_sigma_c), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_row_ss), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_col_ss), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_row_bw), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_col_bw), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_row_wgt), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_col_wgt), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_psf_size), ctypes.c_void_p),
    )

    it = 0
    while it < max_iters:
        curr_peak = float(h_max_val.value)
        flat_idx = int(h_max_idx.value)
        history_peaks.append(curr_peak)

        if curr_peak <= stop_thresh or curr_peak == 0.0:
            if verbose:
                print(f"[CLEAN] Converged at iteration {it}: peak {curr_peak:.4e} <= {stop_thresh:.4e}")
            break

        r0 = flat_idx // W
        c0 = flat_idx % W
        history_coords.append((r0, c0))

        # Read complex amplitude at peak
        src_ptr = ctypes.c_void_p(d_residual.value + flat_idx * 8)
        _CUDA_LIB.cuMemcpyDtoH_v2(ctypes.byref(h_peak_complex), src_ptr, 8)
        amp_re = h_peak_complex[0]
        amp_im = h_peak_complex[1]

        comp_re = gain * amp_re
        comp_im = gain * amp_im

        if config:
            rg, cg = config.chip_to_global(r0, c0)
            xr, yc = config.global_to_metric(rg, cg)
            theta = math.atan2(yc, config.scp_slant_range + xr)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
        else:
            cos_t = 1.0
            sin_t = 0.0

        c_r0.value = r0
        c_c0.value = c0
        c_comp_re.value = comp_re
        c_comp_im.value = comp_im
        c_cos_t.value = cos_t
        c_sin_t.value = sin_t

        _CUDA_LIB.cuLaunchKernel(
            k_fused,
            grid_dim, grid_dim, 1,
            block_dim, block_dim, 1,
            0, None,
            args_fused, None
        )

        # Re-run reduction for next iteration
        _CUDA_LIB.cuLaunchKernel(k_reduce1, NUM_BLOCKS_PASS1, 1, 1, 256, 1, 1, 0, None, args_red1, None)
        _CUDA_LIB.cuLaunchKernel(k_reduce2, 1, 1, 1, 256, 1, 1, 0, None, args_red2, None)
        _CUDA_LIB.cuMemcpyDtoH_v2(ctypes.byref(h_max_val), d_global_val, 4)
        _CUDA_LIB.cuMemcpyDtoH_v2(ctypes.byref(h_max_idx), d_global_idx, 4)

        it += 1
        if verbose and it % 250 == 0:
            print(f"  Iter {it:5d}: Current max residual = {curr_peak:.4e} (Peak at [{r0}, {c0}])")

    # Synthesize clean image: Clean = Model + Residual
    grid_synth = (total_elements + 255) // 256
    args_synth = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(d_residual), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_model), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_clean), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(c_total_elements), ctypes.c_void_p),
    )
    _CUDA_LIB.cuLaunchKernel(k_synth, grid_synth, 1, 1, 256, 1, 1, 0, None, args_synth, None)
    _CUDA_LIB.cuCtxSynchronize()
    t_compute = time.perf_counter() - t0_compute

    # 5. Device to Host Copy (D2H)
    t0_d2h = time.perf_counter()
    clean_out = np.empty((H, W), dtype=np.complex64)
    residual_out = np.empty((H, W), dtype=np.complex64)
    model_out = np.empty((H, W), dtype=np.complex64)
    comp_out = np.empty((H, W), dtype=np.complex64)

    _CUDA_LIB.cuMemcpyDtoH_v2(clean_out.ctypes.data_as(ctypes.c_void_p), d_clean, bytes_complex)
    _CUDA_LIB.cuMemcpyDtoH_v2(residual_out.ctypes.data_as(ctypes.c_void_p), d_residual, bytes_complex)
    _CUDA_LIB.cuMemcpyDtoH_v2(model_out.ctypes.data_as(ctypes.c_void_p), d_model, bytes_complex)
    _CUDA_LIB.cuMemcpyDtoH_v2(comp_out.ctypes.data_as(ctypes.c_void_p), d_components, bytes_complex)
    t_d2h = time.perf_counter() - t0_d2h

    # 6. Cleanup GPU allocations
    _CUDA_LIB.cuMemFree_v2(d_residual)
    _CUDA_LIB.cuMemFree_v2(d_model)
    _CUDA_LIB.cuMemFree_v2(d_components)
    _CUDA_LIB.cuMemFree_v2(d_clean)
    _CUDA_LIB.cuMemFree_v2(d_block_vals)
    _CUDA_LIB.cuMemFree_v2(d_block_idxs)
    _CUDA_LIB.cuMemFree_v2(d_global_val)
    _CUDA_LIB.cuMemFree_v2(d_global_idx)

    t_total = time.perf_counter() - t_pipeline_start

    res = CleanResult(
        clean_image=clean_out,
        components_map=comp_out,
        residual_image=residual_out,
        restored_model=model_out,
        iterations=it,
        execution_time_sec=t_total,
        history_peaks=history_peaks,
        history_coords=history_coords,
    )
    res.h2d_time_sec = t_h2d
    res.pure_compute_time_sec = t_compute
    res.d2h_time_sec = t_d2h
    return res
