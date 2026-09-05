"""
cuda_backend.py: Native CUDA implementation of Complex Hogbom CLEAN deconvolution
compiled directly to PTX via NVIDIA Runtime Compiler (NVRTC) and launched via CUDA Driver API.
"""

import os
import sys
import math
import time
import ctypes
from typing import Optional, Union, Tuple, List, Literal
import numpy as np

from ..config import CleanPhysicsConfig
from .cuda_check import CudaDriver, Nvrtc, CudaError, NvrtcError


# ---------------------------------------------------------------------------
# NVRTC & CUDA Driver Singleton Manager
# ---------------------------------------------------------------------------

_CUDA_DRIVER: Optional[CudaDriver] = None
_CUDA_CTX: Optional[ctypes.c_void_p] = None
_CUDA_MODULE: Optional[ctypes.c_void_p] = None
_CUDA_KERNELS: dict = {}

NVRTC_CANDIDATE_PATHS = [
    "/home/feildaw/mypyenv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
    "libnvrtc.so.12",
    "libnvrtc.so",
]


def _init_cuda_driver() -> bool:
    global _CUDA_DRIVER, _CUDA_CTX, _CUDA_MODULE, _CUDA_KERNELS

    if _CUDA_MODULE is not None and _CUDA_DRIVER is not None and _CUDA_CTX is not None:
        _CUDA_DRIVER.call("cuCtxSetCurrent", _CUDA_CTX)
        return True

    # 1. Load CUDA Driver
    try:
        driver = CudaDriver().init()
    except Exception:
        return False

    dev = driver.device(0)

    # 2. Retain Primary Context (cooperates with PyTorch and survives context switches)
    try:
        ctx = driver.primary_context(dev)
    except Exception:
        return False

    # 3. Detect GPU architecture dynamically (C09)
    try:
        major, minor = driver.compute_capability(dev)
        arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode()
    except Exception:
        arch_flag = b"--gpu-architecture=compute_86"

    # 4. Load NVRTC and compile CUDA source
    cu_file = os.path.join(os.path.dirname(__file__), "c_src", "clean_hogbom.cu")
    if not os.path.isfile(cu_file):
        return False

    with open(cu_file, "rb") as f:
        cuda_src = f.read()

    try:
        nvrtc = Nvrtc(NVRTC_CANDIDATE_PATHS)
        ptx = nvrtc.compile_to_ptx(
            cuda_src,
            b"clean_hogbom.cu",
            [b"--std=c++14", arch_flag]
        )
    except NvrtcError as err:
        print(f"[ERROR] NVRTC CUDA compilation failed:\n{err}", file=sys.stderr)
        return False
    except Exception:
        return False

    # 5. Load PTX Module
    module = ctypes.c_void_p()
    try:
        driver.call("cuModuleLoadData", ctypes.byref(module), ptx)
    except Exception:
        return False

    # 6. Extract Kernel Functions
    kernels = {}
    kernel_names = [
        b"reduce_max_pass1_kernel",
        b"reduce_max_pass2_kernel",
        b"fused_clean_sub_add_kernel",
        b"synthesize_clean_image_kernel",
    ]
    for name in kernel_names:
        k = ctypes.c_void_p()
        try:
            driver.call("cuModuleGetFunction", ctypes.byref(k), module, name)
            kernels[name.decode("utf-8")] = k
        except Exception:
            return False

    _CUDA_DRIVER = driver
    _CUDA_CTX = ctx
    _CUDA_MODULE = module
    _CUDA_KERNELS = kernels
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

    # Validate inputs: do not silently drop options or fabricate physics
    if config is None:
        raise ValueError("CleanPhysicsConfig 'config' must be provided for the native CUDA backend.")

    if beam_type != "gaussian":
        raise NotImplementedError(
            f"Native CUDA backend currently only supports beam_type='gaussian' (got '{beam_type}'). "
            "Please use backend='pytorch' for 'mainlobe' restoring beam."
        )

    if clean_mask is not None:
        raise NotImplementedError(
            "Native CUDA backend does not currently support 'clean_mask'. "
            "Please use backend='pytorch' for masked CLEAN deconvolution."
        )

    if psf_generator is not None:
        raise NotImplementedError(
            "Native CUDA backend evaluates the analytic PSF directly in-kernel and does not accept "
            "an external 'psf_generator'. Please pass 'config' (CleanPhysicsConfig), or use backend='pytorch'."
        )

    if not _init_cuda_driver():
        raise NotImplementedError(
            "Native CUDA backend could not be initialized via NVRTC / CUDA Driver API."
        )

    # Explicitly bind the primary CUDA driver context
    _CUDA_DRIVER.call("cuCtxSetCurrent", _CUDA_CTX)

    t_pipeline_start = time.perf_counter()

    dirty_arr = np.ascontiguousarray(dirty_image, dtype=np.complex64)
    H, W = dirty_arr.shape
    total_elements = H * W
    bytes_complex = total_elements * 8

    NUM_BLOCKS_PASS1 = 64

    # Device pointers
    d_residual = None
    d_model = None
    d_components = None
    d_clean = None
    d_block_vals = None
    d_block_idxs = None
    d_global_val = None
    d_global_idx = None

    try:
        # 1. Device Memory Allocations (with size_t widths)
        d_residual = _CUDA_DRIVER.mem_alloc(bytes_complex)
        d_model = _CUDA_DRIVER.mem_alloc(bytes_complex)
        d_components = _CUDA_DRIVER.mem_alloc(bytes_complex)
        d_clean = _CUDA_DRIVER.mem_alloc(bytes_complex)

        d_block_vals = _CUDA_DRIVER.mem_alloc(NUM_BLOCKS_PASS1 * 4)
        d_block_idxs = _CUDA_DRIVER.mem_alloc(NUM_BLOCKS_PASS1 * 4)
        d_global_val = _CUDA_DRIVER.mem_alloc(4)
        d_global_idx = _CUDA_DRIVER.mem_alloc(4)

        _CUDA_DRIVER.memset_d8(d_model, 0, bytes_complex)
        _CUDA_DRIVER.memset_d8(d_components, 0, bytes_complex)

        # 2. Host to Device Copy (H2D)
        t0_h2d = time.perf_counter()
        _CUDA_DRIVER.memcpy_htod(d_residual, dirty_arr.ctypes.data_as(ctypes.c_void_p), bytes_complex)
        t_h2d = time.perf_counter() - t0_h2d

        # 3. Setup Kernel Arguments
        _WINDOW_MAP = {"UNIFORM": 0, "RECT": 0, "TAYLOR": 1, "HAMMING": 2, "HANN": 3}
        row_wgt_code = _WINDOW_MAP.get(str(config.row_wgt).upper(), 0)
        col_wgt_code = _WINDOW_MAP.get(str(config.col_wgt).upper(), 0)

        fwhm_const = 2.0 * math.sqrt(math.log(2.0))
        sigma_r = config.row_wid / fwhm_const
        sigma_c = config.col_wid / fwhm_const

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
        _CUDA_DRIVER.launch(k_reduce1, (NUM_BLOCKS_PASS1, 1, 1), (256, 1, 1), args_red1)
        _CUDA_DRIVER.launch(k_reduce2, (1, 1, 1), (256, 1, 1), args_red2)
        _CUDA_DRIVER.memcpy_dtoh(ctypes.byref(h_max_val), d_global_val, 4)
        _CUDA_DRIVER.memcpy_dtoh(ctypes.byref(h_max_idx), d_global_idx, 4)

        init_peak = float(h_max_val.value)
        stop_thresh = (threshold * init_peak) if threshold < 1.0 else threshold

        if verbose:
            origin_str = f"origin ({config.chip_start_row}, {config.chip_start_col})"
            print(f"[CLEAN] Initial peak: {init_peak:.4e}, Stopping threshold: {stop_thresh:.4e}, Max iters: {max_iters} ({origin_str})")
            print(f"        Backend: CUDA (Native C++/NVRTC) | Beam: GAUSSIAN | PSF size: {psf_size}x{psf_size}")

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
        c_row_ss = ctypes.c_float(config.row_ss)
        c_col_ss = ctypes.c_float(config.col_ss)
        c_row_bw = ctypes.c_float(config.row_bw)
        c_col_bw = ctypes.c_float(config.col_bw)
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
            _CUDA_DRIVER.memcpy_dtoh(ctypes.byref(h_peak_complex), src_ptr, 8)
            amp_re = h_peak_complex[0]
            amp_im = h_peak_complex[1]

            comp_re = gain * amp_re
            comp_im = gain * amp_im

            rg, cg = config.chip_to_global(r0, c0)
            xr, yc = config.global_to_metric(rg, cg)
            theta = math.atan2(yc, config.scp_slant_range + xr)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            c_r0.value = r0
            c_c0.value = c0
            c_comp_re.value = comp_re
            c_comp_im.value = comp_im
            c_cos_t.value = cos_t
            c_sin_t.value = sin_t

            _CUDA_DRIVER.launch(
                k_fused,
                (grid_dim, grid_dim, 1),
                (block_dim, block_dim, 1),
                args_fused
            )

            # Re-run reduction for next iteration
            _CUDA_DRIVER.launch(k_reduce1, (NUM_BLOCKS_PASS1, 1, 1), (256, 1, 1), args_red1)
            _CUDA_DRIVER.launch(k_reduce2, (1, 1, 1), (256, 1, 1), args_red2)
            _CUDA_DRIVER.memcpy_dtoh(ctypes.byref(h_max_val), d_global_val, 4)
            _CUDA_DRIVER.memcpy_dtoh(ctypes.byref(h_max_idx), d_global_idx, 4)

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
        _CUDA_DRIVER.launch(k_synth, (grid_synth, 1, 1), (256, 1, 1), args_synth)
        _CUDA_DRIVER.synchronize()
        t_compute = time.perf_counter() - t0_compute

        # 5. Device to Host Copy (D2H) with safe zero-initialized arrays
        t0_d2h = time.perf_counter()
        clean_out = np.zeros((H, W), dtype=np.complex64)
        residual_out = np.zeros((H, W), dtype=np.complex64)
        model_out = np.zeros((H, W), dtype=np.complex64)
        comp_out = np.zeros((H, W), dtype=np.complex64)

        _CUDA_DRIVER.memcpy_dtoh(clean_out.ctypes.data_as(ctypes.c_void_p), d_clean, bytes_complex)
        _CUDA_DRIVER.memcpy_dtoh(residual_out.ctypes.data_as(ctypes.c_void_p), d_residual, bytes_complex)
        _CUDA_DRIVER.memcpy_dtoh(model_out.ctypes.data_as(ctypes.c_void_p), d_model, bytes_complex)
        _CUDA_DRIVER.memcpy_dtoh(comp_out.ctypes.data_as(ctypes.c_void_p), d_components, bytes_complex)
        t_d2h = time.perf_counter() - t0_d2h

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

    finally:
        # 6. Exception-safe GPU memory cleanup (C13)
        if _CUDA_DRIVER is not None:
            for d_ptr in (d_residual, d_model, d_components, d_clean,
                          d_block_vals, d_block_idxs, d_global_val, d_global_idx):
                if d_ptr is not None:
                    _CUDA_DRIVER.mem_free(d_ptr)
