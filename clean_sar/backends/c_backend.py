"""
clean_sar/backends/c_backend.py: C CPU fallback implementation of Complex Hogbom CLEAN
deconvolution mirroring the CUDA implementation.

Loads precompiled libclean_c.so via ctypes, compiles on-the-fly if a C compiler
(gcc/clang) is available, or provides a vectorized NumPy fallback if no C compiler is present.
"""

import os
import math
import time
import ctypes
import subprocess
from typing import Optional, Tuple, List
import numpy as np

from ..config import CleanPhysicsConfig


# ---------------------------------------------------------------------------
# C Struct & Library Loader
# ---------------------------------------------------------------------------

class CPhysicsConfig(ctypes.Structure):
    _fields_ = [
        ("row_ss", ctypes.c_float),
        ("col_ss", ctypes.c_float),
        ("row_bw", ctypes.c_float),
        ("col_bw", ctypes.c_float),
        ("row_wid", ctypes.c_float),
        ("col_wid", ctypes.c_float),
        ("scp_slant_range", ctypes.c_float),
        ("scp_row", ctypes.c_float),
        ("scp_col", ctypes.c_float),
        ("chip_start_row", ctypes.c_int),
        ("chip_start_col", ctypes.c_int),
        ("row_wgt_type", ctypes.c_int),
        ("col_wgt_type", ctypes.c_int),
    ]


_C_LIB: Optional[ctypes.CDLL] = None
_C_RUN_FN = None


def _get_c_so_path() -> str:
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "c_src", "libclean_c.so")


def _compile_c_lib() -> bool:
    """Attempts to compile clean_hogbom.c into libclean_c.so if a C compiler is available."""
    base_dir = os.path.dirname(__file__)
    c_src = os.path.join(base_dir, "c_src", "clean_hogbom.c")
    c_so = os.path.join(base_dir, "c_src", "libclean_c.so")

    if not os.path.isfile(c_src):
        return False

    for compiler in ["gcc", "clang", "cc"]:
        try:
            cmd = [compiler, "-O3", "-fPIC", "-shared", c_src, "-o", c_so, "-lm"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.isfile(c_so):
                return True
        except Exception:
            continue
    return False


def _init_c_lib() -> bool:
    global _C_LIB, _C_RUN_FN
    if _C_LIB is not None and _C_RUN_FN is not None:
        return True

    c_so = _get_c_so_path()
    if not os.path.isfile(c_so):
        _compile_c_lib()

    if os.path.isfile(c_so):
        try:
            lib = ctypes.CDLL(c_so)
            fn = getattr(lib, "run_hogbom_c", None)
            if fn is not None:
                fn.argtypes = [
                    ctypes.c_void_p,                 # dirty_image
                    ctypes.c_int,                    # height
                    ctypes.c_int,                    # width
                    ctypes.POINTER(CPhysicsConfig),  # config
                    ctypes.c_int,                    # psf_size
                    ctypes.c_float,                  # gain
                    ctypes.c_float,                  # threshold
                    ctypes.c_int,                    # max_iters
                    ctypes.c_int,                    # guard_margin
                    ctypes.c_void_p,                 # out_clean
                    ctypes.c_void_p,                 # out_residual
                    ctypes.c_void_p,                 # out_model
                    ctypes.c_void_p,                 # out_components
                    ctypes.POINTER(ctypes.c_int),    # out_iters
                    ctypes.POINTER(ctypes.c_float),  # history_peaks
                    ctypes.POINTER(ctypes.c_int),    # history_coords_r
                    ctypes.POINTER(ctypes.c_int),    # history_coords_c
                ]
                fn.restype = ctypes.c_int
                _C_LIB = lib
                _C_RUN_FN = fn
                return True
        except Exception:
            pass
    return False


def is_c_lib_available() -> bool:
    """Returns True if the compiled C library is ready for ctypes execution."""
    return _init_c_lib()


def get_c_compilation_mode() -> str:
    """Returns 'c_lib' if native libclean_c.so is loaded, else 'numpy_fallback'."""
    return "c_lib" if _init_c_lib() else "numpy_fallback"


# ---------------------------------------------------------------------------
# NumPy Equivalent Engine (used if C compiler / shared lib not yet installed)
# ---------------------------------------------------------------------------

def _sinc_1d(x: np.ndarray) -> np.ndarray:
    return np.sinc(x).astype(np.float32)


def _eval_1d_window_sinc(pos: np.ndarray, bw: float, wgt_type: int) -> np.ndarray:
    x = bw * pos
    if wgt_type == 0:  # UNIFORM
        return _sinc_1d(x)
    elif wgt_type == 1:  # TAYLOR
        f1, f2, f3 = 0.29265601, -0.01578375, 0.00218104
        return (
            _sinc_1d(x)
            + f1 * (_sinc_1d(x - 1.0) + _sinc_1d(x + 1.0))
            + f2 * (_sinc_1d(x - 2.0) + _sinc_1d(x + 2.0))
            + f3 * (_sinc_1d(x - 3.0) + _sinc_1d(x + 3.0))
        )
    elif wgt_type == 2:  # HAMMING
        return (0.54 * _sinc_1d(x) + 0.23 * (_sinc_1d(x - 1.0) + _sinc_1d(x + 1.0))) * (1.0 / 0.54)
    elif wgt_type == 3:  # HANN
        return (0.50 * _sinc_1d(x) + 0.25 * (_sinc_1d(x - 1.0) + _sinc_1d(x + 1.0))) * 2.0
    return _sinc_1d(x)


def _run_hogbom_c_numpy_fallback(
    dirty_image: np.ndarray,
    config: CleanPhysicsConfig,
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    guard_margin: int = 0,
    verbose: bool = False,
):
    from ..algorithm import CleanResult

    t_start = time.perf_counter()
    residual = np.array(dirty_image, dtype=np.complex64, copy=True)
    H, W = residual.shape
    model = np.zeros((H, W), dtype=np.complex64)
    components = np.zeros((H, W), dtype=np.complex64)

    _WINDOW_MAP = {"UNIFORM": 0, "RECT": 0, "TAYLOR": 1, "HAMMING": 2, "HANN": 3}
    row_wgt_code = _WINDOW_MAP.get(str(config.row_wgt).upper(), 0)
    col_wgt_code = _WINDOW_MAP.get(str(config.col_wgt).upper(), 0)

    fwhm_const = 2.0 * math.sqrt(math.log(2.0))
    sigma_r = config.row_wid / fwhm_const
    sigma_c = config.col_wid / fwhm_const

    if psf_size % 2 == 0:
        psf_size += 1
    kh = psf_size // 2
    kw = psf_size // 2

    dr_grid = np.arange(-kh, kh + 1, dtype=np.float32)
    dc_grid = np.arange(-kw, kw + 1, dtype=np.float32)
    dr_mesh, dc_mesh = np.meshgrid(dr_grid, dc_grid, indexing="ij")
    u_base = dr_mesh * float(config.row_ss)
    v_base = dc_mesh * float(config.col_ss)

    mag = np.abs(residual)
    if guard_margin > 0:
        search_mag = mag[guard_margin:H - guard_margin, guard_margin:W - guard_margin]
        init_peak = float(np.max(search_mag)) if search_mag.size > 0 else 0.0
    else:
        init_peak = float(np.max(mag))

    stop_thresh = (threshold * init_peak) if threshold < 1.0 else threshold

    if verbose:
        print(f"[CLEAN] Initial peak: {init_peak:.4e}, Stopping threshold: {stop_thresh:.4e}, Max iters: {max_iters}")
        print(f"        Backend: C (NumPy fallback engine) | PSF size: {psf_size}x{psf_size}")

    history_peaks: List[float] = []
    history_coords: List[Tuple[int, int]] = []

    it = 0
    while it < max_iters:
        mag = np.abs(residual)
        if guard_margin > 0:
            search_mag = mag.copy()
            search_mag[:guard_margin, :] = -1.0
            search_mag[H - guard_margin:, :] = -1.0
            search_mag[:, :guard_margin] = -1.0
            search_mag[:, W - guard_margin:] = -1.0
            flat_idx = int(np.argmax(search_mag))
            curr_peak = float(search_mag.ravel()[flat_idx])
        else:
            flat_idx = int(np.argmax(mag))
            curr_peak = float(mag.ravel()[flat_idx])

        history_peaks.append(curr_peak)
        if curr_peak <= stop_thresh or curr_peak <= 0.0:
            if verbose:
                print(f"[CLEAN] Converged at iteration {it}: peak {curr_peak:.4e} <= {stop_thresh:.4e}")
            break

        r0 = flat_idx // W
        c0 = flat_idx % W
        history_coords.append((r0, c0))

        amp = residual[r0, c0]
        comp = gain * amp
        components[r0, c0] += comp

        rg = float(r0 + config.chip_start_row)
        cg = float(c0 + config.chip_start_col)
        xr = (rg - config.scp_row) * config.row_ss
        yc = (cg - config.scp_col) * config.col_ss
        theta = math.atan2(yc, config.scp_slant_range + xr)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        u_prime =  u_base * cos_t + v_base * sin_t
        v_prime = -u_base * sin_t + v_base * cos_t

        psf_r = _eval_1d_window_sinc(u_prime, config.row_bw, row_wgt_code)
        psf_c = _eval_1d_window_sinc(v_prime, config.col_bw, col_wgt_code)
        h_dirty = (psf_r * psf_c).astype(np.complex64)

        r_term = u_prime / max(sigma_r, 1e-6)
        c_term = v_prime / max(sigma_c, 1e-6)
        h_clean = np.exp(-0.5 * (r_term * r_term + c_term * c_term)).astype(np.complex64)

        r_min = max(0, r0 - kh)
        r_max = min(H, r0 + kh + 1)
        c_min = max(0, c0 - kw)
        c_max = min(W, c0 + kw + 1)

        pr_min = kh - (r0 - r_min)
        pr_max = kh + (r_max - r0)
        pc_min = kw - (c0 - c_min)
        pc_max = kw + (c_max - c0)

        residual[r_min:r_max, c_min:c_max] -= comp * h_dirty[pr_min:pr_max, pc_min:pc_max]
        model[r_min:r_max, c_min:c_max] += comp * h_clean[pr_min:pr_max, pc_min:pc_max]

        it += 1
        if verbose and it % 250 == 0:
            print(f"  Iter {it:5d}: Current max residual = {curr_peak:.4e} (Peak at [{r0}, {c0}])")

    clean_image = model + residual
    t_total = time.perf_counter() - t_start

    res = CleanResult(
        clean_image=clean_image,
        components_map=components,
        residual_image=residual,
        restored_model=model,
        iterations=it,
        execution_time_sec=t_total,
        history_peaks=history_peaks,
        history_coords=history_coords,
    )
    res.pure_compute_time_sec = t_total
    return res


# ---------------------------------------------------------------------------
# C Backend Entrypoint
# ---------------------------------------------------------------------------

def run_hogbom_c_native(
    dirty_image: np.ndarray,
    config: Optional[CleanPhysicsConfig] = None,
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    guard_margin: int = 0,
    verbose: bool = False,
):
    """
    Executes Complex Hogbom CLEAN deconvolution on CPU using the native C implementation
    (or transparent NumPy fallback if the C shared library is not yet compiled).
    """
    from ..algorithm import CleanResult

    if config is None:
        raise ValueError("CleanPhysicsConfig 'config' must be provided for the C backend.")

    dirty_arr = np.ascontiguousarray(dirty_image, dtype=np.complex64)
    H, W = dirty_arr.shape

    # If compiled C library is not available, run fallback engine
    if not _init_c_lib():
        return _run_hogbom_c_numpy_fallback(
            dirty_image=dirty_arr,
            config=config,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            guard_margin=guard_margin,
            verbose=verbose,
        )

    t_start = time.perf_counter()

    if verbose:
        mag = np.abs(dirty_arr)
        init_peak = float(np.max(mag))
        stop_thresh = (threshold * init_peak) if threshold < 1.0 else threshold
        print(f"[CLEAN] Initial peak: {init_peak:.4e}, Stopping threshold: {stop_thresh:.4e}, Max iters: {max_iters}")
        print(f"        Backend: C (Native libclean_c.so via gcc/clang) | PSF size: {psf_size}x{psf_size}")

    _WINDOW_MAP = {"UNIFORM": 0, "RECT": 0, "TAYLOR": 1, "HAMMING": 2, "HANN": 3}
    row_wgt_code = _WINDOW_MAP.get(str(config.row_wgt).upper(), 0)
    col_wgt_code = _WINDOW_MAP.get(str(config.col_wgt).upper(), 0)

    c_config = CPhysicsConfig(
        row_ss=ctypes.c_float(config.row_ss),
        col_ss=ctypes.c_float(config.col_ss),
        row_bw=ctypes.c_float(config.row_bw),
        col_bw=ctypes.c_float(config.col_bw),
        row_wid=ctypes.c_float(config.row_wid),
        col_wid=ctypes.c_float(config.col_wid),
        scp_slant_range=ctypes.c_float(config.scp_slant_range),
        scp_row=ctypes.c_float(config.scp_row),
        scp_col=ctypes.c_float(config.scp_col),
        chip_start_row=ctypes.c_int(config.chip_start_row),
        chip_start_col=ctypes.c_int(config.chip_start_col),
        row_wgt_type=ctypes.c_int(row_wgt_code),
        col_wgt_type=ctypes.c_int(col_wgt_code),
    )

    clean_out = np.zeros((H, W), dtype=np.complex64)
    residual_out = np.zeros((H, W), dtype=np.complex64)
    model_out = np.zeros((H, W), dtype=np.complex64)
    comp_out = np.zeros((H, W), dtype=np.complex64)

    out_iters = ctypes.c_int(0)
    history_peaks_arr = (ctypes.c_float * max_iters)()
    history_coords_r = (ctypes.c_int * max_iters)()
    history_coords_c = (ctypes.c_int * max_iters)()

    ret = _C_RUN_FN(
        dirty_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(H),
        ctypes.c_int(W),
        ctypes.byref(c_config),
        ctypes.c_int(psf_size),
        ctypes.c_float(gain),
        ctypes.c_float(threshold),
        ctypes.c_int(max_iters),
        ctypes.c_int(guard_margin),
        clean_out.ctypes.data_as(ctypes.c_void_p),
        residual_out.ctypes.data_as(ctypes.c_void_p),
        model_out.ctypes.data_as(ctypes.c_void_p),
        comp_out.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(out_iters),
        history_peaks_arr,
        history_coords_r,
        history_coords_c,
    )

    if ret != 0:
        raise RuntimeError(f"run_hogbom_c failed with error code {ret}")

    n_iters = out_iters.value
    t_total = time.perf_counter() - t_start

    history_peaks = [float(history_peaks_arr[i]) for i in range(n_iters)]
    history_coords = [(int(history_coords_r[i]), int(history_coords_c[i])) for i in range(n_iters)]

    res = CleanResult(
        clean_image=clean_out,
        components_map=comp_out,
        residual_image=residual_out,
        restored_model=model_out,
        iterations=n_iters,
        execution_time_sec=t_total,
        history_peaks=history_peaks,
        history_coords=history_coords,
    )
    res.pure_compute_time_sec = t_total
    return res
