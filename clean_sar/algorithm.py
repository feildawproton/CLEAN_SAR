from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal
import numpy as np

from .config import CleanPhysicsConfig
from .psf import calculate_psf_size
from .backends import resolve_backend
from .backends.cuda_backend import run_hogbom_cuda_native
from .backends.c_backend import run_hogbom_c_native


@dataclass
class CleanResult:
    """Dataclass holding output products and convergence history from Complex Hogbom CLEAN."""
    clean_image: np.ndarray
    components_map: np.ndarray
    residual_image: np.ndarray
    restored_model: np.ndarray
    iterations: int
    execution_time_sec: float
    history_peaks: List[float] = field(default_factory=list)
    history_coords: List[Tuple[int, int]] = field(default_factory=list)
    h2d_time_sec: float = 0.0
    pure_compute_time_sec: float = 0.0
    d2h_time_sec: float = 0.0

    @property
    def initial_peak(self) -> float:
        return self.history_peaks[0] if self.history_peaks else 0.0

    @property
    def final_peak(self) -> float:
        return self.history_peaks[-1] if self.history_peaks else 0.0

    @property
    def peak_reduction_db(self) -> float:
        """Peak-residual reduction: 20*log10(initial_peak / final_peak)."""
        if self.initial_peak > 0 and self.final_peak > 0:
            return 20.0 * float(np.log10(self.initial_peak / self.final_peak))
        return 0.0

    @property
    def suppression_db(self) -> float:
        """Alias for peak_reduction_db for backward compatibility."""
        return self.peak_reduction_db

    @property
    def num_components(self) -> int:
        return int(np.count_nonzero(self.components_map))


def run_hogbom_clean(
    dirty_image: np.ndarray,
    config: Optional[CleanPhysicsConfig] = None,
    backend: Literal["auto", "cuda", "c"] = "auto",
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    guard_margin: int = 0,
    verbose: bool = False,
    **kwargs,
) -> CleanResult:
    """
    Executes Complex Hogbom CLEAN deconvolution on a 2D complex SAR image.

    Parameters
    ----------
    dirty_image : np.ndarray
        2D complex dirty image (numpy complex64 array).
    config : CleanPhysicsConfig
        Radar physics configuration containing geometric and sampling parameters.
    backend : 'auto', 'cuda', or 'c', default 'auto'
        Compute backend. 'auto' selects CUDA if available, otherwise falls back to C.
    gain : float, default 0.1
        Loop damping factor (gamma).
    threshold : float, default 0.02
        Stopping threshold (fraction of initial peak if < 1.0, or absolute magnitude).
    max_iters : int, default 2500
        Maximum number of CLEAN iterations.
    guard_margin : int, default 0
        Pixel margin along image borders excluded from peak selection.
    verbose : bool, default False
        If True, prints periodic convergence diagnostics.

    Returns
    -------
    CleanResult
        Container with deconvolved image, residual, component map, and metrics.
    """
    if config is None:
        raise ValueError("CleanPhysicsConfig 'config' must be provided.")

    clean_mask = kwargs.get("clean_mask", None)
    if clean_mask is not None:
        raise NotImplementedError("CLEAN deconvolution does not currently support 'clean_mask'.")

    # Optimal PSF grid size is automatically calculated from radar physics and bounded by image dimensions
    psf_size = calculate_psf_size(config, image_shape=dirty_image.shape)

    chosen_backend = resolve_backend(backend)

    if chosen_backend == "cuda":
        return run_hogbom_cuda_native(
            dirty_image=dirty_image,
            config=config,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            guard_margin=guard_margin,
            verbose=verbose,
        )
    else:
        return run_hogbom_c_native(
            dirty_image=dirty_image,
            config=config,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            guard_margin=guard_margin,
            verbose=verbose,
        )
