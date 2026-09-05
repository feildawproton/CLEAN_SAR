from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Literal
import numpy as np
import torch

from .config import CleanPhysicsConfig
from .psf import PSFGenerator
from .backends import resolve_backend
from .backends.pytorch_backend import run_hogbom_pytorch
from .backends.cuda_backend import run_hogbom_cuda_native


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
            return 20.0 * np.log10(self.initial_peak / self.final_peak)
        return 0.0

    @property
    def suppression_db(self) -> float:
        """Alias for peak_reduction_db for backward compatibility."""
        return self.peak_reduction_db

    @property
    def num_components(self) -> int:
        return int(np.count_nonzero(self.components_map))


def run_hogbom_clean(
    dirty_image: Union[np.ndarray, torch.Tensor],
    config: Optional[CleanPhysicsConfig] = None,
    psf_generator: Optional[PSFGenerator] = None,
    backend: Literal["auto", "pytorch", "cuda"] = "auto",
    beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    clean_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    guard_margin: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> CleanResult:
    """
    Executes Complex Hogbom CLEAN deconvolution, dispatching to the requested compute backend.
    """
    chosen_backend = resolve_backend(backend)

    if chosen_backend == "cuda":
        return run_hogbom_cuda_native(
            dirty_image=dirty_image,
            config=config,
            psf_generator=psf_generator,
            beam_type=beam_type,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            clean_mask=clean_mask,
            guard_margin=guard_margin,
            device=device,
            verbose=verbose,
        )
    else:
        return run_hogbom_pytorch(
            dirty_image=dirty_image,
            config=config,
            psf_generator=psf_generator,
            beam_type=beam_type,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            clean_mask=clean_mask,
            guard_margin=guard_margin,
            device=device,
            verbose=verbose,
        )
