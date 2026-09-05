import os
import time
from typing import Optional, Tuple, Union, Literal
import torch

from .sicd_handler import SICDHandler
from .config import CleanPhysicsConfig
from .algorithm import run_hogbom_clean, CleanResult
from .backends import AvailableBackend


class CLEANProcessor:
    """
    High-level processor for Complex Hogbom CLEAN SAR deconvolution on NITF SICD files.

    Manages SICD metadata loading, spatial bounds / chipping, coordinates,
    scalar physics extraction, backend selection, and compliant NITF writing.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        chip_bounds: Optional[Tuple[int, int, int, int]] = None,
        backend: AvailableBackend = "auto",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Parameters
        ----------
        input_path : str
            Path to input raw/dirty NITF SICD file.
        output_path : str
            Path to output deconvolved NITF SICD file.
        chip_bounds : tuple of (start_row, start_col, stop_row, stop_col), optional
            Pixel bounding box for sub-image deconvolution. If None, full scene is processed.
        backend : 'auto', 'pytorch', or 'cuda'
            Compute backend selection (default: 'auto').
        device : str or torch.device, optional
            Compute device ('cuda' or 'cpu'). If None, GPU is used if available.
        """
        self.input_path = os.path.abspath(input_path)
        self.output_path = os.path.abspath(output_path)
        self.chip_bounds = chip_bounds
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize SICD handler
        self.handler = SICDHandler(self.input_path)

    def run(
        self,
        gain: float = 0.1,
        threshold: float = 0.02,
        max_iters: int = 2500,
        backend: Optional[AvailableBackend] = None,
        beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
        psf_size: int = 65,
        guard_margin: int = 0,
        verbose: bool = True,
    ) -> CleanResult:
        """
        Executes the end-to-end CLEAN deconvolution pipeline and writes the output SICD NITF.

        Parameters
        ----------
        gain : float
            Loop damping factor (gamma), default 0.1.
        threshold : float
            Stopping threshold (fraction of initial peak if < 1.0, or absolute magnitude).
        max_iters : int
            Maximum number of CLEAN iterations.
        backend : 'auto', 'pytorch', or 'cuda', optional
            Override compute backend for this run (defaults to self.backend).
        beam_type : 'gaussian' or 'mainlobe'
            Clean restoring beam shape.
        psf_size : int
            PSF kernel dimension (must be odd, e.g. 65).
        guard_margin : int
            Margin around image edges excluded from peak search (default: 0).
        verbose : bool
            If True, prints progress periodically.

        Returns
        -------
        CleanResult
            Deconvolution result containing clean image, residual, components, and metrics.
        """
        t0 = time.perf_counter()
        active_backend = backend or self.backend

        # 1. Load image data (sub-image chip or full scene)
        if self.chip_bounds is not None:
            r_min, c_min, r_max, c_max = self.chip_bounds
            if verbose:
                print(f"[*] Reading chip region: rows [{r_min}:{r_max}], cols [{c_min}:{c_max}]")
            dirty_image, custom_xmltree = self.handler.read_chip(r_min, c_min, r_max, c_max)
            chip_origin = (r_min, c_min)
        else:
            if verbose:
                print(f"[*] Reading full scene: {self.handler.num_rows} x {self.handler.num_cols}")
            dirty_image = self.handler.read_full_image()
            custom_xmltree = None
            chip_origin = (0, 0)

        # 2. Extract plain scalar physics configuration from SICD handler
        config = CleanPhysicsConfig.from_sicd_handler(self.handler, chip_start=chip_origin)

        # 3. Execute deconvolution algorithm with chosen backend
        result = run_hogbom_clean(
            dirty_image=dirty_image,
            config=config,
            backend=active_backend,
            beam_type=beam_type,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            guard_margin=guard_margin,
            device=self.device,
            verbose=verbose,
        )

        # 4. Write output SICD NITF
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        if verbose:
            print(f"[*] Writing deconvolved SICD NITF to: {self.output_path}")
        self.handler.write_nitf(
            output_path=self.output_path,
            complex_image=result.clean_image,
            custom_xmltree=custom_xmltree,
        )

        total_elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[+] CLEAN pipeline complete in {total_elapsed:.2f}s "
                  f"({result.iterations} iterations, {result.suppression_db:.1f} dB suppression)")

        return result
