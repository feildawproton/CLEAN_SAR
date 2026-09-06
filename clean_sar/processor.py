import os
import time
from typing import Optional, Literal

from .sicd_handler import SICDHandler
from .config import CleanPhysicsConfig
from .algorithm import run_hogbom_clean, CleanResult
from .backends import AvailableBackend


class CLEANProcessor:
    """
    High-level processor for Complex Hogbom CLEAN SAR deconvolution on NITF SICD files.

    Manages SICD metadata loading, scalar physics extraction,
    backend selection, and compliant NITF writing.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        backend: AvailableBackend = "auto",
    ):
        """
        Parameters
        ----------
        input_path : str
            Path to input raw/dirty NITF SICD file (full scene or caller-prepared chip).
        output_path : str
            Path to output deconvolved NITF SICD file.
        backend : 'auto', 'cuda', or 'c', default 'auto'
            Compute backend selection.
        """
        self.input_path = os.path.abspath(input_path)
        self.output_path = os.path.abspath(output_path)
        self.backend = backend

        # Initialize SICD handler
        self.handler = SICDHandler(self.input_path)

    def run(
        self,
        gain: float = 0.1,
        threshold: float = 0.02,
        max_iters: int = 2500,
        backend: Optional[AvailableBackend] = None,
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
        backend : 'auto', 'cuda', or 'c', optional
            Override compute backend for this run (defaults to self.backend).
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

        # 1. Load image data from SICD file
        if verbose:
            print(f"[*] Reading image: {self.handler.num_rows} x {self.handler.num_cols}")
        dirty_image = self.handler.read_full_image()

        # 2. Extract plain scalar physics configuration from SICD handler
        config = CleanPhysicsConfig.from_sicd_handler(self.handler)

        # 3. Execute deconvolution algorithm with chosen backend
        result = run_hogbom_clean(
            dirty_image=dirty_image,
            config=config,
            backend=active_backend,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            guard_margin=guard_margin,
            verbose=verbose,
        )

        # 4. Write output SICD NITF
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        if verbose:
            print(f"[*] Writing deconvolved SICD NITF to: {self.output_path}")
        self.handler.write_nitf(
            output_path=self.output_path,
            complex_image=result.clean_image,
        )

        total_elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[+] CLEAN pipeline complete in {total_elapsed:.2f}s "
                  f"({result.iterations} iterations, {result.suppression_db:.1f} dB suppression)")

        return result
