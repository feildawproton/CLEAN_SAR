"""
CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution with Exact Spatially-Varying IPR for NITF SICD.
"""

from .sicd_handler import SICDHandler
from .psf import PSFGenerator
from .engine import run_hogbom_clean, CleanResult
from .utils import (
    db_scale,
    taylor_window_1d,
    get_1d_window,
    get_2d_window,
    plot_clean_comparison,
)

__version__ = "0.2.0"
__all__ = [
    "SICDHandler",
    "PSFGenerator",
    "run_hogbom_clean",
    "CleanResult",
    "db_scale",
    "taylor_window_1d",
    "get_1d_window",
    "get_2d_window",
    "plot_clean_comparison",
]
