"""
CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution with Exact Spatially-Varying IPR for NITF SICD.
"""

from .processor import CLEANProcessor
from .config import CleanPhysicsConfig
from .algorithm import run_hogbom_clean, CleanResult
from .psf import PSFGenerator
from .sicd_handler import SICDHandler
from .utils import (
    db_scale,
    taylor_window_1d,
    get_1d_window,
    get_2d_window,
)
from .quality import (
    find_bright_targets,
    ipr_quality,
    ipr_quality_multi,
    verdict,
)

__version__ = "0.2.0"
__all__ = [
    "CLEANProcessor",
    "CleanPhysicsConfig",
    "run_hogbom_clean",
    "CleanResult",
    "PSFGenerator",
    "SICDHandler",
    "db_scale",
    "taylor_window_1d",
    "get_1d_window",
    "get_2d_window",
    "find_bright_targets",
    "ipr_quality",
    "ipr_quality_multi",
    "verdict",
]
