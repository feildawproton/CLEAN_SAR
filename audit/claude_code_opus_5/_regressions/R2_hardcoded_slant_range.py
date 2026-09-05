
import numpy as np
from clean_sar.psf import PSFGenerator
from clean_sar.config import CleanPhysicsConfig

def _grid(gen, row, col, psf_size):
    _, _, xrow, ycol = gen._get_metric_coords(row, col)
    theta = np.arctan2(ycol, gen.config.scp_slant_range + xrow)
    ct, st = np.cos(theta), np.sin(theta)
    d = np.arange(psf_size) - psf_size // 2
    DR, DC = np.meshgrid(d, d, indexing="ij")
    U = DR * gen.config.row_ss
    V = DC * gen.config.col_ss
    return U * ct + V * st, -U * st + V * ct

_orig = CleanPhysicsConfig.from_sicd_handler.__func__
def _bad(cls, handler, chip_start=(0, 0)):
    cfg = _orig(cls, handler, chip_start)
    cfg.scp_slant_range = 10000.0                      # <-- REGRESSION
    return cfg
CleanPhysicsConfig.from_sicd_handler = classmethod(_bad)
