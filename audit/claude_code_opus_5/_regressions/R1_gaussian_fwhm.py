
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

def _bad_beam(self, row, col, psf_size=65, beam_type="gaussian"):
    if psf_size % 2 == 0: psf_size += 1
    Up, Vp = _grid(self, row, col, psf_size)
    fwhm_const = 2.0 * np.sqrt(2.0 * np.log(2.0))     # <-- REGRESSION
    sr = self.config.row_wid / fwhm_const
    sc = self.config.col_wid / fwhm_const
    return np.exp(-0.5*((Up/max(sr,1e-6))**2 + (Vp/max(sc,1e-6))**2)).astype(np.complex64)
PSFGenerator.compute_clean_beam = _bad_beam
