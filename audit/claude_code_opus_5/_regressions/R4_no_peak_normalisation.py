
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

def _pat(pos, bw, wgt):
    if wgt == "HAMMING":
        return 0.54*np.sinc(bw*pos)+0.23*np.sinc(bw*pos-1)+0.23*np.sinc(bw*pos+1)
    if wgt in ("HANN","HANNING"):
        return 0.5*np.sinc(bw*pos)+0.25*np.sinc(bw*pos-1)+0.25*np.sinc(bw*pos+1)
    if wgt == "TAYLOR":
        p = np.sinc(bw*pos)
        for m, c in enumerate([0.29265601,-0.01578375,0.00218104], start=1):
            p = p + c*(np.sinc(bw*pos-m)+np.sinc(bw*pos+m))
        return p
    return np.sinc(bw*pos)
def _bad_psf(self, row, col, psf_size=65, window_row=None, window_col=None):
    if psf_size % 2 == 0: psf_size += 1
    Up, Vp = _grid(self, row, col, psf_size)
    wr = (window_row or self.config.row_wgt or "UNIFORM").upper()
    wc = (window_col or self.config.col_wgt or "UNIFORM").upper()
    psf = _pat(Up, self.config.row_bw, wr) * _pat(Vp, self.config.col_bw, wc)
    return psf.astype(np.complex64)                    # <-- REGRESSION: no /peak
PSFGenerator.compute_psf = _bad_psf
