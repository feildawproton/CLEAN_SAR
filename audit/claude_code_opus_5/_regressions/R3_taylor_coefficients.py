
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

BAD_FM = [0.20, -0.03, 0.01]                           # <-- REGRESSION
def _pat(pos, bw, wgt):
    if wgt in ("UNIFORM","RECT","RECTANGULAR","NONE"): return np.sinc(bw*pos)
    if wgt == "HAMMING":
        return 0.54*np.sinc(bw*pos)+0.23*np.sinc(bw*pos-1)+0.23*np.sinc(bw*pos+1)
    if wgt in ("HANN","HANNING"):
        return 0.5*np.sinc(bw*pos)+0.25*np.sinc(bw*pos-1)+0.25*np.sinc(bw*pos+1)
    if wgt == "TAYLOR":
        p = np.sinc(bw*pos)
        for m, c in enumerate(BAD_FM, start=1):
            p = p + c*(np.sinc(bw*pos-m)+np.sinc(bw*pos+m))
        return p
    return np.sinc(bw*pos)
def _bad_psf(self, row, col, psf_size=65, window_row=None, window_col=None):
    if psf_size % 2 == 0: psf_size += 1
    Up, Vp = _grid(self, row, col, psf_size)
    wr = (window_row or self.config.row_wgt or "UNIFORM").upper()
    wc = (window_col or self.config.col_wgt or "UNIFORM").upper()
    psf = _pat(Up, self.config.row_bw, wr) * _pat(Vp, self.config.col_bw, wc)
    pk = psf[psf_size//2, psf_size//2]
    if abs(pk) > 0: psf = psf / pk
    return psf.astype(np.complex64)
PSFGenerator.compute_psf = _bad_psf
