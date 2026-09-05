
import numpy as np, torch
from clean_sar.psf import PSFGenerator

_orig_psf = PSFGenerator.compute_psf
_orig_beam = PSFGenerator.compute_clean_beam

def _patched(self, row, col, psf_size=65, beam_type="gaussian", device=None):
    dirty = _orig_psf(self, row, col, psf_size=psf_size)
    clean = _orig_beam(self, row, col, psf_size=psf_size, beam_type=beam_type)
    cfg = self.config
    import copy as _c
    c2 = _c.copy(cfg); c2.row_bw = cfg.row_bw*2; c2.col_bw = cfg.col_bw*2
    g2 = PSFGenerator(c2); dirty = _orig_psf(g2, row, col, psf_size=psf_size)
    return (torch.as_tensor(dirty, dtype=torch.complex64, device=device),
            torch.as_tensor(clean, dtype=torch.complex64, device=device))

PSFGenerator.get_psfs_torch = _patched
