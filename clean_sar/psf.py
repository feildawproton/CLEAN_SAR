import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Tuple, Union, Literal, Dict
from .config import CleanPhysicsConfig


class PSFGenerator:
    """
    Spatially-varying Impulse Response (IPR) / Point Spread Function (PSF) generator
    parameterized by a pure scalar CleanPhysicsConfig.

    Computes the exact analytic spatial-domain IPR via continuous modulated sinc series,
    polar shear angle rotation, and aperture weighting.
    """

    def __init__(self, config: CleanPhysicsConfig, max_cache_size: int = 4096):
        self.config = config
        self.max_cache_size = max_cache_size
        self._cache_dirty: OrderedDict[Tuple, torch.Tensor] = OrderedDict()
        self._cache_clean: OrderedDict[Tuple, torch.Tensor] = OrderedDict()

    def clear_cache(self):
        """Clears precomputed PSF tensors from memory."""
        self._cache_dirty.clear()
        self._cache_clean.clear()

    def _get_metric_coords(
        self,
        row: Union[int, float],
        col: Union[int, float],
    ) -> Tuple[float, float, float, float]:
        """
        Resolves input row/col (chip or global) to global (row_g, col_g) and metric (xrow, ycol).
        """
        r_g, c_g = self.config.chip_to_global(float(row), float(col))
        xrow, ycol = self.config.global_to_metric(r_g, c_g)
        return float(r_g), float(c_g), float(xrow), float(ycol)

    def compute_psf(
        self,
        row: Union[int, float],
        col: Union[int, float],
        psf_size: int = 65,
        window_row: Optional[str] = None,
        window_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Computes the local complex Dirty PSF analytically using modulated sinc series
        rotated by the exact local polar shear angle theta(x, y).

        Parameters
        ----------
        row, col : int or float
            Image pixel coordinates (chip local or global).
        psf_size : int
            Kernel dimension (must be odd, default: 65).
        window_row, window_col : str, optional
            Aperture weighting window name ('UNIFORM', 'TAYLOR', 'HAMMING', 'HANN').
            If None, uses metadata window from config.

        Returns
        -------
        np.ndarray
            2D complex dirty PSF array of shape (psf_size, psf_size), normalized to 1.0 at center.
        """
        if psf_size % 2 == 0:
            psf_size += 1

        _, _, xrow, ycol = self._get_metric_coords(row, col)

        # Exact local polar shear angle theta relative to SCP and slant range R0
        theta = np.arctan2(ycol, self.config.scp_slant_range + xrow)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dr_idx = np.arange(psf_size) - psf_size // 2
        dc_idx = np.arange(psf_size) - psf_size // 2
        DR, DC = np.meshgrid(dr_idx, dc_idx, indexing="ij")

        U = DR * self.config.row_ss
        V = DC * self.config.col_ss

        # Rotate spatial metric coordinates by local polar angle
        U_prime =  U * cos_t + V * sin_t
        V_prime = -U * sin_t + V * cos_t

        w_row_type = (window_row or self.config.row_wgt or "UNIFORM").upper()
        w_col_type = (window_col or self.config.col_wgt or "UNIFORM").upper()

        def _eval_1d_pattern(pos, bw, wgt):
            if wgt in ["UNIFORM", "RECT", "RECTANGULAR", "NONE"]:
                return np.sinc(bw * pos)
            elif wgt == "HAMMING":
                return (
                    0.54 * np.sinc(bw * pos)
                    + 0.23 * np.sinc(bw * pos - 1.0)
                    + 0.23 * np.sinc(bw * pos + 1.0)
                )
            elif wgt in ["HANN", "HANNING"]:
                return (
                    0.5 * np.sinc(bw * pos)
                    + 0.25 * np.sinc(bw * pos - 1.0)
                    + 0.25 * np.sinc(bw * pos + 1.0)
                )
            elif wgt == "TAYLOR":
                # Standard Taylor window (nbar=4, SLL=-30dB)
                fm = [0.29265601, -0.01578375, 0.00218104]
                pat = np.sinc(bw * pos)
                for m_idx, coeff in enumerate(fm, start=1):
                    pat += coeff * (
                        np.sinc(bw * pos - m_idx) + np.sinc(bw * pos + m_idx)
                    )
                return pat
            return np.sinc(bw * pos)

        psf_r = _eval_1d_pattern(U_prime, self.config.row_bw, w_row_type)
        psf_a = _eval_1d_pattern(V_prime, self.config.col_bw, w_col_type)

        psf = psf_r * psf_a
        max_idx = (psf_size // 2, psf_size // 2)
        peak_val = psf[max_idx]
        if np.abs(peak_val) > 0:
            psf = psf / peak_val

        return psf.astype(np.complex64)

    # Alias for backward compatibility
    compute_psf_analytic = compute_psf

    def compute_clean_beam(
        self,
        row: Union[int, float],
        col: Union[int, float],
        psf_size: int = 65,
        beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
    ) -> np.ndarray:
        """
        Computes the local clean restoring beam (matched 3dB Gaussian or dirty mainlobe).
        """
        if psf_size % 2 == 0:
            psf_size += 1

        if beam_type == "mainlobe":
            dirty = self.compute_psf(row, col, psf_size=psf_size)
            center = psf_size // 2
            beam = np.zeros_like(dirty, dtype=np.complex64)
            beam[center, center] = 1.0 + 0j
            visited = np.zeros((psf_size, psf_size), dtype=bool)
            visited[center, center] = True
            queue = [(center, center)]
            while queue:
                r, c = queue.pop(0)
                beam[r, c] = dirty[r, c]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < psf_size and 0 <= nc < psf_size and not visited[nr, nc]:
                        visited[nr, nc] = True
                        if np.abs(dirty[nr, nc]) < np.abs(dirty[r, c]) and np.abs(dirty[nr, nc]) > 0.1:
                            queue.append((nr, nc))
            return beam

        # Gaussian beam matched to -3dB resolution width
        dr_idx = np.arange(psf_size) - psf_size // 2
        dc_idx = np.arange(psf_size) - psf_size // 2
        DR, DC = np.meshgrid(dr_idx, dc_idx, indexing="ij")

        U = DR * self.config.row_ss
        V = DC * self.config.col_ss

        _, _, xrow, ycol = self._get_metric_coords(row, col)
        theta = np.arctan2(ycol, self.config.scp_slant_range + xrow)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        U_prime =  U * cos_t + V * sin_t
        V_prime = -U * sin_t + V * cos_t

        fwhm_const = 2.0 * np.sqrt(np.log(2.0))
        sigma_r = self.config.row_wid / fwhm_const
        sigma_a = self.config.col_wid / fwhm_const

        beam = np.exp(-0.5 * ((U_prime / max(sigma_r, 1e-6))**2 + (V_prime / max(sigma_a, 1e-6))**2))
        return beam.astype(np.complex64)

    def get_psfs_torch(
        self,
        row: int,
        col: int,
        psf_size: int = 65,
        beam_type: str = "gaussian",
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves exact local dirty PSF and clean beam as PyTorch tensors on target device,
        utilizing a bounded LRU cache.
        """
        r_g, c_g = self.config.chip_to_global(float(row), float(col))
        key = (int(r_g), int(c_g), psf_size, beam_type)

        if key in self._cache_dirty and key in self._cache_clean:
            self._cache_dirty.move_to_end(key)
            self._cache_clean.move_to_end(key)
            return self._cache_dirty[key], self._cache_clean[key]

        # Compute exact dirty PSF & clean beam
        dirty_np = self.compute_psf(row, col, psf_size=psf_size)
        clean_np = self.compute_clean_beam(row, col, psf_size=psf_size, beam_type=beam_type)

        dirty_t = torch.as_tensor(dirty_np, dtype=torch.complex64, device=device)
        clean_t = torch.as_tensor(clean_np, dtype=torch.complex64, device=device)

        if len(self._cache_dirty) >= self.max_cache_size:
            self._cache_dirty.popitem(last=False)
            self._cache_clean.popitem(last=False)

        self._cache_dirty[key] = dirty_t
        self._cache_clean[key] = clean_t

        return dirty_t, clean_t
