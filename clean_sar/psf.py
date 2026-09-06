import numpy as np
from typing import Optional, Tuple, Union
from .config import CleanPhysicsConfig


def calculate_psf_size(
    config: CleanPhysicsConfig,
    image_shape: Optional[Tuple[int, int]] = None,
    target_power: float = 0.999,
    min_size: int = 15,
) -> int:
    """
    Calculates the optimal odd PSF grid size based on the integrated power
    of the continuous analytic dirty PSF for the given radar physics configuration,
    with maximum size bounded by the incoming image dimensions.

    Parameters
    ----------
    config : CleanPhysicsConfig
        Radar physics configuration containing sampling, bandwidth, and weighting.
    image_shape : tuple of (int, int), optional
        Shape (H, W) of the incoming image to enforce maximum grid size constraint.
    target_power : float, default 0.999
        Fraction of total discrete PSF power enclosed by the kernel window (e.g. 99.9%).
    min_size : int, default 15
        Minimum allowable kernel dimension (ensures coverage of mainlobe and first sidelobes).

    Returns
    -------
    int
        Odd integer kernel size.
    """
    sizes = []
    for ss, bw, wid, wgt in [
        (config.row_ss, config.row_bw, config.row_wid, config.row_wgt),
        (config.col_ss, config.col_bw, config.col_wid, config.col_wgt),
    ]:
        w_str = (wgt or "UNIFORM").upper()
        max_r = 512
        if image_shape is not None:
            max_r = min(max_r, max(image_shape) // 2)
        r_grid = np.arange(0, max_r + 1, dtype=np.float32)
        u = r_grid * float(ss)
        x = float(bw) * u

        sinc_x = np.sinc(x)
        if w_str in ["UNIFORM", "RECT", "RECTANGULAR", "NONE"]:
            pat = sinc_x
        elif "TAYLOR" in w_str:
            f1, f2, f3 = 0.29265601, -0.01578375, 0.00218104
            pat = (
                sinc_x
                + f1 * (np.sinc(x - 1.0) + np.sinc(x + 1.0))
                + f2 * (np.sinc(x - 2.0) + np.sinc(x + 2.0))
                + f3 * (np.sinc(x - 3.0) + np.sinc(x + 3.0))
            )
        elif "HAMMING" in w_str:
            pat = (0.54 * sinc_x + 0.23 * (np.sinc(x - 1.0) + np.sinc(x + 1.0))) * (1.0 / 0.54)
        elif "HANN" in w_str:
            pat = (0.50 * sinc_x + 0.25 * (np.sinc(x - 1.0) + np.sinc(x + 1.0))) * 2.0
        else:
            pat = sinc_x

        pwr = pat ** 2
        tot_pwr = pwr[0] + 2.0 * np.sum(pwr[1:])
        if tot_pwr > 0:
            cum_pwr = pwr[0] + 2.0 * np.cumsum(pwr[1:])
            frac = cum_pwr / tot_pwr
            rad = int(np.searchsorted(frac, target_power)) + 1
        else:
            rad = 16

        # Ensure at least 3 resolution cells to capture mainlobe and first sidelobes
        if ss > 0 and wid > 0:
            res_cells = int(np.ceil(float(wid) / float(ss)))
            rad = max(rad, res_cells * 3)

        sizes.append(2 * rad + 1)

    grid_size = max(sizes)
    if min_size:
        grid_size = max(grid_size, min_size)

    # Maximum size bounded by the incoming image dimensions
    if image_shape is not None and len(image_shape) >= 2:
        max_dim = min(image_shape[0], image_shape[1])
        if max_dim % 2 == 0:
            max_dim -= 1
        max_dim = max(max_dim, 3)
        grid_size = min(grid_size, max_dim)

    if grid_size % 2 == 0:
        grid_size += 1

    return grid_size


class PSFGenerator:
    """
    Spatially-varying Impulse Response (IPR) / Point Spread Function (PSF) generator
    parameterized by a pure scalar CleanPhysicsConfig.

    Computes the exact analytic spatial-domain IPR via continuous modulated sinc series,
    polar shear angle rotation, and aperture weighting.
    """

    def __init__(self, config: CleanPhysicsConfig):
        self.config = config

    def clear_cache(self):
        """Clears precomputed PSF arrays from memory (no-op retained for backward compatibility)."""
        pass

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
    ) -> np.ndarray:
        """
        Computes the local clean restoring beam (matched 3dB Gaussian).
        """
        if psf_size % 2 == 0:
            psf_size += 1

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
