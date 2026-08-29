import numpy as np
import torch
from typing import Optional, Tuple, Union, Literal, Dict
from .sicd_handler import SICDHandler
from .utils import taylor_window_1d


class PSFGenerator:
    """
    Spatially-varying Impulse Response (IPR) / Point Spread Function (PSF) generator
    for SAR complex images formed from SICD metadata.

    Supports:
    - Option A: 2D Spatial Frequency Support + IFFT (k-space domain)
    - Option B: Analytic / Polynomial Spatial-Domain formulation with local geometric shear
    - Clean Beam generation (matched Gaussian restoring beam)
    - Exact coordinate caching for fast per-iteration evaluation
    """

    def __init__(self, sicd: SICDHandler):
        self.sicd = sicd
        self._cache_dirty: Dict[Tuple, torch.Tensor] = {}
        self._cache_clean: Dict[Tuple, torch.Tensor] = {}

    def clear_cache(self):
        """Clears precomputed PSF tensors from memory."""
        self._cache_dirty.clear()
        self._cache_clean.clear()

    def _get_metric_coords(
        self,
        row: Union[int, float],
        col: Union[int, float],
        chip_origin: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float, float, float]:
        """
        Resolves input row/col (global or chip) to global (row_g, col_g) and metric (xrow, ycol).
        """
        if chip_origin is not None:
            r_g, c_g = self.sicd.chip_to_global_rowcol(row, col, chip_origin[0], chip_origin[1])
        else:
            r_g, c_g = float(row), float(col)

        xrow, ycol = self.sicd.global_to_metric(r_g, c_g)
        return float(r_g), float(c_g), float(xrow), float(ycol)

    def _eval_window_continuous(self, norm_freq: np.ndarray, name: Optional[str]) -> np.ndarray:
        """
        Evaluates a 1D weighting window over continuous normalized frequency [-1, 1].
        """
        if name is None or name.upper() in ["UNIFORM", "RECT", "RECTANGULAR", "NONE"]:
            w = np.ones_like(norm_freq, dtype=np.float64)
            w[np.abs(norm_freq) > 1.0] = 0.0
            return w

        name_upper = name.upper()
        in_band = np.abs(norm_freq) <= 1.0
        w = np.zeros_like(norm_freq, dtype=np.float64)

        if name_upper == "HAMMING":
            w[in_band] = 0.54 + 0.46 * np.cos(np.pi * norm_freq[in_band])
        elif name_upper in ["HANN", "HANNING"]:
            w[in_band] = 0.5 * (1.0 + np.cos(np.pi * norm_freq[in_band]))
        elif name_upper == "TAYLOR":
            u = norm_freq[in_band]
            n_pts = len(u)
            if n_pts > 0:
                t_w = taylor_window_1d(n_pts, nbar=4, sll=-30.0)
                w[in_band] = t_w
        else:
            w[in_band] = 1.0

        return w

    def compute_psf_kspace(
        self,
        row: Union[int, float],
        col: Union[int, float],
        psf_size: int = 65,
        window_row: Optional[str] = None,
        window_col: Optional[str] = None,
        chip_origin: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Option A: Computes the local complex Dirty PSF via 2D spatial frequency aperture support and IFFT.

        Parameters
        ----------
        row, col : int or float
            Image coordinates (global or relative to chip_origin).
        psf_size : int
            Kernel dimension (must be odd, e.g. 65 or 129).
        window_row, window_col : str, optional
            Aperture weighting windows. If None, uses SICD Grid.Row/Col.WgtType.
        chip_origin : tuple of (start_row, start_col), optional

        Returns
        -------
        psf : np.ndarray (complex64)
            Normalized 2D complex dirty PSF of shape (psf_size, psf_size) centered at (psf_size//2, psf_size//2).
        """
        if psf_size % 2 == 0:
            psf_size += 1

        r_g, c_g, xrow, ycol = self._get_metric_coords(row, col, chip_origin)

        ss_r = self.sicd.row_ss
        ss_c = self.sicd.col_ss
        bw_r = self.sicd.row_bw
        bw_c = self.sicd.col_bw

        # Frequency sampling steps: dK = 1 / (N * SS)
        dk_r = 1.0 / (psf_size * ss_r)
        dk_c = 1.0 / (psf_size * ss_c)

        # Baseband frequency grid centered at (0, 0)
        kh = psf_size // 2
        k_r_vec = (np.arange(psf_size) - kh) * dk_r
        k_c_vec = (np.arange(psf_size) - kh) * dk_c
        K_R, K_C = np.meshgrid(k_r_vec, k_c_vec, indexing="ij")

        # Nominal bandwidth limits in baseband
        half_bw_r = bw_r / 2.0
        half_bw_c = bw_c / 2.0

        # Local geometry and PFA spatial frequency distortion
        if self.sicd.is_pfa:
            # Local polar angle variation from position
            theta_local = np.arctan2(ycol, 10000.0 + xrow)

            # Rotated / sheared frequency coordinates
            cos_t = np.cos(theta_local)
            sin_t = np.sin(theta_local)
            K_R_prime = K_R * cos_t + K_C * sin_t
            K_C_prime = -K_R * sin_t + K_C * cos_t
        else:
            K_R_prime = K_R
            K_C_prime = K_C

        # Active aperture mask
        mask_r = np.abs(K_R_prime) <= half_bw_r
        mask_c = np.abs(K_C_prime) <= half_bw_c
        aperture_mask = mask_r & mask_c

        # Window weighting
        w_r_name = window_row if window_row is not None else self.sicd.row_wgt_name
        w_c_name = window_col if window_col is not None else self.sicd.col_wgt_name

        norm_kr = np.clip(K_R_prime / half_bw_r, -1.0, 1.0)
        norm_kc = np.clip(K_C_prime / half_bw_c, -1.0, 1.0)

        w_2d_r = self._eval_window_continuous(norm_kr, w_r_name)
        w_2d_c = self._eval_window_continuous(norm_kc, w_c_name)
        window_2d = w_2d_r * w_2d_c

        spectrum = np.zeros((psf_size, psf_size), dtype=np.complex128)
        spectrum[aperture_mask] = window_2d[aperture_mask]

        # 2D Inverse FFT to image domain
        psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))

        # Normalize peak to 1.0 and zero phase at center
        center_val = psf[kh, kh]
        if np.abs(center_val) > 0:
            psf = psf / center_val

        return psf.astype(np.complex64)

    def compute_psf_analytic(
        self,
        row: Union[int, float],
        col: Union[int, float],
        psf_size: int = 65,
        window_row: Optional[str] = None,
        window_col: Optional[str] = None,
        chip_origin: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Option B: Computes the local complex Dirty PSF analytically in the spatial domain
        with local resolution bandwidths and geometric shear.

        Parameters
        ----------
        row, col : int or float
            Image coordinates (global or relative to chip_origin).
        psf_size : int
            Kernel dimension (must be odd).
        chip_origin : tuple of (start_row, start_col), optional

        Returns
        -------
        psf : np.ndarray (complex64)
            Normalized 2D complex dirty PSF of shape (psf_size, psf_size).
        """
        if psf_size % 2 == 0:
            psf_size += 1

        r_g, c_g, xrow, ycol = self._get_metric_coords(row, col, chip_origin)

        ss_r = self.sicd.row_ss
        ss_c = self.sicd.col_ss
        bw_r = self.sicd.row_bw
        bw_c = self.sicd.col_bw

        kh = psf_size // 2
        u_vec = (np.arange(psf_size) - kh) * ss_r
        v_vec = (np.arange(psf_size) - kh) * ss_c
        U, V = np.meshgrid(u_vec, v_vec, indexing="ij")

        # Local shear angle from position
        theta_local = np.arctan2(ycol, 10000.0 + xrow)
        cos_t = np.cos(theta_local)
        sin_t = np.sin(theta_local)

        U_prime = U * cos_t + V * sin_t
        V_prime = -U * sin_t + V * cos_t

        resp_u = np.sinc(bw_r * U_prime)
        resp_v = np.sinc(bw_c * V_prime)
        psf = (resp_u * resp_v).astype(np.complex128)

        center_val = psf[kh, kh]
        if np.abs(center_val) > 0:
            psf = psf / center_val

        return psf.astype(np.complex64)

    def compute_clean_beam(
        self,
        row: Union[int, float],
        col: Union[int, float],
        psf_size: int = 65,
        beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
        chip_origin: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generates the Clean (restoring) beam.

        - 'gaussian': 2D elliptical Gaussian matched to local 3dB impulse response widths.
        - 'mainlobe': Truncated mainlobe of the dirty PSF without sidelobes.

        Parameters
        ----------
        row, col : int or float
            Image coordinates.
        psf_size : int
            Kernel dimension (must be odd).
        beam_type : str
            'gaussian' or 'mainlobe'.

        Returns
        -------
        clean_beam : np.ndarray (complex64)
            Sidelobe-free restoring beam normalized to peak 1.0.
        """
        if psf_size % 2 == 0:
            psf_size += 1

        r_g, c_g, xrow, ycol = self._get_metric_coords(row, col, chip_origin)
        kh = psf_size // 2

        if beam_type == "gaussian":
            wid_r = self.sicd.row_wid
            wid_c = self.sicd.col_wid
            ss_r = self.sicd.row_ss
            ss_c = self.sicd.col_ss

            # FWHM to Gaussian sigma: FWHM ≈ 2.35482 * sigma
            fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
            sigma_r = wid_r / fwhm_factor
            sigma_c = wid_c / fwhm_factor

            u_vec = (np.arange(psf_size) - kh) * ss_r
            v_vec = (np.arange(psf_size) - kh) * ss_c
            U, V = np.meshgrid(u_vec, v_vec, indexing="ij")

            # Local shear rotation
            theta_local = np.arctan2(ycol, 10000.0 + xrow)
            cos_t = np.cos(theta_local)
            sin_t = np.sin(theta_local)
            U_p = U * cos_t + V * sin_t
            V_p = -U * sin_t + V * cos_t

            exponent = -0.5 * ((U_p / sigma_r) ** 2 + (V_p / sigma_c) ** 2)
            clean_beam = np.exp(exponent).astype(np.complex64)
            clean_beam /= np.abs(clean_beam[kh, kh])
            return clean_beam

        elif beam_type == "mainlobe":
            dirty_psf = self.compute_psf_kspace(row, col, psf_size=psf_size, chip_origin=chip_origin)
            mag = np.abs(dirty_psf)
            clean_beam = np.zeros_like(dirty_psf)
            window_len = int(max(self.sicd.row_wid / self.sicd.row_ss, self.sicd.col_wid / self.sicd.col_ss) * 1.5)
            r_s = max(0, kh - window_len)
            r_e = min(psf_size, kh + window_len + 1)
            c_s = max(0, kh - window_len)
            c_e = min(psf_size, kh + window_len + 1)

            sub_mag = mag[r_s:r_e, c_s:c_e]
            taper_r = np.hanning(r_e - r_s)
            taper_c = np.hanning(c_e - c_s)
            taper_2d = np.outer(taper_r, taper_c)

            clean_beam[r_s:r_e, c_s:c_e] = sub_mag * taper_2d
            clean_beam /= np.max(np.abs(clean_beam))
            return clean_beam.astype(np.complex64)
        else:
            raise ValueError(f"Unknown beam_type: {beam_type}")

    def get_psfs_torch(
        self,
        row: int,
        col: int,
        psf_size: int = 65,
        method: Literal["kspace", "analytic"] = "kspace",
        beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
        chip_origin: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves cached or computes exact (dirty_psf, clean_beam) PyTorch tensors on device
        for global coordinate (row_g, col_g).
        """
        r_g = int(row + chip_origin[0]) if chip_origin is not None else int(row)
        c_g = int(col + chip_origin[1]) if chip_origin is not None else int(col)

        key_dirty = (r_g, c_g, psf_size, method)
        key_clean = (r_g, c_g, psf_size, beam_type)

        if key_dirty in self._cache_dirty:
            dirty_t = self._cache_dirty[key_dirty]
        else:
            if method == "kspace":
                dirty_np = self.compute_psf_kspace(r_g, c_g, psf_size=psf_size)
            else:
                dirty_np = self.compute_psf_analytic(r_g, c_g, psf_size=psf_size)
            dirty_t = torch.as_tensor(dirty_np, dtype=torch.complex64, device=device)
            self._cache_dirty[key_dirty] = dirty_t

        if key_clean in self._cache_clean:
            clean_t = self._cache_clean[key_clean]
        else:
            clean_np = self.compute_clean_beam(r_g, c_g, psf_size=psf_size, beam_type=beam_type)
            clean_t = torch.as_tensor(clean_np, dtype=torch.complex64, device=device)
            self._cache_clean[key_clean] = clean_t

        return dirty_t, clean_t
