import os
import lxml.etree as etree
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Tuple, Union, Literal, Dict
import sarkit.sicd as ss

from clean_sar.sicd_handler import SICDHandler
from clean_sar.utils import taylor_window_1d, get_1d_window


class EnhancedSICDHandler(SICDHandler):
    """
    Enhanced SICDHandler that dynamically loads SCPCOA SlantRange and additional
    geometric parameters for exact spatially-varying PSF synthesis.
    """
    def _parse_metadata(self):
        super()._parse_metadata()
        
        # Load SlantRange from SCPCOA (m)
        slant_range_val = self._safe_load("./{*}SCPCOA/{*}SlantRange", None)
        if slant_range_val is not None:
            self.slant_range = float(slant_range_val)
        else:
            # Fallback to 10km if not found
            self.slant_range = 10000.0


class EnhancedPSFGenerator:
    """
    Enhanced Spatially-Varying PSF & Clean Beam Generator with:
    1. Correct half-power (-3dB) Gaussian restoring beam sigma scaling:
       sigma = ImpRespWid / (2 * sqrt(ln 2))
    2. Dynamic slant range R0 from SICD SCPCOA metadata
    3. Bounded LRU cache to prevent GPU VRAM exhaustion
    4. Analytic window tapering support (Taylor / Hamming / Hann)
    """

    def __init__(self, sicd: Union[SICDHandler, EnhancedSICDHandler], max_cache_size: int = 4096):
        self.sicd = sicd
        self.max_cache_size = max_cache_size
        self._cache_dirty: OrderedDict = OrderedDict()
        self._cache_clean: OrderedDict = OrderedDict()
        
        # Determine slant range R0
        if hasattr(self.sicd, "slant_range") and self.sicd.slant_range is not None:
            self.r0 = float(self.sicd.slant_range)
        else:
            # Check XML directly
            sr = self.sicd._safe_load("./{*}SCPCOA/{*}SlantRange", 10000.0)
            self.r0 = float(sr)

    def clear_cache(self):
        self._cache_dirty.clear()
        self._cache_clean.clear()

    def _get_metric_coords(
        self,
        row: Union[int, float],
        col: Union[int, float],
        chip_origin: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float, float, float]:
        if chip_origin is not None:
            r_g, c_g = self.sicd.chip_to_global_rowcol(row, col, chip_origin[0], chip_origin[1])
        else:
            r_g, c_g = float(row), float(col)

        xrow, ycol = self.sicd.global_to_metric(r_g, c_g)
        return float(r_g), float(c_g), float(xrow), float(ycol)

    def _eval_window_continuous(self, norm_freq: np.ndarray, name: Optional[str]) -> np.ndarray:
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

        kh = psf_size // 2
        k_r_vec = (np.arange(psf_size) - kh) * dk_r
        k_c_vec = (np.arange(psf_size) - kh) * dk_c
        K_R, K_C = np.meshgrid(k_r_vec, k_c_vec, indexing="ij")

        half_bw_r = bw_r / 2.0
        half_bw_c = bw_c / 2.0

        # Local geometry and dynamic slant range PFA spatial frequency distortion
        if self.sicd.is_pfa:
            theta_local = np.arctan2(ycol, self.r0 + xrow)
            cos_t = np.cos(theta_local)
            sin_t = np.sin(theta_local)
            K_R_prime = K_R * cos_t + K_C * sin_t
            K_C_prime = -K_R * sin_t + K_C * cos_t
        else:
            K_R_prime = K_R
            K_C_prime = K_C

        mask_r = np.abs(K_R_prime) <= half_bw_r
        mask_c = np.abs(K_C_prime) <= half_bw_c
        aperture_mask = mask_r & mask_c

        w_r_name = window_row if window_row is not None else self.sicd.row_wgt_name
        w_c_name = window_col if window_col is not None else self.sicd.col_wgt_name

        norm_kr = np.clip(K_R_prime / half_bw_r, -1.0, 1.0)
        norm_kc = np.clip(K_C_prime / half_bw_c, -1.0, 1.0)

        w_2d_r = self._eval_window_continuous(norm_kr, w_r_name)
        w_2d_c = self._eval_window_continuous(norm_kc, w_c_name)
        window_2d = w_2d_r * w_2d_c

        spectrum = np.zeros((psf_size, psf_size), dtype=np.complex128)
        spectrum[aperture_mask] = window_2d[aperture_mask]

        psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))

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

        # Local shear angle using dynamic slant range
        theta_local = np.arctan2(ycol, self.r0 + xrow)
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
        - 'gaussian': 2D elliptical Gaussian strictly matched to 3dB (half-power) width:
                      sigma = ImpRespWid / (2 * sqrt(ln 2)) ≈ ImpRespWid / 1.66511
        - 'mainlobe': Truncated mainlobe of the dirty PSF without sidelobes.
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

            # EXACT 3dB (half-power) Gaussian sigma formula:
            # P(x) = exp(-x^2 / sigma^2); P(W/2) = 0.5 => sigma = W / (2 * sqrt(ln 2))
            factor_3db = 2.0 * np.sqrt(np.log(2.0)) # ≈ 1.665109
            sigma_r = wid_r / factor_3db
            sigma_c = wid_c / factor_3db

            u_vec = (np.arange(psf_size) - kh) * ss_r
            v_vec = (np.arange(psf_size) - kh) * ss_c
            U, V = np.meshgrid(u_vec, v_vec, indexing="ij")

            # Local shear rotation with dynamic slant range
            theta_local = np.arctan2(ycol, self.r0 + xrow)
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
        LRU-cached retrieval of (dirty_psf, clean_beam) tensors on device.
        """
        r_g = int(row + chip_origin[0]) if chip_origin is not None else int(row)
        c_g = int(col + chip_origin[1]) if chip_origin is not None else int(col)

        key_dirty = (r_g, c_g, psf_size, method)
        key_clean = (r_g, c_g, psf_size, beam_type)

        if key_dirty in self._cache_dirty:
            dirty_t = self._cache_dirty[key_dirty]
            self._cache_dirty.move_to_end(key_dirty)
        else:
            if method == "kspace":
                dirty_np = self.compute_psf_kspace(r_g, c_g, psf_size=psf_size)
            else:
                dirty_np = self.compute_psf_analytic(r_g, c_g, psf_size=psf_size)
            dirty_t = torch.as_tensor(dirty_np, dtype=torch.complex64, device=device)
            if len(self._cache_dirty) >= self.max_cache_size:
                self._cache_dirty.popitem(last=False)
            self._cache_dirty[key_dirty] = dirty_t

        if key_clean in self._cache_clean:
            clean_t = self._cache_clean[key_clean]
            self._cache_clean.move_to_end(key_clean)
        else:
            clean_np = self.compute_clean_beam(r_g, c_g, psf_size=psf_size, beam_type=beam_type)
            clean_t = torch.as_tensor(clean_np, dtype=torch.complex64, device=device)
            if len(self._cache_clean) >= self.max_cache_size:
                self._cache_clean.popitem(last=False)
            self._cache_clean[key_clean] = clean_t

        return dirty_t, clean_t
