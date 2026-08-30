import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")

from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator
from clean_sar.engine import run_hogbom_clean, CleanResult

def audit_gaussian_beam_math():
    print("=" * 70)
    print("1. Restoring Clean Beam Mathematical Derivation Audit")
    print("=" * 70)
    
    # Let W = 1.0 m (3dB / half-power width from SICD ImpRespWid)
    W = 1.0
    
    # Current codebase implementation
    fwhm_factor_code = 2.0 * np.sqrt(2.0 * np.log(2.0)) # 2.35482
    sigma_code = W / fwhm_factor_code
    
    # Physical 3dB (half-power) formulation for complex SAR
    # P(x) = |exp(-x^2 / (2*sigma^2))|^2 = exp(-x^2 / sigma^2)
    # P(W/2) = exp(-(W/2)^2 / sigma^2) = 0.5 => sigma = W / (2*sqrt(ln 2))
    factor_half_power = 2.0 * np.sqrt(np.log(2.0)) # 1.66511
    sigma_correct_3db = W / factor_half_power
    
    x_half = W / 2.0
    
    amp_code = np.exp(-0.5 * (x_half / sigma_code)**2)
    power_code = amp_code ** 2
    power_code_db = 10.0 * np.log10(power_code)
    
    amp_correct = np.exp(-0.5 * (x_half / sigma_correct_3db)**2)
    power_correct = amp_correct ** 2
    power_correct_db = 10.0 * np.log10(power_correct)
    
    print(f"Codebase sigma:              {sigma_code:.4f}")
    print(f"Power at W_3dB/2 in code:    {power_code:.4f} ({power_code_db:.2f} dB)")
    print(f"Correct 3dB sigma:           {sigma_correct_3db:.4f}")
    print(f"Power at W_3dB/2 in correct: {power_correct:.4f} ({power_correct_db:.2f} dB)")
    print(f"Discrepancy: Beam in codebase is narrower by factor of {sigma_correct_3db / sigma_code:.4f} (sqrt(2))")
    print(f"Effect: Restoring beam in codebase is matched to -6 dB power width rather than -3 dB half-power width.")


def audit_pfa_shear_formula():
    print("\n" + "=" * 70)
    print("2. PFA Spatial Shearing / Polar Angle Formulation Audit")
    print("=" * 70)
    
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    handler = SICDHandler(data_path)
    
    print(f"Loaded SICD: {os.path.basename(data_path)}")
    print(f"SICD PFA PolarAngPoly: {handler.pfa_meta.get('PolarAngPoly')}")
    print(f"SICD PFA SpatialFreqSFPoly: {handler.pfa_meta.get('SpatialFreqSFPoly')}")
    
    # In psf.py (line 129 and line 211):
    # theta_local = np.arctan2(ycol, 10000.0 + xrow)
    print("\nIn psf.py lines 129 and 211:")
    print("  theta_local = np.arctan2(ycol, 10000.0 + xrow)")
    print("Analysis:")
    print("  - The 10000.0 constant represents a hardcoded 10,000 meter (10 km) nominal range.")
    print("  - In actual SAR collections, the slant range to SCP (R0) varies per collection (e.g. 500 km for spaceborne, 10-50 km for airborne).")
    print("  - In SICD standard (NGA.STND.0024-1, Vol 1 / Vol 3), the PFA spatial frequency grid distortion at metric coordinate (xrow, ycol)")
    print("    is governed by the Polar Angle and Spatial Frequency Scale Factor polynomials or the sensor slant range vector to SCP.")


def audit_analytic_windowing():
    print("\n" + "=" * 70)
    print("3. PSFGenerator compute_psf_analytic Windowing Audit")
    print("=" * 70)
    
    # In psf.py lines 169-226:
    # def compute_psf_analytic(self, row, col, psf_size=65, window_row=None, window_col=None, chip_origin=None):
    # Notice: window_row and window_col are in the signature, but completely unused!
    # It directly does:
    # resp_u = np.sinc(bw_r * U_prime)
    # resp_v = np.sinc(bw_c * V_prime)
    print("In psf.py compute_psf_analytic:")
    print("  - Signature accepts `window_row` and `window_col` parameters.")
    print("  - Implementation only computes unwindowed sinc: np.sinc(bw_r * U_prime) * np.sinc(bw_c * V_prime)")
    print("  - Sidelobes in `compute_psf_analytic` will be unwindowed sinc sidelobes (~ -13.3 dB),")
    print("    even if SICD Grid.Row.WgtType specifies Taylor or Hamming.")


def audit_caching_and_memory():
    print("\n" + "=" * 70)
    print("4. Exact Coordinate Caching & Memory Scalability Audit")
    print("=" * 70)
    
    # For a 65x65 complex64 PSF:
    # 65 * 65 * 8 bytes = 33,800 bytes (~33 KB) per PSF
    # Dirty + Clean beam = ~67.6 KB per unique coordinate.
    # For a large image CLEAN with 50,000 iterations finding 20,000 unique coordinates:
    # 20,000 * 67.6 KB = 1.35 GB of GPU VRAM.
    # For 129x129 kernels: 129 * 129 * 8 * 2 = 266 KB per coord => 20,000 * 266 KB = 5.3 GB VRAM.
    psf_size = 65
    bytes_per_psf = psf_size * psf_size * 8
    total_bytes_per_coord = bytes_per_psf * 2
    print(f"PSF Kernel Size: {psf_size}x{psf_size}")
    print(f"Memory per unique (r, c) coordinate (Dirty + Clean): {total_bytes_per_coord / 1024:.2f} KB")
    print(f"Memory for 1,000 unique peaks:   {total_bytes_per_coord * 1000 / (1024**2):.2f} MB")
    print(f"Memory for 10,000 unique peaks:  {total_bytes_per_coord * 10000 / (1024**2):.2f} MB")
    print(f"Memory for 50,000 unique peaks:  {total_bytes_per_coord * 50000 / (1024**2):.2f} MB")
    print("Cache eviction policy:")
    print("  - Currently unbounded dictionary (_cache_dirty and _cache_clean).")
    print("  - PSFGenerator.clear_cache() is provided, but during a single run_hogbom_clean loop, cache grows monotonically.")


if __name__ == "__main__":
    audit_gaussian_beam_math()
    audit_pfa_shear_formula()
    audit_analytic_windowing()
    audit_caching_and_memory()
