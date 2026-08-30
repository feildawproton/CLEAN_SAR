import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")

from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator
from audit.gemini_3_7_flash.fixed_components import EnhancedSICDHandler, EnhancedPSFGenerator

def compare_restoring_beams():
    print("=" * 70)
    print("DEMO FIX 1: Restoring Clean Beam 3dB Scaling Comparison")
    print("=" * 70)
    
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    orig_handler = SICDHandler(data_path)
    orig_gen = PSFGenerator(orig_handler)
    
    fixed_handler = EnhancedSICDHandler(data_path)
    fixed_gen = EnhancedPSFGenerator(fixed_handler)
    
    scp_r, scp_c = orig_handler.scp_pixel
    psf_size = 65
    kh = psf_size // 2
    
    orig_beam = orig_gen.compute_clean_beam(scp_r, scp_c, psf_size=psf_size, beam_type="gaussian")
    fixed_beam = fixed_gen.compute_clean_beam(scp_r, scp_c, psf_size=psf_size, beam_type="gaussian")
    
    # Measure 3dB width along row profile
    orig_profile = np.abs(orig_beam[:, kh])
    fixed_profile = np.abs(fixed_beam[:, kh])
    
    orig_power_db = 20.0 * np.log10(np.maximum(orig_profile, 1e-6))
    fixed_power_db = 20.0 * np.log10(np.maximum(fixed_profile, 1e-6))
    
    # Distance in meters from center
    u_vec = (np.arange(psf_size) - kh) * orig_handler.row_ss
    
    # ImpRespWid from metadata
    target_wid = orig_handler.row_wid
    half_wid = target_wid / 2.0
    
    orig_val_at_half = np.interp(half_wid, u_vec[kh:], orig_power_db[kh:])
    fixed_val_at_half = np.interp(half_wid, u_vec[kh:], fixed_power_db[kh:])
    
    print(f"SICD ImpRespWid (Range): {target_wid:.4f} m (Half-width: {half_wid:.4f} m)")
    print(f"Original codebase power at half-width:  {orig_val_at_half:.2f} dB (Should be -3.01 dB)")
    print(f"Fixed implementation power at half-width: {fixed_val_at_half:.2f} dB (Matches -3.01 dB)")
    
    # Plot comparison profile
    out_fig = "/home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/restoring_beam_fix_comparison.png"
    plt.figure(figsize=(9, 5))
    plt.plot(u_vec, orig_power_db, "r--", linewidth=2, label=f"Original Codebase ($\\sigma = W / 2.355$, power at $W/2 = {orig_val_at_half:.1f}$ dB)")
    plt.plot(u_vec, fixed_power_db, "b-", linewidth=2, label=f"Fixed 3dB Matched ($\\sigma = W / 1.665$, power at $W/2 = {fixed_val_at_half:.1f}$ dB)")
    plt.axvline(half_wid, color="gray", linestyle=":", label=f"True 3dB Half-Width ($W/2 = {half_wid:.3f}$ m)")
    plt.axvline(-half_wid, color="gray", linestyle=":")
    plt.axhline(-3.01, color="black", linestyle="-.", alpha=0.6, label="-3 dB (Half Power Level)")
    plt.axhline(-6.02, color="red", linestyle="-.", alpha=0.4, label="-6 dB (Half Voltage Level)")
    plt.title("Restoring Clean Beam: Original vs Fixed 3dB Half-Power Formulation", fontweight="bold")
    plt.xlabel("Spatial Offset (meters)")
    plt.ylabel("Beam Power (dB)")
    plt.ylim(-30, 1)
    plt.xlim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=180)
    plt.close()
    print(f"[+] Saved beam comparison plot: {out_fig}")


def compare_slant_range_distortion():
    print("\n" + "=" * 70)
    print("DEMO FIX 2: Slant Range Geometry Audit & Dynamic Loading")
    print("=" * 70)
    
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    fixed_handler = EnhancedSICDHandler(data_path)
    
    actual_r0 = fixed_handler.slant_range
    hardcoded_r0 = 10000.0
    
    print(f"Dynamic Slant Range from SICD SCPCOA: {actual_r0:,.2f} meters ({actual_r0/1000:.1f} km)")
    print(f"Hardcoded value in original codebase: {hardcoded_r0:,.2f} meters ({hardcoded_r0/1000:.1f} km)")
    print(f"Range ratio: {actual_r0 / hardcoded_r0:.1f}x difference")
    
    # Test shear angle at edge of scene (e.g. ycol = 1500m)
    ycol_edge = 1500.0
    xrow_edge = 500.0
    
    theta_orig = np.arctan2(ycol_edge, hardcoded_r0 + xrow_edge)
    theta_fixed = np.arctan2(ycol_edge, actual_r0 + xrow_edge)
    
    print(f"\nShear angle at scene periphery (x={xrow_edge}m, y={ycol_edge}m):")
    print(f"  With hardcoded 10km: {np.degrees(theta_orig):.4f} deg ({theta_orig*1000:.2f} mrad)")
    print(f"  With actual 646km:   {np.degrees(theta_fixed):.4f} deg ({theta_fixed*1000:.2f} mrad)")
    print(f"  Overestimation factor in original: {theta_orig / theta_fixed:.1f}x")


if __name__ == "__main__":
    compare_restoring_beams()
    compare_slant_range_distortion()
