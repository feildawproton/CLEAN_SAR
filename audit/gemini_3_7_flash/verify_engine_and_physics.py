import os
import sys
import time
import numpy as np
import torch

# Ensure package is in path
sys.path.insert(0, "/home/feildaw/CLEAN_SAR")

from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator
from clean_sar.engine import run_hogbom_clean, CleanResult
from clean_sar.utils import db_scale, taylor_window_1d, get_1d_window

def test_1_handler_and_io():
    print("\n--- TEST 1: SICD Handler & NITF I/O ---")
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    assert os.path.exists(data_path), f"File {data_path} not found"
    
    handler = SICDHandler(data_path)
    print(f"[+] Loaded SICD: {os.path.basename(data_path)}")
    print(f"    Image Size: {handler.num_rows} x {handler.num_cols}")
    print(f"    SS (Row, Col): ({handler.row_ss:.4f}, {handler.col_ss:.4f}) m")
    print(f"    BW (Row, Col): ({handler.row_bw:.4f}, {handler.col_bw:.4f}) 1/m")
    print(f"    Wid (Row, Col): ({handler.row_wid:.4f}, {handler.col_wid:.4f}) m")
    print(f"    SCP Pixel: {handler.scp_pixel}")
    print(f"    Is PFA: {handler.is_pfa}")
    
    # Test chipping
    r0, c0, h, w = 1000, 2000, 128, 128
    chip, chip_xml = handler.read_chip(r0, c0, r0 + h, c0 + w)
    assert chip.shape == (h, w), f"Expected chip shape ({h}, {w}), got {chip.shape}"
    assert chip.dtype == np.complex64, f"Expected complex64, got {chip.dtype}"
    print(f"[+] Sub-image chipping verified: shape {chip.shape}, dtype {chip.dtype}")
    
    # Test metric coordinate mapping
    scp_r, scp_c = handler.scp_pixel
    xr, yc = handler.global_to_metric(scp_r, scp_c)
    print(f"[+] Metric coords at SCP: ({xr:.6f}, {yc:.6f}) m (Expected ~ 0,0)")
    assert np.isclose(xr, 0.0, atol=1e-3) and np.isclose(yc, 0.0, atol=1e-3), "SCP metric conversion failed"
    
    # Round-trip coordinate mapping
    r_back, c_back = handler.metric_to_global(xr, yc)
    assert np.isclose(r_back, scp_r, atol=1e-3) and np.isclose(c_back, scp_c, atol=1e-3), "Metric round trip failed"
    print(f"[+] Coordinate round-trip verified.")


def test_2_psf_physics_and_properties():
    print("\n--- TEST 2: PSF Physics, Symmetry & Normalization ---")
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    handler = SICDHandler(data_path)
    psf_gen = PSFGenerator(handler)
    
    scp_r, scp_c = handler.scp_pixel
    psf_size = 65
    kh = psf_size // 2
    
    # 2.1 K-space PSF
    psf_k = psf_gen.compute_psf_kspace(scp_r, scp_c, psf_size=psf_size)
    assert psf_k.shape == (psf_size, psf_size)
    assert np.isclose(np.abs(psf_k[kh, kh]), 1.0, atol=1e-5), f"K-space PSF center abs != 1.0 ({np.abs(psf_k[kh, kh])})"
    assert np.isclose(psf_k[kh, kh].real, 1.0, atol=1e-5), "K-space PSF center real != 1.0"
    assert np.isclose(psf_k[kh, kh].imag, 0.0, atol=1e-5), "K-space PSF center imag != 0.0"
    print(f"[+] K-space PSF generated: shape {psf_k.shape}, center peak = {psf_k[kh, kh]}")
    
    # 2.2 Analytic PSF
    psf_a = psf_gen.compute_psf_analytic(scp_r, scp_c, psf_size=psf_size)
    assert psf_a.shape == (psf_size, psf_size)
    assert np.isclose(np.abs(psf_a[kh, kh]), 1.0, atol=1e-5)
    print(f"[+] Analytic PSF generated: shape {psf_a.shape}, center peak = {psf_a[kh, kh]}")
    
    # 2.3 Restoring Clean Beams
    beam_g = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=psf_size, beam_type="gaussian")
    beam_m = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=psf_size, beam_type="mainlobe")
    assert np.isclose(np.abs(beam_g[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(np.abs(beam_m[kh, kh]), 1.0, atol=1e-5)
    print(f"[+] Clean beams (Gaussian & Mainlobe) generated: center peaks = 1.0")
    
    # Check that peak is strictly centered
    assert np.argmax(np.abs(psf_k)) == (kh * psf_size + kh)
    assert np.argmax(np.abs(psf_a)) == (kh * psf_size + kh)
    assert np.argmax(np.abs(beam_g)) == (kh * psf_size + kh)
    print("[+] All PSF peaks confirmed at exact center (kh, kw).")


def test_3_synthetic_clean_recovery():
    print("\n--- TEST 3: Synthetic Target Deconvolution Accuracy ---")
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    handler = SICDHandler(data_path)
    psf_gen = PSFGenerator(handler)
    
    H, W = 128, 128
    psf_size = 45
    kh, kw = psf_size // 2, psf_size // 2
    
    # Ground truth synthetic point targets
    # (row, col, amplitude, phase_rad)
    targets = [
        (40, 40, 10.0, 0.0),
        (40, 80, 5.0, np.pi / 4),
        (80, 50, 7.5, -np.pi / 3),
        (85, 95, 3.0, np.pi / 2),
    ]
    
    dirty_synthetic = np.zeros((H, W), dtype=np.complex64)
    origin = (1000, 1000)
    
    # Synthesize dirty image by placing exact spatially varying PSFs
    for r, c, amp, ph in targets:
        c_val = amp * np.exp(1j * ph)
        psf = psf_gen.compute_psf_kspace(r, c, psf_size=psf_size, chip_origin=origin)
        r_min, r_max = max(0, r - kh), min(H, r + kh + 1)
        c_min, c_max = max(0, c - kw), min(W, c + kw + 1)
        pr_min = kh - (r - r_min)
        pr_max = kh + (r_max - r)
        pc_min = kw - (c - c_min)
        pc_max = kw + (c_max - c)
        dirty_synthetic[r_min:r_max, c_min:c_max] += c_val * psf[pr_min:pr_max, pc_min:pc_max]
    
    init_max = np.max(np.abs(dirty_synthetic))
    print(f"[+] Synthesized dirty image with {len(targets)} targets. Max dirty magnitude = {init_max:.4f}")
    
    # Run CLEAN
    res = run_hogbom_clean(
        dirty_image=dirty_synthetic,
        psf_generator=psf_gen,
        method="kspace",
        beam_type="gaussian",
        psf_size=psf_size,
        gain=0.1,
        threshold=0.01,
        max_iters=1000,
        chip_origin=origin,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    
    res_max = np.max(np.abs(res.residual_image))
    print(f"[+] CLEAN finished in {res.iterations} iterations ({res.execution_time_sec:.3f}s).")
    print(f"    Initial peak: {init_max:.4f} -> Final residual peak: {res_max:.4f} (Reduction: {20*np.log10(res_max/init_max):.2f} dB)")
    
    # Verify recovered components
    print("    Recovered component amplitudes vs Ground Truth:")
    for r, c, amp, ph in targets:
        true_c = amp * np.exp(1j * ph)
        # sum recovered in a 3x3 window around target
        rec_c = np.sum(res.components_map[r-1:r+2, c-1:c+2])
        amp_err = abs(abs(rec_c) - amp) / amp
        print(f"      Target at ({r}, {c}): True={true_c:.2f} (|{abs(true_c):.2f}|), Recovered={rec_c:.2f} (|{abs(rec_c):.2f}|), Rel Amp Err={amp_err*100:.2f}%")
        assert amp_err < 0.05, f"Component at ({r}, {c}) recovery error too high ({amp_err:.3f})"


def test_4_real_sar_chip():
    print("\n--- TEST 4: Real SAR Chip Deconvolution & Benchmark ---")
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    handler = SICDHandler(data_path)
    psf_gen = PSFGenerator(handler)
    
    # Read a bright feature chip
    r0, c0 = 2777, 6410
    h, w = 128, 128
    chip, chip_xml = handler.read_chip(r0, c0, r0 + h, c0 + w)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running on device: {device}")
    
    t0 = time.time()
    res = run_hogbom_clean(
        dirty_image=chip,
        psf_generator=psf_gen,
        method="kspace",
        beam_type="gaussian",
        psf_size=65,
        gain=0.1,
        threshold=0.02,
        max_iters=500,
        chip_origin=(r0, c0),
        device=device,
        verbose=False,
    )
    t_clean = time.time() - t0
    
    init_p = np.max(np.abs(chip))
    final_p = np.max(np.abs(res.residual_image))
    num_pts = np.count_nonzero(np.abs(res.components_map) > 0)
    print(f"[+] Real chip CLEAN completed in {t_clean:.2f}s ({res.iterations} iterations, {res.iterations/t_clean:.1f} iters/s)")
    print(f"    Peak reduction: {init_p:.4e} -> {final_p:.4e} ({20*np.log10(final_p/init_p):.2f} dB)")
    print(f"    Unique scatterers extracted: {num_pts}")
    print(f"    Cache size: dirty={len(psf_gen._cache_dirty)}, clean={len(psf_gen._cache_clean)}")


if __name__ == "__main__":
    test_1_handler_and_io()
    test_2_psf_physics_and_properties()
    test_3_synthetic_clean_recovery()
    test_4_real_sar_chip()
    print("\n[SUCCESS] All verification tests passed successfully!")
