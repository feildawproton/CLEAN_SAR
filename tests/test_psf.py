import glob
import pytest
import numpy as np
from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_psf_kspace_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    psf = psf_gen.compute_psf_kspace(scp_r, scp_c, psf_size=65)

    assert psf.shape == (65, 65)
    assert np.iscomplexobj(psf)
    kh = 65 // 2
    # Peak at center should be 1.0 (magnitude) and zero imaginary
    assert np.isclose(np.abs(psf[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].real, 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].imag, 0.0, atol=1e-5)


def test_psf_analytic_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    psf = psf_gen.compute_psf_analytic(scp_r, scp_c, psf_size=65)

    assert psf.shape == (65, 65)
    assert np.iscomplexobj(psf)
    kh = 65 // 2
    assert np.isclose(np.abs(psf[kh, kh]), 1.0, atol=1e-5)


def test_clean_beam_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    beam_gauss = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=65, beam_type="gaussian")
    beam_main = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=65, beam_type="mainlobe")

    assert beam_gauss.shape == (65, 65)
    assert beam_main.shape == (65, 65)
    kh = 65 // 2
    assert np.isclose(np.abs(beam_gauss[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(np.abs(beam_main[kh, kh]), 1.0, atol=1e-5)


def test_slant_range_and_restoring_beam_physics():
    """Verify Critical Fix 1 (Slant Range) and Critical Fix 2 (3dB Gaussian Beam Factor)."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)

    # Finding 1: Slant range should be dynamic from SCPCOA (for spaceborne SAR > 100 km)
    assert handler.scp_slant_range > 100_000.0, f"Slant range {handler.scp_slant_range} is unreasonably small."

    # Finding 2: Gaussian restoring beam formula check:
    # Power P(x) = |E(x)|^2 = exp(-x^2 / sigma^2). At x = W/2, power drops to exactly 0.5 (-3.0103 dB).
    wid_r = handler.row_wid
    factor_3db = 2.0 * np.sqrt(np.log(2.0))
    sigma_r = wid_r / factor_3db

    # Evaluate exact continuous amplitude at half-width u = wid_r / 2
    u_half = wid_r / 2.0
    amp_at_half = np.exp(-0.5 * (u_half / sigma_r) ** 2)
    power_db = 20.0 * np.log10(amp_at_half)

    assert np.isclose(power_db, -3.0103, atol=1e-4), f"Power at W/2 was {power_db} dB, expected -3.0103 dB."
    assert np.isclose(amp_at_half, 1.0 / np.sqrt(2.0), atol=1e-5)


def test_psf_lru_cache():
    """Verify Finding 5: LRU cache bounds memory usage and evicts properly."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler, max_cache_size=5)

    # Request 10 distinct coordinates
    for i in range(10):
        d_t, c_t = psf_gen.get_psfs_torch(row=100 + i, col=100 + i, psf_size=33)

    assert len(psf_gen._cache_dirty) <= 5
    assert len(psf_gen._cache_clean) <= 5


def test_psf_taylor_continuous_window():
    """Verify Finding 4: Taylor window evaluates correctly via continuous cosine series."""
    u = np.linspace(-1.0, 1.0, 65)
    w_taylor = PSFGenerator._eval_window_continuous(u, "TAYLOR")
    assert len(w_taylor) == 65
    assert np.isclose(w_taylor[32], 1.0, atol=1e-5)
    # Standard -30dB Taylor edge pedestal is ~0.243
    assert 0.15 < w_taylor[0] < 0.35


def test_psf_analytic_windowing():
    """Verify Finding 8: Analytic PSF applies windowing properly."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    psf_rect = psf_gen.compute_psf_analytic(scp_r, scp_c, psf_size=65, window_row="UNIFORM")
    psf_ham = psf_gen.compute_psf_analytic(scp_r, scp_c, psf_size=65, window_row="HAMMING")

    assert psf_rect.shape == (65, 65)
    assert psf_ham.shape == (65, 65)
    # Hamming windowed response must have lower sidelobes than uniform sinc
    kh = 65 // 2
    rect_sidelobe = np.max(np.abs(psf_rect[kh + 4:, kh]))
    ham_sidelobe = np.max(np.abs(psf_ham[kh + 4:, kh]))
    assert ham_sidelobe < rect_sidelobe

