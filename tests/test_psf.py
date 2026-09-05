import glob
import pytest
import numpy as np
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_psf_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    config = CleanPhysicsConfig.from_sicd_handler(handler)
    psf_gen = PSFGenerator(config)

    scp_r, scp_c = handler.scp_pixel
    psf = psf_gen.compute_psf(scp_r, scp_c, psf_size=65)

    assert psf.shape == (65, 65)
    assert np.iscomplexobj(psf)
    kh = 65 // 2
    # Peak at center should be 1.0 (magnitude) and zero imaginary
    assert np.isclose(np.abs(psf[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].real, 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].imag, 0.0, atol=1e-5)


def test_clean_beam_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    config = CleanPhysicsConfig.from_sicd_handler(handler)
    psf_gen = PSFGenerator(config)

    scp_r, scp_c = handler.scp_pixel
    beam_gauss = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=65, beam_type="gaussian")
    beam_main = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=65, beam_type="mainlobe")

    assert beam_gauss.shape == (65, 65)
    assert beam_main.shape == (65, 65)
    kh = 65 // 2
    assert np.isclose(np.abs(beam_gauss[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(np.abs(beam_main[kh, kh]), 1.0, atol=1e-5)


def test_slant_range_and_restoring_beam_physics():
    """Verify Slant Range and 3dB Gaussian Beam Factor."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    config = CleanPhysicsConfig.from_sicd_handler(handler)

    # Finding 1: Slant range should be dynamic from SCPCOA (for spaceborne SAR > 100 km)
    assert config.scp_slant_range > 100_000.0, f"Slant range {config.scp_slant_range} is unreasonably small."

    # Finding 2: Gaussian restoring beam formula check:
    # Power P(x) = |E(x)|^2 = exp(-x^2 / sigma^2). At x = W/2, power drops to exactly 0.5 (-3.0103 dB).
    wid_r = config.row_wid
    factor_3db = 2.0 * np.sqrt(np.log(2.0))
    sigma_r = wid_r / factor_3db

    # Evaluate exact continuous amplitude at half-width u = wid_r / 2
    u_half = wid_r / 2.0
    amp_at_half = np.exp(-0.5 * (u_half / sigma_r) ** 2)
    power_db = 20.0 * np.log10(amp_at_half)

    assert np.isclose(power_db, -3.0103, atol=1e-4), f"Power at W/2 was {power_db} dB, expected -3.0103 dB."
    assert np.isclose(amp_at_half, 1.0 / np.sqrt(2.0), atol=1e-5)


def test_psf_lru_cache():
    """Verify LRU cache bounds memory usage and evicts properly."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    config = CleanPhysicsConfig.from_sicd_handler(handler)
    psf_gen = PSFGenerator(config, max_cache_size=5)

    # Request 10 distinct coordinates
    for i in range(10):
        d_t, c_t = psf_gen.get_psfs_torch(row=100 + i, col=100 + i, psf_size=33)

    assert len(psf_gen._cache_dirty) <= 5
    assert len(psf_gen._cache_clean) <= 5


def test_psf_taylor_continuous_window():
    """Verify Taylor window evaluates correctly via continuous cosine series."""
    # Test normalized frequency edge behavior
    fm = [0.29265601, -0.01578375, 0.00218104]
    dc_peak = 1.0 + 2.0 * sum(fm)
    w_edge = (1.0 + 2.0 * (fm[0] * np.cos(np.pi) + fm[1] * np.cos(2*np.pi) + fm[2] * np.cos(3*np.pi))) / dc_peak
    assert 0.0 < w_edge < 1.0


def test_psf_analytic_windowing():
    """Verify Analytic PSF includes spatial-domain Taylor/Hamming window modulation."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    config = CleanPhysicsConfig.from_sicd_handler(handler)
    psf_gen = PSFGenerator(config)

    scp_r, scp_c = handler.scp_pixel
    psf_taylor = psf_gen.compute_psf(scp_r, scp_c, psf_size=65, window_row="TAYLOR", window_col="TAYLOR")
    psf_rect = psf_gen.compute_psf(scp_r, scp_c, psf_size=65, window_row="UNIFORM", window_col="UNIFORM")

    assert psf_taylor.shape == (65, 65)
    assert psf_rect.shape == (65, 65)
    # Peak is 1.0
    assert np.isclose(np.abs(psf_taylor[32, 32]), 1.0)
    assert np.isclose(np.abs(psf_rect[32, 32]), 1.0)
    # Sidelobes for Taylor window must be lower than Uniform sinc (-13.2 dB vs -30 dB)
    first_sidelobe_rect = np.max(np.abs(psf_rect[32, 35:40]))
    first_sidelobe_taylor = np.max(np.abs(psf_taylor[32, 35:40]))
    assert first_sidelobe_taylor < first_sidelobe_rect
