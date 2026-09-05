import glob
import numpy as np
import pytest

from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.sicd_handler import SICDHandler


@pytest.fixture(scope="module")
def handler(test_sicd_path):
    return SICDHandler(test_sicd_path)


def _oversampled_config(bw=1.5, oversample=16.0, wgt="UNIFORM"):
    ss = 1.0 / (bw * oversample)
    return CleanPhysicsConfig(
        row_ss=ss, col_ss=ss, row_bw=bw, col_bw=bw,
        row_wid=0.886 / bw, col_wid=0.886 / bw,
        scp_slant_range=600_000.0, scp_row=0.0, scp_col=0.0,
        row_wgt=wgt, col_wgt=wgt,
    )


def test_psf_generation(handler):
    config = CleanPhysicsConfig.from_sicd_handler(handler)
    psf_gen = PSFGenerator(config)

    scp_r, scp_c = handler.scp_pixel
    psf = psf_gen.compute_psf(scp_r, scp_c, psf_size=65)

    assert psf.shape == (65, 65)
    assert np.iscomplexobj(psf)
    kh = 65 // 2
    assert np.isclose(np.abs(psf[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].real, 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].imag, 0.0, atol=1e-5)


def test_clean_beam_generation(handler):
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


def test_slant_range_comes_from_scpcoa(handler):
    """Guards against hardcoded slant range; checks library matches SCPCOA XML."""
    cfg = CleanPhysicsConfig.from_sicd_handler(handler)
    xml_value = float(handler.xh.load("./{*}SCPCOA/{*}SlantRange"))

    assert cfg.scp_slant_range == pytest.approx(xml_value, rel=1e-9)
    assert cfg.scp_slant_range > 100_000.0


@pytest.mark.parametrize("bw", [0.6, 1.5, 3.0])
def test_gaussian_beam_halfpower_width_matches_imprespwid(bw):
    """Recovers sigma from the computed beam array and checks -3dB width equals ImpRespWid."""
    cfg = _oversampled_config(bw=bw)
    gen = PSFGenerator(cfg)

    n = 129
    c = n // 2
    beam = np.abs(gen.compute_clean_beam(0, 0, psf_size=n, beam_type="gaussian"))

    assert beam[c, c] == pytest.approx(1.0, abs=1e-6)

    for axis, ss, wid in (("row", cfg.row_ss, cfg.row_wid),
                          ("col", cfg.col_ss, cfg.col_wid)):
        nxt = beam[c + 1, c] if axis == "row" else beam[c, c + 1]
        sigma = ss / np.sqrt(-2.0 * np.log(nxt))
        width_3db = 2.0 * sigma * np.sqrt(np.log(2.0))
        assert width_3db == pytest.approx(wid, rel=2e-3)


def test_dirty_psf_matches_independent_fft_ipr():
    """Ground truth is the inverse FT of a rect aperture computed by FFT."""
    cfg = _oversampled_config(bw=1.5, wgt="UNIFORM")
    gen = PSFGenerator(cfg)

    n = 129
    c = n // 2
    psf = gen.compute_psf(0, 0, psf_size=n)
    cut = np.real(psf[:, c])

    M = 1 << 15
    k = np.fft.fftfreq(M, d=cfg.row_ss)
    W = (np.abs(k) <= cfg.row_bw / 2.0).astype(float)
    ipr = np.fft.fftshift(np.fft.ifft(W)).real
    ipr /= ipr.max()
    x_ref = (np.arange(M) - M // 2) * (1.0 / (M * (k[1] - k[0])))

    x_cut = (np.arange(n) - c) * cfg.row_ss
    expected = np.interp(x_cut, x_ref, ipr)

    assert np.max(np.abs(cut - expected)) < 1e-3


def _peak_sidelobe_db(cut):
    a = np.abs(cut) / np.max(np.abs(cut))
    c = int(np.argmax(a))
    i = c
    while i + 1 < len(a) and a[i + 1] < a[i]:
        i += 1
    return 20.0 * np.log10(np.max(a[i:]))


@pytest.mark.parametrize("wgt,expected_db,tol", [
    ("UNIFORM", -13.26, 0.6),
    ("TAYLOR", -30.0, 1.5),
    ("HAMMING", -42.7, 3.0),
    ("HANN", -31.5, 3.0),
])
def test_psf_sidelobe_level_matches_window(wgt, expected_db, tol):
    """Calls compute_psf and measures the sidelobe level of the array it returns."""
    cfg = _oversampled_config(bw=1.5, oversample=24.0, wgt=wgt)
    gen = PSFGenerator(cfg)

    n = 401
    c = n // 2
    psf = gen.compute_psf(0, 0, psf_size=n, window_row=wgt, window_col=wgt)
    psl = _peak_sidelobe_db(np.real(psf[:, c]))

    assert psl == pytest.approx(expected_db, abs=tol)


def test_taylor_sidelobes_below_uniform():
    cfg = _oversampled_config(bw=1.5, oversample=24.0)
    gen = PSFGenerator(cfg)
    n, c = 401, 200
    psl_u = _peak_sidelobe_db(np.real(gen.compute_psf(0, 0, psf_size=n,
                                                     window_row="UNIFORM",
                                                     window_col="UNIFORM")[:, c]))
    psl_t = _peak_sidelobe_db(np.real(gen.compute_psf(0, 0, psf_size=n,
                                                     window_row="TAYLOR",
                                                     window_col="TAYLOR")[:, c]))
    assert psl_t < psl_u - 10.0


@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_psf_peak_is_unity(wgt):
    gen = PSFGenerator(_oversampled_config(wgt=wgt))
    psf = gen.compute_psf(0, 0, psf_size=65)
    assert np.abs(psf[32, 32]) == pytest.approx(1.0, abs=1e-6)


def test_psf_lru_cache(handler):
    config = CleanPhysicsConfig.from_sicd_handler(handler)
    psf_gen = PSFGenerator(config, max_cache_size=5)

    for i in range(10):
        d_t, c_t = psf_gen.get_psfs_torch(row=100 + i, col=100 + i, psf_size=33)

    assert len(psf_gen._cache_dirty) <= 5
    assert len(psf_gen._cache_clean) <= 5
