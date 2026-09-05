"""
audit/claude_code_opus_5/test_psf_physics_reference.py

Drop-in replacement for the two self-referential tests in tests/test_psf.py:
    test_slant_range_and_restoring_beam_physics
    test_psf_taylor_continuous_window
Lift into tests/ roughly verbatim.

WHY THE EXISTING TESTS CANNOT WORK
----------------------------------
Both were written to lock in fixes from the earlier audit, and neither calls the
code it is protecting.

  test_slant_range_and_restoring_beam_physics re-derives the Gaussian sigma
  inline and asserts its own arithmetic. compute_clean_beam is never called, so
  reverting psf.py to the old buggy factor 2*sqrt(2 ln 2) still passes.

  test_psf_taylor_continuous_window operates entirely on a locally-defined `fm`
  list and asserts 0 < w_edge < 1. It imports nothing from psf.py and would
  pass against an empty library.

PRINCIPLE
---------
A test must exercise the library and compare against a value derived
INDEPENDENTLY of the library -- a physical constant, a closed-form result, or a
separately computed transform. A test that recomputes the implementation and
compares to itself measures nothing.

Every assertion below reads a value out of clean_sar and checks it against
physics. audit/claude_code_opus_5/a15_regression_bite.py proves each one fails when the
corresponding historical bug is reintroduced.
"""
import glob

import numpy as np
import pytest

from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.sicd_handler import SICDHandler


def _sicd():
    for pat in ("/home/feildaw/data/*.nitf",
                "/home/feildaw/diffpfa/workspace/output/*.nitf"):
        files = sorted(glob.glob(pat))
        if files:
            return files[0]
    pytest.skip("no test SICD available")


@pytest.fixture(scope="module")
def handler():
    return SICDHandler(_sicd())


def _oversampled_config(bw=1.5, oversample=16.0, wgt="UNIFORM"):
    """
    A config whose sample spacing finely oversamples the resolution cell, so the
    returned PSF array resolves its own sidelobe structure. SCP is placed at
    (0,0) so evaluating there gives theta = 0 exactly and the row/col cuts are
    the pure 1-D patterns.
    """
    ss = 1.0 / (bw * oversample)
    return CleanPhysicsConfig(
        row_ss=ss, col_ss=ss, row_bw=bw, col_bw=bw,
        row_wid=0.886 / bw, col_wid=0.886 / bw,
        scp_slant_range=600_000.0, scp_row=0.0, scp_col=0.0,
        row_wgt=wgt, col_wgt=wgt,
    )


# ---------------------------------------------------------------------------
# 1. Slant range is read from SCPCOA, not hardcoded
# ---------------------------------------------------------------------------
def test_slant_range_comes_from_scpcoa(handler):
    """
    Guards the earlier audit's Finding 1 (a hardcoded 10 km reference range).
    Compares the library's value against an independent read of the SICD XML.
    """
    cfg = CleanPhysicsConfig.from_sicd_handler(handler)
    xml_value = float(handler.xh.load("./{*}SCPCOA/{*}SlantRange"))

    assert cfg.scp_slant_range == pytest.approx(xml_value, rel=1e-9), (
        f"config.scp_slant_range={cfg.scp_slant_range} does not match "
        f"SCPCOA/SlantRange={xml_value} from the file"
    )
    assert cfg.scp_slant_range > 100_000.0, (
        f"slant range {cfg.scp_slant_range} m is implausible for a spaceborne "
        f"collect; a hardcoded constant has likely been reintroduced"
    )


# ---------------------------------------------------------------------------
# 2. The RETURNED Gaussian beam has the declared half-power width
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bw", [0.6, 1.5, 3.0])
def test_gaussian_beam_halfpower_width_matches_imprespwid(bw):
    """
    Guards the earlier audit's Finding 2 (sigma using the 2*sqrt(2 ln 2)
    convention, giving a beam sqrt(2) too narrow).

    Recovers sigma from the beam array itself. For beam = exp(-0.5 (u/sigma)^2)
    sampled at spacing ss, the ratio of two adjacent samples about the peak is
    exp(-0.5 (ss/sigma)^2), so

        sigma = ss / sqrt(-2 ln(beam[c+1]/beam[c]))

    is exact and needs no curve fitting. The half-power (-3 dB) width of an
    amplitude Gaussian is then 2*sigma*sqrt(ln 2), which must equal ImpRespWid.
    """
    cfg = _oversampled_config(bw=bw)
    gen = PSFGenerator(cfg)

    n = 129
    c = n // 2
    beam = np.abs(gen.compute_clean_beam(0, 0, psf_size=n, beam_type="gaussian"))

    assert beam[c, c] == pytest.approx(1.0, abs=1e-6), "beam peak is not 1.0"

    for axis, ss, wid in (("row", cfg.row_ss, cfg.row_wid),
                          ("col", cfg.col_ss, cfg.col_wid)):
        nxt = beam[c + 1, c] if axis == "row" else beam[c, c + 1]
        sigma = ss / np.sqrt(-2.0 * np.log(nxt))
        width_3db = 2.0 * sigma * np.sqrt(np.log(2.0))
        assert width_3db == pytest.approx(wid, rel=2e-3), (
            f"{axis}: restoring beam -3 dB width = {width_3db:.6f} m, "
            f"ImpRespWid = {wid:.6f} m (ratio {width_3db / wid:.4f}). "
            f"A ratio near 0.707 means the FWHM convention regressed."
        )


# ---------------------------------------------------------------------------
# 3. The RETURNED dirty PSF matches an independently computed band-limited IPR
# ---------------------------------------------------------------------------
def test_dirty_psf_matches_independent_fft_ipr():
    """
    Ground truth is the inverse FT of a rect aperture of width ImpRespBW,
    computed by FFT -- a different construction from psf.py's analytic sinc
    series, so agreement is meaningful.
    """
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

    assert np.max(np.abs(cut - expected)) < 1e-3, (
        f"analytic PSF row cut departs from the band-limited IPR by "
        f"{np.max(np.abs(cut - expected)):.3e}"
    )


# ---------------------------------------------------------------------------
# 4. The RETURNED weighted PSF has the sidelobe level its window promises
# ---------------------------------------------------------------------------
def _peak_sidelobe_db(cut):
    """Peak sidelobe level in dB relative to the mainlobe, from a 1-D cut."""
    a = np.abs(cut) / np.max(np.abs(cut))
    c = int(np.argmax(a))
    i = c
    while i + 1 < len(a) and a[i + 1] < a[i]:   # walk down to the first null
        i += 1
    return 20.0 * np.log10(np.max(a[i:]))


@pytest.mark.parametrize("wgt,expected_db,tol", [
    ("UNIFORM", -13.26, 0.6),   # first sinc sidelobe
    ("TAYLOR", -30.0, 1.5),     # nbar=4, SLL=-30 dB by construction
    ("HAMMING", -42.7, 3.0),
    ("HANN", -31.5, 3.0),
])
def test_psf_sidelobe_level_matches_window(wgt, expected_db, tol):
    """
    Replaces test_psf_taylor_continuous_window, which asserted 0 < w_edge < 1 on
    a local list. This calls compute_psf and measures the sidelobe level of the
    array it returns against the window's defining specification.
    """
    cfg = _oversampled_config(bw=1.5, oversample=24.0, wgt=wgt)
    gen = PSFGenerator(cfg)

    n = 401
    c = n // 2
    psf = gen.compute_psf(0, 0, psf_size=n, window_row=wgt, window_col=wgt)
    psl = _peak_sidelobe_db(np.real(psf[:, c]))

    assert psl == pytest.approx(expected_db, abs=tol), (
        f"{wgt}: measured peak sidelobe {psl:.2f} dB, expected "
        f"{expected_db:.2f} +/- {tol} dB"
    )


def test_taylor_sidelobes_below_uniform():
    """Taylor must actually buy sidelobe suppression over an untapered aperture."""
    cfg = _oversampled_config(bw=1.5, oversample=24.0)
    gen = PSFGenerator(cfg)
    n, c = 401, 200
    psl_u = _peak_sidelobe_db(np.real(gen.compute_psf(0, 0, psf_size=n,
                                                     window_row="UNIFORM",
                                                     window_col="UNIFORM")[:, c]))
    psl_t = _peak_sidelobe_db(np.real(gen.compute_psf(0, 0, psf_size=n,
                                                     window_row="TAYLOR",
                                                     window_col="TAYLOR")[:, c]))
    assert psl_t < psl_u - 10.0, (
        f"Taylor sidelobes ({psl_t:.2f} dB) not meaningfully below uniform "
        f"({psl_u:.2f} dB)"
    )


# ---------------------------------------------------------------------------
# 5. PSF peak must be unity for every window (photometric requirement, F1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_psf_peak_is_unity(wgt):
    """
    Hogbom subtracts gain*amp*PSF but books gain*amp as extracted flux. Those
    agree only if PSF(0,0) == 1, so this is photometry, not convention.
    """
    gen = PSFGenerator(_oversampled_config(wgt=wgt))
    psf = gen.compute_psf(0, 0, psf_size=65)
    assert np.abs(psf[32, 32]) == pytest.approx(1.0, abs=1e-6), (
        f"{wgt}: PSF centre = {psf[32, 32]!r}, expected 1.0"
    )
