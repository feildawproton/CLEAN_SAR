"""
audit/claude_code_opus_5/a01_psf_math.py

Validates the analytic IPR/PSF math in clean_sar/psf.py against independently
derived ground truth (numerical Fourier transform of the aperture weighting).

Checks:
  1. Taylor Fm coefficients hardcoded in psf.py vs. clean_sar.utils.taylor_window_1d
     and vs. scipy.signal.windows.taylor.
  2. Analytic 1D pattern vs. numerical inverse-FT of the windowed k-space support.
  3. Gaussian restoring beam -3 dB width vs. the declared ImpRespWid.
  4. Dirty PSF -3 dB width vs. the declared ImpRespWid.
  5. Imaginary content of the "complex" PSF.
"""
import numpy as np
import sys

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.utils import taylor_window_1d
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


# ---------------------------------------------------------------- 1. Taylor Fm
log("=" * 78)
log("1. TAYLOR COEFFICIENTS: psf.py hardcoded vs. utils.taylor_window_1d vs scipy")
log("=" * 78)

FM_HARDCODED = [0.29265601, -0.01578375, 0.00218104]


def taylor_fm(nbar=4, sll=-30.0):
    """Fm coefficients using the same closed form as clean_sar.utils."""
    eta = 10.0 ** (-sll / 20.0)
    a = np.arccosh(eta) / np.pi
    sigma2 = (nbar**2) / (a**2 + (nbar - 0.5) ** 2)
    fm = []
    for mi in range(1, nbar):
        num, den = 1.0, 1.0
        for n in range(1, nbar):
            num *= 1.0 - (mi**2) / (sigma2 * (a**2 + (n - 0.5) ** 2))
            if n != mi:
                den *= 1.0 - (mi**2) / (n**2)
        fm.append(((-1.0) ** (mi + 1) * num) / (2.0 * den))
    return np.array(fm)


for nbar, sll in [(4, -30.0), (4, -35.0), (5, -30.0), (4, -25.0), (3, -30.0)]:
    fm = taylor_fm(nbar, sll)
    tag = ""
    if len(fm) == 3 and np.allclose(fm, FM_HARDCODED, atol=2e-6):
        tag = "   <== MATCHES psf.py hardcoded values"
    log(f"  nbar={nbar}, SLL={sll:6.1f} dB -> Fm = {np.array2string(fm, precision=8)}{tag}")

log()
log(f"  psf.py hardcoded  Fm = {FM_HARDCODED}")
fm_ref = taylor_fm(4, -30.0)
log(f"  utils(nbar=4,-30) Fm = {np.array2string(fm_ref, precision=8)}")
log(f"  max abs diff         = {np.max(np.abs(fm_ref - np.array(FM_HARDCODED))):.3e}")

try:
    from scipy.signal.windows import taylor as scipy_taylor

    # Recover Fm from scipy's window by projecting onto the cosine series
    N = 4096
    w = scipy_taylor(N, nbar=4, sll=30, norm=False)
    n = (np.arange(N) - (N - 1) / 2.0) / N  # in [-0.5, 0.5]
    fm_proj = [2.0 * np.mean(w * np.cos(2 * np.pi * m * n)) / np.mean(w) / 2.0 for m in (1, 2, 3)]
    log(f"  scipy taylor proj Fm = {np.array2string(np.array(fm_proj), precision=8)}")
except Exception as e:
    log(f"  (scipy unavailable: {e})")


# --------------------------------------------- 2. Analytic pattern vs numeric FT
log()
log("=" * 78)
log("2. ANALYTIC 1D PATTERN vs. NUMERICAL INVERSE-FT OF WINDOWED APERTURE")
log("=" * 78)


def analytic_pattern(x, bw, wgt):
    """Verbatim reimplementation of psf.py::_eval_1d_pattern."""
    if wgt == "UNIFORM":
        return np.sinc(bw * x)
    if wgt == "HAMMING":
        return 0.54 * np.sinc(bw * x) + 0.23 * np.sinc(bw * x - 1.0) + 0.23 * np.sinc(bw * x + 1.0)
    if wgt == "HANN":
        return 0.5 * np.sinc(bw * x) + 0.25 * np.sinc(bw * x - 1.0) + 0.25 * np.sinc(bw * x + 1.0)
    if wgt == "TAYLOR":
        pat = np.sinc(bw * x)
        for m, c in enumerate(FM_HARDCODED, start=1):
            pat = pat + c * (np.sinc(bw * x - m) + np.sinc(bw * x + m))
        return pat
    raise ValueError(wgt)


def numeric_ipr(x, bw, wgt, nk=40001):
    """Ground truth: IPR(x) = Int_{-bw/2}^{bw/2} W(k) exp(j2 pi k x) dk, computed by quadrature."""
    k = np.linspace(-bw / 2.0, bw / 2.0, nk)
    kn = k / bw  # normalized to [-0.5, 0.5]
    if wgt == "UNIFORM":
        W = np.ones_like(kn)
    elif wgt == "HAMMING":
        W = 0.54 + 0.46 * np.cos(2 * np.pi * kn)
    elif wgt == "HANN":
        W = 0.50 + 0.50 * np.cos(2 * np.pi * kn)
    elif wgt == "TAYLOR":
        W = np.ones_like(kn)
        for m, c in enumerate(FM_HARDCODED, start=1):
            W = W + 2.0 * c * np.cos(2 * np.pi * m * kn)
    else:
        raise ValueError(wgt)
    phase = np.exp(2j * np.pi * np.outer(x, k))
    val = np.trapezoid(W[None, :] * phase, k, axis=1)
    return val.real


BW = 1.3  # cycles / m, representative
xs = np.linspace(-8.0 / BW, 8.0 / BW, 401)
for wgt in ["UNIFORM", "HAMMING", "HANN", "TAYLOR"]:
    a = analytic_pattern(xs, BW, wgt)
    g = numeric_ipr(xs, BW, wgt)
    g = g / g[np.argmin(np.abs(xs))] * a[np.argmin(np.abs(xs))]  # match at x=0
    err = np.max(np.abs(a - g)) / np.max(np.abs(g))
    log(f"  {wgt:8s}: max rel error analytic vs numeric-FT = {err:.3e}   "
        f"[{'OK' if err < 1e-4 else 'MISMATCH'}]")
    log(f"            analytic peak value at x=0        = {a[np.argmin(np.abs(xs))]:.6f}")


# ------------------------------------------------------ 3/4. -3 dB width checks
log()
log("=" * 78)
log("3. RESOLUTION WIDTHS: analytic sinc IPR and Gaussian beam vs ImpRespWid")
log("=" * 78)


def half_power_width(x, amp):
    """Full width where |amp|^2 drops to half its peak, by interpolation."""
    a = np.abs(amp) / np.max(np.abs(amp))
    target = 1.0 / np.sqrt(2.0)
    c = np.argmax(a)
    r = c
    while r + 1 < len(a) and a[r + 1] > target:
        r += 1
    xr = np.interp(-target, [-a[r], -a[r + 1]], [x[r], x[r + 1]])
    l = c
    while l - 1 >= 0 and a[l - 1] > target:
        l -= 1
    xl = np.interp(-target, [-a[l], -a[l - 1]], [x[l], x[l - 1]])
    return xr - xl


xs_fine = np.linspace(-4.0 / BW, 4.0 / BW, 400001)
log(f"  Using ImpRespBW = {BW} cycles/m")
for wgt, k_expect in [("UNIFORM", 0.886), ("HAMMING", 1.30), ("HANN", 1.44), ("TAYLOR", 1.0)]:
    pat = analytic_pattern(xs_fine, BW, wgt)
    w = half_power_width(xs_fine, pat)
    log(f"  {wgt:8s}: IPR -3dB width = {w:.5f} m  ->  width*BW = {w*BW:.4f} "
        f"(textbook ~{k_expect})")

log()
log("  Gaussian restoring beam, as coded: beam = exp(-0.5 * (u/sigma)^2),")
log("                                    sigma = ImpRespWid / (2*sqrt(ln 2))")
for wid in [0.886 / BW, 1.30 / BW]:
    sigma = wid / (2.0 * np.sqrt(np.log(2.0)))
    beam = np.exp(-0.5 * (xs_fine / sigma) ** 2)
    w = half_power_width(xs_fine, beam)
    log(f"    ImpRespWid = {wid:.5f} m -> beam -3dB width = {w:.5f} m   "
        f"ratio = {w/wid:.6f}  [{'OK' if abs(w/wid-1) < 1e-3 else 'MISMATCH'}]")

log()
log("  For reference, if the code had used sigma = Wid/(2*sqrt(2 ln 2)) (true FWHM):")
for wid in [0.886 / BW]:
    sigma = wid / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    beam = np.exp(-0.5 * (xs_fine / sigma) ** 2)
    w = half_power_width(xs_fine, beam)
    log(f"    ImpRespWid = {wid:.5f} m -> beam -3dB width = {w:.5f} m   ratio = {w/wid:.6f}")


# ------------------------------------------------------------ 5. Complex or not
log()
log("=" * 78)
log("5. IS THE 'COMPLEX' PSF ACTUALLY COMPLEX?")
log("=" * 78)
cfg = CleanPhysicsConfig(
    row_ss=0.25, col_ss=0.25, row_bw=3.0, col_bw=3.0,
    row_wid=0.295, col_wid=0.295, scp_slant_range=500000.0,
    scp_row=1000.0, scp_col=1000.0, row_wgt="TAYLOR", col_wgt="TAYLOR",
)
gen = PSFGenerator(cfg)
for (r, c) in [(1000, 1000), (0, 0), (2000, 2000)]:
    psf = gen.compute_psf(r, c, psf_size=65)
    log(f"  PSF at ({r},{c}): dtype={psf.dtype}  max|imag| = {np.max(np.abs(psf.imag)):.3e}  "
        f"max|real| = {np.max(np.abs(psf.real)):.4f}")
beam = gen.compute_clean_beam(1000, 1000, psf_size=65, beam_type="gaussian")
log(f"  Gaussian beam:   dtype={beam.dtype}  max|imag| = {np.max(np.abs(beam.imag)):.3e}")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a01_psf_math.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
