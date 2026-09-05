"""
audit/claude_code_opus_5/a13_diffpfa_irw_check.py

SICD DIDD 1.5 (references/NGA.STND.0024-1), p.40/42:
    ImpRespWid  R  DBL  "Half power impulse response width"
DIDD p.174/175:
    Rg_IRW = k_RG / Krg_IRBW    "Impulse response broadening factor k is
                                 associated with weighting function w(s).
                                 For uniform weighting, k = 0.886."

So the normative relation is  ImpRespWid = k(window) / ImpRespBW.

The DiffPFA products declare k = 1.0000 exactly. Two possibilities:
    (a) the declared width is wrong (should be 0.886/BW for a uniform aperture)
    (b) the aperture really is tapered such that k = 1.0

This decides between them by measuring the TRUE half-power IRW directly from
the image data -- inverse-transforming the measured k-space support profile,
which needs no isolated point target -- and comparing against both candidates.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


PAIRS = [
    ("RAW Umbra", "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"),
    ("DiffPFA  ", "/home/feildaw/diffpfa/workspace/output/2023-11-14-03-38-20_UMBRA-04_SICDU_X_X.nitf"),
    ("RAW Umbra", "/home/feildaw/data/2023-09-13-21-18-21_UMBRA-06_SICD.nitf"),
    ("DiffPFA  ", "/home/feildaw/diffpfa/workspace/output/2023-09-13-21-18-21_UMBRA-06_SICDU_X_X.nitf"),
]


def half_power_width(x, amp):
    a = np.abs(amp) / np.max(np.abs(amp))
    t = 1.0 / np.sqrt(2.0)
    c = int(np.argmax(a))
    r = c
    while r + 1 < len(a) and a[r + 1] > t:
        r += 1
    xr = np.interp(-t, [-a[r], -a[r + 1]], [x[r], x[r + 1]])
    l = c
    while l - 1 >= 0 and a[l - 1] > t:
        l -= 1
    xl = np.interp(-t, [-a[l], -a[l - 1]], [x[l], x[l - 1]])
    return xr - xl


log("=" * 92)
log("MEASURED half-power IRW vs. DECLARED ImpRespWid")
log("=" * 92)
log(f"  {'product':<10} {'file':<34} {'declared k':>11} {'measured k':>11} {'verdict':>22}")
log(f"  {'-'*10} {'-'*34} {'-'*11} {'-'*11} {'-'*22}")

for tag, fp in PAIRS:
    h = SICDHandler(fp)
    N = 1024
    r0 = int(np.clip(h.scp_pixel[0] - N // 2, 0, h.num_rows - N))
    c0 = int(np.clip(h.scp_pixel[1] - N // 2, 0, h.num_cols - N))
    chip, _ = h.read_chip(r0, c0, r0 + N, c0 + N)

    # Measured k-space amplitude support profile, averaged over the orthogonal axis.
    # sqrt of the mean power spectrum = the aperture amplitude weighting |W(k)|.
    P = np.abs(np.fft.fft2(chip)) ** 2
    Wr = np.sqrt(np.fft.fftshift(P.mean(axis=1)))
    kr = np.fft.fftshift(np.fft.fftfreq(N, d=h.row_ss))

    # Zero-pad the measured aperture and transform -> the true IPR of this product.
    # Wr is already fftshifted (DC-centred), so drop it straight into the centre
    # of the padded array and ifftshift once before transforming.
    M = 1 << 16
    Wc = np.zeros(M)
    lo = M // 2 - N // 2
    Wc[lo:lo + N] = Wr
    ipr = np.fft.fftshift(np.abs(np.fft.ifft(np.fft.ifftshift(Wc))))
    xs = (np.arange(M) - M // 2) * (1.0 / (M * (kr[1] - kr[0])))

    irw_meas = half_power_width(xs, ipr)
    k_meas = irw_meas * h.row_bw
    k_decl = h.row_wid * h.row_bw

    if abs(k_meas - k_decl) / k_decl < 0.05:
        verdict = "declared OK"
    else:
        verdict = f"declared {k_decl/k_meas:.3f}x too wide"
    log(f"  {tag:<10} {fp.split('/')[-1][:33]:<34} {k_decl:>11.4f} {k_meas:>11.4f} {verdict:>22}")

log()
log("  Reference broadening factors k (DIDD p.174):")
log("     uniform 0.886 | Taylor(4,-30dB) 1.125 | Hamming 1.303 | Hann 1.441")
log()
log("  Note: measured k carries a small positive bias because the k-space support")
log("  is estimated from clutter rather than a calibrated point target; treat")
log("  agreement to a few percent as agreement.")

# ---- what the declared value should be, per the standard
log()
log("=" * 92)
log("IF THE APERTURE IS UNIFORM, WHAT SHOULD ImpRespWid BE?")
log("=" * 92)
for tag, fp in PAIRS:
    if not tag.startswith("DiffPFA"):
        continue
    h = SICDHandler(fp)
    log(f"  {fp.split('/')[-1]}")
    for ax, bw, wid in (("Row", h.row_bw, h.row_wid), ("Col", h.col_bw, h.col_wid)):
        log(f"     {ax}: ImpRespBW={bw:.6f}  declared ImpRespWid={wid:.6f} "
            f"(= 1.000/BW)   standard-conformant uniform value = {0.886/bw:.6f} (= 0.886/BW)")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a13_diffpfa_irw_check.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
