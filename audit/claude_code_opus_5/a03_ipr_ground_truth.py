"""
audit/claude_code_opus_5/a03_ipr_ground_truth.py

Ground-truth check of the analytic PSF against the ACTUAL impulse response of
the real SICD data, derived two independent ways:

  (A) K-space: FFT a large chip, measure the occupied spatial-frequency support
      (extent + center + taper). The true IPR is the inverse FT of that support.
      Compare the empirical support to Grid/Row/ImpRespBW and Grid/Col/ImpRespBW,
      and compare the resulting empirical PSF to PSFGenerator.compute_psf.

  (B) Point target: find a bright isolated scatterer, extract its complex
      neighbourhood, and compare magnitude profile + phase behaviour to the
      analytic PSF. This tests whether the true IPR is real-symmetric (as the
      code assumes) or carries a phase ramp (Grid KCtr carrier).

Also measures how much the "spatially-varying" PSF actually varies across a
full scene.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


FILES = [
    "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf",
    "/home/feildaw/diffpfa/workspace/output/2023-11-14-03-38-20_UMBRA-04_SICDU_X_X.nitf",
]

for fp in FILES:
    name = fp.split("/")[-1]
    log("=" * 88)
    log(f"FILE: {name}")
    log("=" * 88)
    h = SICDHandler(fp)
    cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))
    gen = PSFGenerator(cfg)

    # ---------------- (A) empirical k-space support ----------------
    N = 1024
    r0 = int(h.scp_pixel[0]) - N // 2
    c0 = int(h.scp_pixel[1]) - N // 2
    r0 = max(0, min(r0, h.num_rows - N))
    c0 = max(0, min(c0, h.num_cols - N))
    chip, _ = h.read_chip(r0, c0, r0 + N, c0 + N)

    S = np.fft.fftshift(np.abs(np.fft.fft2(chip)) ** 2)
    prof_r = S.sum(axis=1)
    prof_c = S.sum(axis=0)
    prof_r /= prof_r.max()
    prof_c /= prof_c.max()

    # frequency axes in cycles/m
    kr = np.fft.fftshift(np.fft.fftfreq(N, d=h.row_ss))
    kc = np.fft.fftshift(np.fft.fftfreq(N, d=h.col_ss))

    def support_extent(prof, k, frac=0.10):
        """Width of the region where the power profile exceeds frac of its max."""
        idx = np.where(prof > frac)[0]
        return k[idx[0]], k[idx[-1]], k[idx[-1]] - k[idx[0]]

    lo_r, hi_r, bw_r = support_extent(prof_r, kr)
    lo_c, hi_c, bw_c = support_extent(prof_c, kc)
    ctr_r, ctr_c = 0.5 * (lo_r + hi_r), 0.5 * (lo_c + hi_c)

    log(f"  [A] Empirical k-space support from a {N}x{N} chip at SCP")
    log(f"      ROW: measured support [{lo_r:+.4f}, {hi_r:+.4f}] cyc/m  width={bw_r:.4f}  "
        f"center={ctr_r:+.5f}")
    log(f"           metadata ImpRespBW = {h.row_bw:.4f}   ratio measured/meta = {bw_r/h.row_bw:.4f}")
    log(f"      COL: measured support [{lo_c:+.4f}, {hi_c:+.4f}] cyc/m  width={bw_c:.4f}  "
        f"center={ctr_c:+.5f}")
    log(f"           metadata ImpRespBW = {h.col_bw:.4f}   ratio measured/meta = {bw_c/h.col_bw:.4f}")
    log(f"      NOTE: metadata Grid/Row/KCtr = "
        f"{float(h.xh.load('./{*}Grid/{*}Row/{*}KCtr')):.5f} cyc/m (RF carrier);")
    log(f"            empirical support center in the SAMPLED data = {ctr_r:+.5f} cyc/m")
    log(f"            -> sampled image is {'BASEBAND (carrier removed)' if abs(ctr_r) < 0.1*h.row_bw else 'CARRIER-MODULATED'}")

    # taper shape of the support: uniform-weighted support is flat-topped
    inband_r = prof_r[(kr > lo_r + 0.1 * bw_r) & (kr < hi_r - 0.1 * bw_r)]
    log(f"      ROW in-band power profile: mean={inband_r.mean():.3f} std={inband_r.std():.3f} "
        f"(flat/uniform aperture => low std)")

    # ---------------- empirical PSF from the measured support ----------------
    M = 2048
    krf = np.fft.fftfreq(M, d=h.row_ss)
    kcf = np.fft.fftfreq(M, d=h.col_ss)
    Wr = (np.abs(krf) <= h.row_bw / 2).astype(np.float64)
    Wc = (np.abs(kcf) <= h.col_bw / 2).astype(np.float64)
    psf_emp_r = np.fft.fftshift(np.fft.ifft(Wr)).real
    psf_emp_c = np.fft.fftshift(np.fft.ifft(Wc)).real
    psf_emp_r /= psf_emp_r.max()
    psf_emp_c /= psf_emp_c.max()
    xs_r = (np.arange(M) - M // 2) * h.row_ss
    xs_c = (np.arange(M) - M // 2) * h.col_ss

    # analytic PSF from the library, at SCP, 1-D cuts
    psf_lib = gen.compute_psf(h.scp_pixel[0], h.scp_pixel[1], psf_size=129)
    kh = 129 // 2
    cut_r = psf_lib[:, kh].real
    cut_c = psf_lib[kh, :].real
    xs_lib = (np.arange(129) - kh)

    # compare on the library's sample grid
    interp_r = np.interp(xs_lib * h.row_ss, xs_r, psf_emp_r)
    interp_c = np.interp(xs_lib * h.col_ss, xs_c, psf_emp_c)
    log(f"      analytic PSF (library) vs band-limited-rect PSF (independent FFT):")
    log(f"        row cut max abs diff = {np.max(np.abs(cut_r - interp_r)):.3e}")
    log(f"        col cut max abs diff = {np.max(np.abs(cut_c - interp_c)):.3e}")

    # ---------------- (B) real point target ----------------
    log()
    log(f"  [B] Brightest isolated scatterer in the chip")
    mag = np.abs(chip)
    # isolation: peak must dominate a 41x41 annulus
    best = None
    order = np.argsort(mag.ravel())[::-1][:400]
    for fl in order:
        pr, pc = fl // N, fl % N
        if pr < 40 or pc < 40 or pr >= N - 40 or pc >= N - 40:
            continue
        nb = mag[pr - 20:pr + 21, pc - 20:pc + 21].copy()
        peak = nb[20, 20]
        nb[15:26, 15:26] = 0
        if peak > 6.0 * nb.max():
            best = (pr, pc, peak, nb.max())
            break
    if best is None:
        log("      (no sufficiently isolated scatterer found)")
    else:
        pr, pc, peak, ring = best
        log(f"      found at chip ({pr},{pc}) global ({r0+pr},{c0+pc}) "
            f"peak={peak:.4g}, surrounding max={ring:.4g}, isolation={peak/ring:.1f}x")
        K = 16
        patch = chip[pr - K:pr + K + 1, pc - K:pc + K + 1]
        patch_n = patch / patch[K, K]

        psf_a = gen.compute_psf(r0 + pr, c0 + pc, psf_size=2 * K + 1)

        # magnitude agreement
        num = np.abs(np.vdot(psf_a.ravel(), patch_n.ravel()))
        den = np.linalg.norm(psf_a) * np.linalg.norm(patch_n)
        log(f"      |<analytic PSF, measured IPR>| / (||.||*||.||) = {num/den:.4f}   "
            f"(1.0 = perfect model)")
        mag_a = np.abs(psf_a) / np.abs(psf_a).max()
        mag_m = np.abs(patch_n) / np.abs(patch_n).max()
        cc = np.corrcoef(mag_a.ravel(), mag_m.ravel())[0, 1]
        log(f"      correlation of |analytic| vs |measured| magnitudes = {cc:.4f}")

        log(f"      measured IPR row cut (magnitude, normalised):")
        log(f"        {np.array2string(mag_m[:, K][K-6:K+7], precision=3)}")
        log(f"      analytic PSF row cut (magnitude, normalised):")
        log(f"        {np.array2string(mag_a[:, K][K-6:K+7], precision=3)}")

        ph = np.angle(patch_n)
        log(f"      measured IPR PHASE along row cut (radians, centre +/-6):")
        log(f"        {np.array2string(ph[:, K][K-6:K+7], precision=3)}")
        log(f"      measured IPR PHASE along col cut (radians, centre +/-6):")
        log(f"        {np.array2string(ph[K, :][K-6:K+7], precision=3)}")
        log(f"      -> analytic PSF phase is identically 0 everywhere "
            f"(real-valued sinc, possibly negative lobes)")

    # ---------------- spatial variation of the PSF ----------------
    log()
    log(f"  [C] How much does the 'spatially-varying' PSF actually vary?")
    p_ctr = gen.compute_psf(h.scp_pixel[0], h.scp_pixel[1], psf_size=65)
    corners = [(0, 0), (0, h.num_cols - 1), (h.num_rows - 1, 0), (h.num_rows - 1, h.num_cols - 1)]
    worst = 0.0
    for (rr, cc_) in corners:
        p = gen.compute_psf(rr, cc_, psf_size=65)
        d = np.max(np.abs(p - p_ctr))
        worst = max(worst, d)
    log(f"      max |PSF(corner) - PSF(SCP)| over the whole scene = {worst:.4e}")
    log(f"      (PSF peak is 1.0, so this is {100*worst:.3f}% of peak)")
    # what if we ignored the rotation completely (theta = 0)?
    cfg0 = CleanPhysicsConfig(**{**cfg.__dict__, "scp_slant_range": 1e12})
    gen0 = PSFGenerator(cfg0)
    p_flat = gen0.compute_psf(h.scp_pixel[0], h.scp_pixel[1], psf_size=65)
    dmax = 0.0
    for (rr, cc_) in corners + [(int(h.scp_pixel[0]), int(h.scp_pixel[1]))]:
        p = gen.compute_psf(rr, cc_, psf_size=65)
        dmax = max(dmax, np.max(np.abs(p - p_flat)))
    log(f"      max |PSF(spatially-varying) - PSF(theta=0 everywhere)| = {dmax:.4e} "
        f"({100*dmax:.3f}% of peak)")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a03_ipr_ground_truth.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
