"""
audit/claude_code_opus_5/a11_normalization_and_wgt.py

Two follow-ups.

(A) Which normalisation is CORRECT, not merely consistent?
    Hogbom CLEAN subtracts gain*amp*PSF at the peak but records gain*amp in the
    components map. Those two only agree if PSF(0,0) == 1. If PSF(0,0) = 0.2916
    (Hamming, unnormalised), the loop removes 0.2916*comp from the image while
    claiming comp of flux -- and the restoring beam, whose peak IS 1.0, adds the
    full comp back. Test with synthetic ground truth: known targets, known
    amplitudes, Hamming weighting, both backends.

(B) Can the weighting be recovered when WgtType is absent?
    Requiring WgtType would reject 11 of the 12 datasets. Check whether
    Grid/Row/WgtFunct is present, and whether ImpRespWid*ImpRespBW (a
    window-specific constant) identifies the window on its own.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.algorithm import run_hogbom_clean

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


FP = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
h = SICDHandler(FP)
base = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))

# ============================================================ (A) photometry
log("=" * 84)
log("(A) WHICH NORMALISATION IS CORRECT? Synthetic ground truth, HAMMING weighting")
log("=" * 84)

d = dict(base.__dict__)
d.update(row_wgt="HAMMING", col_wgt="HAMMING")
cfg = CleanPhysicsConfig(**d)
gen = PSFGenerator(cfg)

N, PS = 128, 65
kh = PS // 2
targets = [(40, 45, 1.0 + 0.0j), (64, 64, 0.6 - 0.3j), (90, 85, 0.35 + 0.2j)]

# Forward-model the scene with a PHYSICALLY normalised PSF (peak 1.0 at the
# target pixel), which is what a real point target looks like in the image.
dirty = np.zeros((N, N), dtype=np.complex64)
for (r, c, a) in targets:
    psf = gen.compute_psf(r, c, psf_size=PS)   # psf.py: normalised to 1.0
    dirty[r - kh:r + kh + 1, c - kh:c + kh + 1] += a * psf

log(f"  synthetic scene: {len(targets)} point targets, Hamming weighting, no noise")
log(f"  each target's peak pixel equals its true complex amplitude")
log()
log(f"  {'backend':<10} {'target':>10} {'true amp':>16} {'recovered':>22} {'ratio':>9}")
log(f"  {'-'*10} {'-'*10} {'-'*16} {'-'*22} {'-'*9}")

for be in ["pytorch", "cuda"]:
    r = run_hogbom_clean(dirty_image=dirty, config=cfg, backend=be,
                         beam_type="gaussian", psf_size=PS, gain=0.1,
                         threshold=0.001, max_iters=6000, verbose=False)
    comp = r.components_map
    for (tr, tc, a) in targets:
        got = comp[tr - 2:tr + 3, tc - 2:tc + 3].sum()
        log(f"  {be:<10} {str((tr,tc)):>10} {a!s:>16} "
            f"{f'{got.real:+.4f}{got.imag:+.4f}j':>22} {abs(got)/abs(a):>8.3f}x")
    tot = np.abs(comp).sum()
    true_tot = sum(abs(a) for _, _, a in targets)
    log(f"  {be:<10} {'TOTAL':>10} {true_tot:>16.4f} {tot:>22.4f} "
        f"{tot/true_tot:>8.3f}x   iters={r.iterations}")
    log()

log("  Interpretation: a ratio of 1.000x means the recovered scatterer amplitude")
log("  equals the truth. Whichever backend deviates has the wrong normalisation --")
log("  this is a correctness question, not a consistency question.")

# ============================================================ (B) weighting id
log()
log("=" * 84)
log("(B) RECOVERING THE WEIGHTING WHEN WgtType IS ABSENT")
log("=" * 84)

# window-specific broadening constant k, where ImpRespWid = k / ImpRespBW
K_TABLE = {"UNIFORM": 0.886, "TAYLOR(nbar4,-30dB)": 1.125, "HAMMING": 1.303, "HANN": 1.441}
log(f"  Window identification constant k = ImpRespWid * ImpRespBW:")
for k, v in K_TABLE.items():
    log(f"     {k:<24} k = {v}")
log()

import glob
files = sorted(glob.glob("/home/feildaw/data/*.nitf")) + \
        sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
log(f"  {'file':<44} {'WgtType':>9} {'WgtFunct':>9} {'k_row':>7} {'k_col':>7} {'implied':>9}")
log(f"  {'-'*44} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*9}")
for fp in files:
    hh = SICDHandler(fp)
    wf = hh.xmltree.find("{*}Grid/{*}Row/{*}WgtFunct")
    n_wf = len(wf) if wf is not None else 0
    kr = hh.row_wid * hh.row_bw
    kc = hh.col_wid * hh.col_bw
    implied = min(K_TABLE, key=lambda w: abs(K_TABLE[w] - kr)).split("(")[0]
    log(f"  {fp.split('/')[-1][:43]:<44} {str(hh.row_wgt_name):>9} "
        f"{(str(n_wf)+' pts') if n_wf else 'absent':>9} {kr:>7.4f} {kc:>7.4f} {implied:>9}")

log()
log("  A cross-check (declared window vs. k implied by ImpRespWid*ImpRespBW) is")
log("  strictly stronger than requiring WgtType, and works on files that omit it.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a11_normalization_and_wgt.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
