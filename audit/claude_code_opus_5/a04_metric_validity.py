"""
audit/claude_code_opus_5/a04_metric_validity.py

Two questions:

  (1) GROUND TRUTH RECOVERY. Build a synthetic scene using the library's OWN
      forward model (point targets convolved with PSFGenerator's analytic PSF).
      Run CLEAN. Does it recover the true target positions and amplitudes?
      This is the test the repo's test-suite never performs.

  (2) IS "suppression_db" A MEANINGFUL QUALITY METRIC? Run CLEAN on the same
      real data with deliberately WRONG physics (wrong bandwidth, wrong sample
      spacing, delta-function PSF, zeroed PSF) and compare the reported
      suppression_db. If a wrong PSF scores the same as the right one, the
      headline metric cannot certify correctness.
"""
import sys
import copy
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
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
gen = PSFGenerator(cfg)

# =====================================================================
log("=" * 88)
log("(1) SYNTHETIC GROUND-TRUTH RECOVERY  (forward model == library's own PSF)")
log("=" * 88)

N = 128
rng = np.random.default_rng(0)
truth = np.zeros((N, N), dtype=np.complex64)
targets = [(30, 40, 1.0 + 0.0j), (64, 64, 0.6 - 0.3j), (95, 88, 0.35 + 0.2j),
           (50, 100, 0.25 + 0.1j), (100, 30, 0.15 - 0.05j)]
for (r, c, a) in targets:
    truth[r, c] = a

# Convolve with the spatially-varying analytic PSF, exactly as the algorithm models it
PS = 65
kh = PS // 2
dirty = np.zeros((N, N), dtype=np.complex64)
for (r, c, a) in targets:
    psf = gen.compute_psf(r, c, psf_size=PS)
    r_min, r_max = max(0, r - kh), min(N, r + kh + 1)
    c_min, c_max = max(0, c - kh), min(N, c + kh + 1)
    dirty[r_min:r_max, c_min:c_max] += a * psf[
        kh - (r - r_min): kh + (r_max - r), kh - (c - c_min): kh + (c_max - c)
    ]

log(f"  synthetic scene {N}x{N}, {len(targets)} point targets, no noise")
log(f"  forward model uses the SAME analytic PSF the algorithm will use "
    f"(best possible case)")

res = run_hogbom_clean(
    dirty_image=dirty, config=cfg, backend="pytorch", beam_type="gaussian",
    psf_size=PS, gain=0.1, threshold=0.001, max_iters=4000, verbose=False,
)
comp = res.components_map
log(f"  iterations = {res.iterations}, suppression = {res.suppression_db:.2f} dB")
log(f"  residual peak / dirty peak = "
    f"{np.max(np.abs(res.residual_image))/np.max(np.abs(dirty)):.3e}")
log()
log(f"  {'true (r,c)':>14} {'true amp':>18} {'recovered amp (3x3)':>24} {'|err|/|true|':>14}")
tot_recovered = 0.0
for (r, c, a) in targets:
    got = comp[r - 1:r + 2, c - 1:c + 2].sum()
    tot_recovered += abs(got)
    log(f"  {str((r,c)):>14} {a!s:>18} {f'{got.real:+.4f}{got.imag:+.4f}j':>24} "
        f"{abs(got-a)/abs(a):>14.4f}")
log(f"  total |components| in scene = {np.abs(comp).sum():.4f}; "
    f"total at true target sites = {tot_recovered:.4f}  "
    f"-> {100*tot_recovered/max(np.abs(comp).sum(),1e-12):.1f}% of CLEAN flux "
    f"landed on true targets")
log(f"  number of distinct pixels receiving components = {np.count_nonzero(comp)}")

# =====================================================================
log()
log("=" * 88)
log("(2) DOES suppression_db DISTINGUISH CORRECT PHYSICS FROM WRONG PHYSICS?")
log("=" * 88)

chip, _ = h.read_chip(2000, 4000, 2256, 4256)
log(f"  Real data chip 256x256 from {FP.split('/')[-1]} at (2000,4000)")
log()


def variant(name, c2):
    r = run_hogbom_clean(
        dirty_image=chip, config=c2, backend="pytorch", beam_type="gaussian",
        psf_size=65, gain=0.1, threshold=0.02, max_iters=1500, verbose=False,
    )
    # An honest quality metric: does the CLEAN image have less total energy in
    # the sidelobe skirts around the brightest target than the dirty image?
    return r


base = copy.deepcopy(cfg.__dict__)

variants = [
    ("CORRECT physics (metadata)", dict()),
    ("row_bw, col_bw x 2  (WRONG)", dict(row_bw=cfg.row_bw * 2, col_bw=cfg.col_bw * 2)),
    ("row_bw, col_bw / 4  (WRONG)", dict(row_bw=cfg.row_bw / 4, col_bw=cfg.col_bw / 4)),
    ("sample spacing x 10 (WRONG)", dict(row_ss=cfg.row_ss * 10, col_ss=cfg.col_ss * 10)),
    ("bw -> 1e6 (PSF = delta)", dict(row_bw=1e6, col_bw=1e6)),
    ("slant range 1 m (absurd)", dict(scp_slant_range=1.0)),
    ("weighting -> HAMMING (wrong)", dict(row_wgt="HAMMING", col_wgt="HAMMING")),
]

log(f"  {'variant':<32} {'iters':>7} {'suppression_db':>16} {'residual L2/dirty L2':>22}")
log(f"  {'-'*32} {'-'*7} {'-'*16} {'-'*22}")
dirty_l2 = np.linalg.norm(chip)
rows = []
for nm, over in variants:
    d = dict(base)
    d.update(over)
    c2 = CleanPhysicsConfig(**d)
    r = variant(nm, c2)
    rl2 = np.linalg.norm(r.residual_image) / dirty_l2
    rows.append((nm, r.iterations, r.suppression_db, rl2))
    log(f"  {nm:<32} {r.iterations:>7d} {r.suppression_db:>16.2f} {rl2:>22.4f}")

sup = [x[2] for x in rows]
log()
log(f"  suppression_db across ALL variants: min={min(sup):.2f} max={max(sup):.2f} "
    f"spread={max(sup)-min(sup):.2f} dB")
log(f"  -> correct physics scored {rows[0][2]:.2f} dB; the best WRONG variant scored "
    f"{max(s for _,_,s,_ in rows[1:]):.2f} dB")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a04_metric_validity.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
