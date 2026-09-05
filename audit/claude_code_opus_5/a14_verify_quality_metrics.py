"""
audit/claude_code_opus_5/a14_verify_quality_metrics.py

Verifies audit/claude_code_opus_5/quality_metrics.py reproduces the F2 table from a10, then
demonstrates the multi-target variant and the pass/fail verdict.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
sys.path.insert(0, "/home/feildaw/CLEAN_SAR/audit")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean
from quality_metrics import ipr_quality, ipr_quality_multi, verdict

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


FP = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
CHIP = (1932, 3931, 2188, 4187)  # demo_clean.py UMBRA_CHIP
h = SICDHandler(FP)
chip, _ = h.read_chip(*CHIP)
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(CHIP[0], CHIP[1]))

VARIANTS = [
    ("CORRECT physics", dict()),
    ("delta PSF (bw=1e6)", dict(row_bw=1e6, col_bw=1e6)),
    ("bandwidth x2 (wrong)", dict(row_bw=cfg.row_bw * 2, col_bw=cfg.col_bw * 2)),
    ("bandwidth /2 (wrong)", dict(row_bw=cfg.row_bw / 2, col_bw=cfg.col_bw / 2)),
]

log("=" * 96)
log("quality_metrics.ipr_quality() vs. the a10 table it must reproduce")
log("=" * 96)
log(f"  {'variant':<22} {'reported':>9} {'islr_change':>12} {'mainlobe':>10} {'verdict':>28}")
log(f"  {'':<22} {'suppr dB':>9} {'dB':>12} {'preserv.':>10} {'':>28}")
log(f"  {'-'*22} {'-'*9} {'-'*12} {'-'*10} {'-'*28}")

base = dict(cfg.__dict__)
for nm, over in VARIANTS:
    d = dict(base)
    d.update(over)
    c2 = CleanPhysicsConfig(**d)
    r = run_hogbom_clean(dirty_image=chip, config=c2, backend="pytorch",
                         beam_type="gaussian", psf_size=65, gain=0.1,
                         threshold=0.02, max_iters=2500, verbose=False)
    m = ipr_quality(chip, r.clean_image,
                    cfg.row_wid, cfg.col_wid, cfg.row_ss, cfg.col_ss)
    log(f"  {nm:<22} {r.suppression_db:>9.2f} {m['islr_change_db']:>12.3f} "
        f"{m['mainlobe_preservation']:>10.3f} {verdict(m):>28}")

log()
log(f"  a10 reference values:  correct -4.455 / 0.988 | delta -5.064 / 1.506 | "
    f"x2 -5.556 / 2.703 | /2 +4.049 / 15.052")
log(f"  geometry used: mainlobe radius {ipr_quality(chip, chip, cfg.row_wid, cfg.col_wid, cfg.row_ss, cfg.col_ss)['mainlobe_radius_px']} px, "
    f"annulus outer 40 px")

# ---- multi-target variant
log()
log("=" * 96)
log("ipr_quality_multi(): averaged over the brightest isolated scatterers")
log("=" * 96)
for nm, over in VARIANTS[:2]:
    d = dict(base)
    d.update(over)
    c2 = CleanPhysicsConfig(**d)
    r = run_hogbom_clean(dirty_image=chip, config=c2, backend="pytorch",
                         beam_type="gaussian", psf_size=65, gain=0.1,
                         threshold=0.02, max_iters=2500, verbose=False)
    m = ipr_quality_multi(chip, r.clean_image, cfg.row_wid, cfg.col_wid,
                          cfg.row_ss, cfg.col_ss, n_targets=5)
    log(f"  {nm:<22} n={m['n_targets']}  "
        f"islr_change={m['islr_change_db']:+.3f}+/-{m['islr_change_db_std']:.3f} dB  "
        f"mainlobe={m['mainlobe_preservation']:.3f}+/-{m['mainlobe_preservation_std']:.3f}")
    log(f"                         targets: {m['targets']}")

# ---- sanity: identical images must be a perfect no-op
m = ipr_quality(chip, chip, cfg.row_wid, cfg.col_wid, cfg.row_ss, cfg.col_ss)
log()
log(f"  sanity (clean == dirty): mainlobe={m['mainlobe_preservation']:.6f} "
    f"islr_change={m['islr_change_db']:+.6f} dB  -> {verdict(m)}")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a14_verify_quality_metrics.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
