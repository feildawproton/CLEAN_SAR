"""
audit/claude_code_opus_5/a20_spatial_variation_regime.py

F9 asks how much the "exact spatially-varying IPR" actually varies. On the 12
datasets here the answer is 0.26 % of peak, because theta = atan2(y, R0 + x) is
tiny at spaceborne slant ranges.

That invites the real question: in WHICH regime does it matter? This maps the
effect against slant range and scene half-width so the feature can be scoped
rather than merely discounted, and marks where the actual datasets sit.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


# Representative X-band geometry from 2023-11-14 UMBRA-04
ROW_SS, COL_SS = 0.4974, 0.6407
ROW_BW, COL_BW = 1.5768, 1.1976
PSF_SIZE = 65


def variation_pct(slant_range_m, half_width_m):
    """
    max |PSF(scene edge) - PSF(SCP)| as a percentage of PSF peak, for a scene
    whose furthest pixel is `half_width_m` from the SCP in both axes.
    """
    cfg = CleanPhysicsConfig(
        row_ss=ROW_SS, col_ss=COL_SS, row_bw=ROW_BW, col_bw=COL_BW,
        row_wid=0.886 / ROW_BW, col_wid=0.886 / COL_BW,
        scp_slant_range=slant_range_m, scp_row=0.0, scp_col=0.0,
        row_wgt="UNIFORM", col_wgt="UNIFORM",
    )
    gen = PSFGenerator(cfg)
    centre = gen.compute_psf(0, 0, psf_size=PSF_SIZE)
    r_edge = half_width_m / ROW_SS
    c_edge = half_width_m / COL_SS
    worst = 0.0
    for sr, sc in ((r_edge, c_edge), (r_edge, -c_edge),
                   (-r_edge, c_edge), (-r_edge, -c_edge)):
        p = gen.compute_psf(sr, sc, psf_size=PSF_SIZE)
        worst = max(worst, float(np.max(np.abs(p - centre))))
    return 100.0 * worst


log("=" * 92)
log("PSF VARIATION ACROSS A SCENE, vs. SLANT RANGE AND SCENE SIZE")
log("=" * 92)
log("  max |PSF(scene corner) - PSF(SCP)|, as % of PSF peak")
log()

RANGES = [(5e3, "5 km   airborne, low"), (20e3, "20 km  airborne"),
          (50e3, "50 km  airborne, standoff"), (100e3, "100 km"),
          (300e3, "300 km"), (650e3, "650 km spaceborne (this data)")]
HALF_WIDTHS = [500, 1000, 2500, 5000, 10000]

log(f"  {'slant range':<30} " + " ".join(f"{h/1000:>6.1f}km" for h in HALF_WIDTHS))
log(f"  {'-'*30} " + " ".join("-" * 8 for _ in HALF_WIDTHS))
for r, label in RANGES:
    cells = []
    for h in HALF_WIDTHS:
        v = variation_pct(r, h)
        cells.append(f"{v:>7.2f}%")
    log(f"  {label:<30} " + " ".join(cells))

log()
log("  (scene half-width = distance from SCP to the furthest pixel in each axis;")
log("   2.5 km half-width is roughly the largest scene in this dataset)")

# ------------------------------------------------------------------ thresholds
log()
log("=" * 92)
log("WHERE THE FEATURE STARTS TO EARN ITS COST")
log("=" * 92)


def max_range_for(threshold_pct, half_width_m):
    """Largest slant range at which variation still exceeds threshold_pct."""
    lo, hi = 1e3, 2e6
    if variation_pct(lo, half_width_m) < threshold_pct:
        return None
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if variation_pct(mid, half_width_m) >= threshold_pct:
            lo = mid
        else:
            hi = mid
    return lo


log(f"  {'scene half-width':<20} {'R0 for >1% variation':>26} {'R0 for >10% variation':>26}")
log(f"  {'-'*20} {'-'*26} {'-'*26}")
for h in (1000, 2500, 5000, 10000):
    r1 = max_range_for(1.0, h)
    r10 = max_range_for(10.0, h)
    f = lambda r: f"R0 < {r/1000:.0f} km" if r else "never"      # noqa: E731
    log(f"  {h/1000:>6.1f} km{'':<12} {f(r1):>26} {f(r10):>26}")

log()
log("  The 12 datasets audited sit at R0 = 647-764 km with half-widths of")
log("  1.4-2.8 km, i.e. far outside every regime above -- hence 0.26 % of peak.")
log()
log("  The feature is real physics and correctly implemented. It becomes")
log("  material for airborne collects (R0 of tens of km) and for very large")
log("  scenes; at spaceborne ranges it is a sub-pixel, quarter-percent")
log("  correction that costs a PSF recomputation on every CLEAN iteration.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/"
          "a20_spatial_variation_regime.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
