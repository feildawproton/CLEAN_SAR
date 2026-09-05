"""
audit/claude_code_opus_5/a10_real_sidelobe_suppression.py

The reported headline metric is called "Sidelobe Suppression (dB)" but is
computed as 20*log10(first residual peak / last residual peak) -- a property of
the greedy peak-chase, not of sidelobes.

Here we measure ACTUAL sidelobe suppression on the developer's own showcase
chip (demo_clean.py UMBRA_CHIP, described as "Strong Scatterer Point Target"):
around the brightest scatterer, how much energy is removed from the sidelobe
annulus while preserving the mainlobe?

Compared across: correct physics, a delta PSF (no deconvolution model at all),
and a deliberately wrong bandwidth -- alongside the reported suppression_db.
"""
import sys
import copy
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


FP = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
CHIP = (1932, 3931, 2188, 4187)  # demo_clean.py UMBRA_CHIP
h = SICDHandler(FP)
chip, _ = h.read_chip(*CHIP)
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(CHIP[0], CHIP[1]))
log(f"showcase chip {chip.shape} from demo_clean.py UMBRA_CHIP {CHIP}")
log(f"declared: row_wid={cfg.row_wid:.4f} m col_wid={cfg.col_wid:.4f} m  "
    f"row_ss={cfg.row_ss:.4f} col_ss={cfg.col_ss:.4f}")

pr, pc = np.unravel_index(np.argmax(np.abs(chip)), chip.shape)
log(f"brightest scatterer at chip ({pr},{pc}), |amp|={np.abs(chip[pr,pc]):.4f}")

# mainlobe radius in pixels (~ one resolution cell), sidelobe annulus beyond it
mainlobe_px = max(cfg.row_wid / cfg.row_ss, cfg.col_wid / cfg.col_ss)
R_IN = int(np.ceil(1.5 * mainlobe_px))
R_OUT = 40
yy, xx = np.ogrid[:chip.shape[0], :chip.shape[1]]
rad = np.sqrt((yy - pr) ** 2 + (xx - pc) ** 2)
annulus = (rad > R_IN) & (rad <= R_OUT)
core = rad <= R_IN
log(f"mainlobe ~{mainlobe_px:.2f} px; core mask r<={R_IN} px, "
    f"sidelobe annulus {R_IN}<r<={R_OUT} px ({annulus.sum()} pixels)")
log()


def sidelobe_db(img):
    """Integrated sidelobe energy in the annulus, relative to the mainlobe peak power."""
    peak = np.max(np.abs(img[core])) ** 2
    e = np.sum(np.abs(img[annulus]) ** 2)
    return 10 * np.log10(e / peak)


base_sl = sidelobe_db(chip)
base_peak = np.max(np.abs(chip[core]))
log(f"DIRTY image: integrated sidelobe energy = {base_sl:+.3f} dB rel. mainlobe peak power; "
    f"mainlobe peak |A| = {base_peak:.4f}")
log()

variants = [
    ("CORRECT physics", dict()),
    ("delta PSF (bw=1e6)", dict(row_bw=1e6, col_bw=1e6)),
    ("bandwidth x2 (wrong)", dict(row_bw=cfg.row_bw * 2, col_bw=cfg.col_bw * 2)),
    ("bandwidth /2 (wrong)", dict(row_bw=cfg.row_bw / 2, col_bw=cfg.col_bw / 2)),
]

log(f"  {'variant':<22} {'reported':>10} {'TRUE sidelobe':>15} {'sidelobe':>10} "
    f"{'mainlobe peak':>15}")
log(f"  {'':<22} {'suppr dB':>10} {'energy dB':>15} {'change dB':>10} {'preserved':>15}")
log(f"  {'-'*22} {'-'*10} {'-'*15} {'-'*10} {'-'*15}")

base_d = dict(cfg.__dict__)
for nm, over in variants:
    d = dict(base_d)
    d.update(over)
    c2 = CleanPhysicsConfig(**d)
    r = run_hogbom_clean(dirty_image=chip, config=c2, backend="pytorch",
                         beam_type="gaussian", psf_size=65, gain=0.1,
                         threshold=0.02, max_iters=2500, verbose=False)
    ci = r.clean_image
    sl = sidelobe_db(ci)
    pk = np.max(np.abs(ci[core]))
    log(f"  {nm:<22} {r.suppression_db:>10.2f} {sl:>15.3f} {sl-base_sl:>+10.3f} "
        f"{pk/base_peak:>14.3f}x")

log()
log("Interpretation:")
log("  'reported suppr dB' is what the README, demo, benchmark CSVs and the")
log("  'parity' test all quote. 'TRUE sidelobe energy change' is what the phrase")
log("  'Sidelobe Suppression' actually promises. They are different quantities.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a10_real_sidelobe_suppression.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
