"""
audit/claude_code_opus_5/a02_metadata.py

Dumps the SICD grid metadata actually driving CleanPhysicsConfig, and checks:
  - which weighting window each file declares (determines which code path runs)
  - Grid Row/Col KCtr (ignored by psf.py -> would make the true IPR complex)
  - ImageData FirstRow/FirstCol vs SCPPixel (chip origin correctness)
  - config.global_to_metric (naive) vs sarkit rowcol_to_xrowycol (rigorous)
  - magnitude of the spatially-varying shear angle theta across the scene
"""
import glob
import sys
import numpy as np
import lxml.etree as etree

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


files = sorted(glob.glob("/home/feildaw/data/*.nitf")) + sorted(
    glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf")
)

for fp in files:
    name = fp.split("/")[-1]
    log("=" * 88)
    log(f"{name}")
    log("=" * 88)
    h = SICDHandler(fp)
    xh = h.xh

    def g(p, d="<absent>"):
        try:
            v = xh.load(p)
            return d if v is None else v
        except Exception:
            return d

    log(f"  NumRows/NumCols   : {h.num_rows} x {h.num_cols}")
    log(f"  FirstRow/FirstCol : {h.first_row}, {h.first_col}")
    log(f"  SCPPixel          : {h.scp_pixel}")
    log(f"  Row SS / Col SS   : {h.row_ss:.6f} / {h.col_ss:.6f}  m")
    log(f"  Row BW / Col BW   : {h.row_bw:.6f} / {h.col_bw:.6f}  cycles/m")
    log(f"  Row Wid / Col Wid : {h.row_wid:.6f} / {h.col_wid:.6f}  m")
    log(f"  Row KCtr / Col KCtr: {g('./{*}Grid/{*}Row/{*}KCtr')} / {g('./{*}Grid/{*}Col/{*}KCtr')}")
    log(f"  Row WgtType       : {h.row_wgt_name!r}   Col WgtType: {h.col_wgt_name!r}")
    log(f"  Row DeltaK1/2     : {g('./{*}Grid/{*}Row/{*}DeltaK1')} / {g('./{*}Grid/{*}Row/{*}DeltaK2')}")
    log(f"  Col DeltaK1/2     : {g('./{*}Grid/{*}Col/{*}DeltaK1')} / {g('./{*}Grid/{*}Col/{*}DeltaK2')}")
    log(f"  Grid ImagePlane   : {g('./{*}Grid/{*}ImagePlane')}   Type: {g('./{*}Grid/{*}Type')}")
    log(f"  SCPCOA SlantRange : {h.scp_slant_range:.2f} m")
    log(f"  SCPCOA GrazeAng   : {g('./{*}SCPCOA/{*}GrazeAng')}  TwistAng: {g('./{*}SCPCOA/{*}TwistAng')}")
    log(f"  ImageFormation    : {g('./{*}ImageFormation/{*}ImageFormAlgo')}")
    log(f"  is_pfa / is_rma   : {h.is_pfa} / {h.is_rma}")

    # oversample factor: how many samples per resolution cell
    log(f"  Oversample (1/(SS*BW)): row {1.0/(h.row_ss*h.row_bw):.4f}  col {1.0/(h.col_ss*h.col_bw):.4f}")

    # ---- naive config metric vs rigorous sarkit projection
    cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))
    pts = [(0, 0), (h.num_rows // 2, h.num_cols // 2), (h.num_rows - 1, h.num_cols - 1)]
    log("  -- config.global_to_metric (naive) vs sarkit rowcol_to_xrowycol (rigorous) --")
    worst = 0.0
    for (r, c) in pts:
        xa, ya = cfg.global_to_metric(float(r), float(c))
        xb, yb = h.global_to_metric(float(r), float(c))
        d = max(abs(xa - float(xb)), abs(ya - float(yb)))
        worst = max(worst, d)
        log(f"     ({r:6d},{c:6d}): naive=({xa:12.3f},{ya:12.3f})  sarkit=({float(xb):12.3f},{float(yb):12.3f})  dmax={d:.4g}")
    log(f"     worst disagreement: {worst:.4g} m")

    # ---- how much does theta actually vary over the scene?
    rr = np.array([0, h.num_rows - 1])
    cc = np.array([0, h.num_cols - 1])
    thetas = []
    for r in rr:
        for c in cc:
            x, y = cfg.global_to_metric(float(r), float(c))
            thetas.append(np.arctan2(y, cfg.scp_slant_range + x))
    thetas = np.array(thetas)
    span = thetas.max() - thetas.min()
    # PSF displacement at the kernel edge (32 px) caused by the full theta span
    kernel_reach_m = 32 * h.col_ss
    log(f"  -- spatially-varying shear angle theta --")
    log(f"     theta range over full scene: [{thetas.min():+.6e}, {thetas.max():+.6e}] rad "
        f"= [{np.degrees(thetas.min()):+.5f}, {np.degrees(thetas.max()):+.5f}] deg")
    log(f"     total span                 : {span:.6e} rad ({np.degrees(span):.6f} deg)")
    log(f"     max PSF-edge displacement from rotation, at 32 px lever arm "
        f"({kernel_reach_m:.2f} m): {kernel_reach_m*span:.6e} m = "
        f"{kernel_reach_m*span/h.col_ss:.6e} pixels")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a02_metadata.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
