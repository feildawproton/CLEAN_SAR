"""
clean_sar/quality.py: IPR quality, target preservation, and annulus ISLR metrics for CLEAN SAR.

Provides discriminating quality metrics for evaluating SAR deconvolution performance:
1. mainlobe_preservation: verifies scatterer amplitude is not eroded.
2. islr_change_db: measures integrated sidelobe energy suppression in an annulus around the scatterer.

WHY THESE TWO
-------------
`CleanResult.suppression_db` = 20*log10(peak_first / peak_last) measures how far
the greedy peak-chase descended. It rises with iteration count whether or not
the PSF is correct: on the showcase chip a delta-function PSF (no deconvolution
at all) scored 33.44 dB against correct physics' 33.98 dB.

Integrated sidelobe reduction ALONE is also not enough -- the delta PSF removed
MORE annulus energy (-5.06 dB vs -4.46 dB), because punching a hole in the image
does reduce surrounding energy. It just destroys the target while doing it.

The pair is what discriminates:

    mainlobe_preservation ~ 1.0  AND  islr_change_db < 0     -> real deconvolution
    mainlobe_preservation >> 1.0                             -> corrupting the target

    variant                 islr_change   mainlobe_preservation
    correct physics           -4.46 dB           0.988   <- only variant that passes
    delta PSF                 -5.06 dB           1.506
    bandwidth x2              -5.56 dB           2.703
    bandwidth /2              +4.05 dB          15.052

DEFINITIONS
-----------
Let (r0,c0) be a bright scatterer located in the DIRTY image (the same pixel is
then used for both images, so the two are measured at the same place).

    mainlobe radius   R_in  = ceil(f * max(row_wid/row_ss, col_wid/col_ss))
                             with f = 1.5 by default (1.5 resolution cells)
    sidelobe annulus  R_in < r <= R_out       (R_out = 40 px by default)

    core power peak   P_pk  = max |img|^2  over  r <= R_in
    annulus energy    E_sl  = sum |img|^2   over  R_in < r <= R_out

    islr_db               = 10*log10(E_sl / P_pk)          [per image]
    islr_change_db        = islr_db(clean) - islr_db(dirty)   negative is good
    mainlobe_preservation = max|clean| / max|dirty|  over the core; 1.0 is ideal

Note islr_db here is an integrated-sidelobe-to-peak ratio over a 2-D annulus,
not the 1-D cut ISLR of the SICD IPDD. It is used for the CHANGE between two
images of the same scene, where the common normalisation cancels.
"""

from typing import Optional, Tuple, Dict, List
import numpy as np


def _masks(shape, r0, c0, r_in, r_out):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    rad = np.sqrt((yy - r0) ** 2 + (xx - c0) ** 2)
    return rad <= r_in, (rad > r_in) & (rad <= r_out)


def find_bright_targets(
    dirty: np.ndarray,
    n: int = 5,
    isolation: float = 6.0,
    edge: int = 40,
    sep: int = 20,
) -> List[Tuple[int, int]]:
    """
    Locate up to `n` bright, reasonably isolated scatterers in the dirty image.

    A candidate qualifies if its peak exceeds `isolation` x the brightest pixel
    in the surrounding 41x41 neighbourhood once the central 11x11 is excluded.
    """
    mag = np.abs(dirty)
    H, W = mag.shape
    out: List[Tuple[int, int]] = []
    for flat in np.argsort(mag.ravel())[::-1]:
        if len(out) >= n:
            break
        r, c = int(flat // W), int(flat % W)
        if r < edge or c < edge or r >= H - edge or c >= W - edge:
            continue
        if any(abs(r - pr) < sep and abs(c - pc) < sep for pr, pc in out):
            continue
        nb = mag[r - 20 : r + 21, c - 20 : c + 21].copy()
        peak = nb[20, 20]
        nb[15:26, 15:26] = 0.0
        ring = nb.max()
        if ring > 0 and peak > isolation * ring:
            out.append((r, c))
    return out


def ipr_quality(
    dirty: np.ndarray,
    clean: np.ndarray,
    row_wid: float,
    col_wid: float,
    row_ss: float,
    col_ss: float,
    target: Optional[Tuple[int, int]] = None,
    mainlobe_factor: float = 1.5,
    r_out_px: int = 40,
) -> Dict[str, float]:
    """
    Mainlobe preservation and integrated-sidelobe change around one scatterer.

    Parameters
    ----------
    dirty, clean : complex 2-D arrays of identical shape.
    row_wid, col_wid : ImpRespWid, metres (Grid/Row|Col/ImpRespWid).
    row_ss, col_ss   : sample spacing, metres (Grid/Row|Col/SS).
    target : (row, col) of the scatterer. If None, the dirty image's peak.
    mainlobe_factor : core radius in resolution cells (default 1.5).
    r_out_px : outer radius of the sidelobe annulus, pixels.

    Returns
    -------
    dict with mainlobe_preservation, islr_dirty_db, islr_clean_db,
    islr_change_db, and the geometry actually used.
    """
    if dirty.shape != clean.shape:
        raise ValueError(f"shape mismatch: {dirty.shape} vs {clean.shape}")

    if target is None:
        target = np.unravel_index(int(np.argmax(np.abs(dirty))), dirty.shape)
    r0, c0 = int(target[0]), int(target[1])

    cells_px = max(row_wid / row_ss, col_wid / col_ss)
    r_in = int(np.ceil(mainlobe_factor * cells_px))
    if r_out_px <= r_in:
        raise ValueError(f"r_out_px ({r_out_px}) must exceed mainlobe radius ({r_in})")

    core, annulus = _masks(dirty.shape, r0, c0, r_in, r_out_px)
    if not annulus.any() or not core.any():
        raise ValueError("target too close to the edge for the requested radii")

    def islr_db(img):
        p_pk = np.max(np.abs(img[core])) ** 2
        e_sl = np.sum(np.abs(img[annulus]) ** 2)
        return 10.0 * np.log10(e_sl / p_pk) if p_pk > 0 else np.nan

    d_db, c_db = islr_db(dirty), islr_db(clean)
    pk_d = np.max(np.abs(dirty[core]))
    pk_c = np.max(np.abs(clean[core]))

    return {
        "target_row": r0,
        "target_col": c0,
        "mainlobe_radius_px": r_in,
        "annulus_outer_px": r_out_px,
        "resolution_cell_px": cells_px,
        "mainlobe_preservation": float(pk_c / pk_d) if pk_d > 0 else np.nan,
        "islr_dirty_db": float(d_db),
        "islr_clean_db": float(c_db),
        "islr_change_db": float(c_db - d_db),
    }


def ipr_quality_multi(dirty, clean, row_wid, col_wid, row_ss, col_ss,
                      n_targets: int = 5, **kw) -> Dict[str, float]:
    """
    ipr_quality averaged over the brightest isolated scatterers. More robust
    than a single target; falls back to the global peak if none qualify.
    """
    targets = find_bright_targets(dirty, n=n_targets)
    if not targets:
        return ipr_quality(dirty, clean, row_wid, col_wid, row_ss, col_ss, **kw)

    rows = []
    for t in targets:
        try:
            rows.append(ipr_quality(dirty, clean, row_wid, col_wid,
                                    row_ss, col_ss, target=t, **kw))
        except ValueError:
            continue
    if not rows:
        return ipr_quality(dirty, clean, row_wid, col_wid, row_ss, col_ss, **kw)

    return {
        "n_targets": len(rows),
        "mainlobe_preservation": float(np.mean([r["mainlobe_preservation"] for r in rows])),
        "mainlobe_preservation_std": float(np.std([r["mainlobe_preservation"] for r in rows])),
        "islr_change_db": float(np.mean([r["islr_change_db"] for r in rows])),
        "islr_change_db_std": float(np.std([r["islr_change_db"] for r in rows])),
        "targets": [(r["target_row"], r["target_col"]) for r in rows],
    }


def verdict(m: Dict[str, float], tol: float = 0.10) -> str:
    """Coarse pass/fail: mainlobe intact AND sidelobes reduced."""
    mp = m.get("mainlobe_preservation", np.nan)
    ch = m.get("islr_change_db", np.nan)
    if not np.isfinite(mp) or not np.isfinite(ch):
        return "INDETERMINATE"
    if abs(mp - 1.0) > tol:
        return f"MAINLOBE CORRUPTED ({mp:.3f}x)"
    return "OK" if ch < 0 else f"SIDELOBES NOT REDUCED ({ch:+.2f} dB)"
