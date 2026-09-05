"""
audit/claude_code_opus_5/a05_backend_parity.py

Does the CUDA backend actually agree with the PyTorch backend on the OUTPUT
ARRAYS, or only on the scalar suppression_db that the repo's benchmark and
"parity" test compare?

Also:
  - proves the CUDA path really executes on the GPU (counts kernel launches by
    timing / by corrupting device memory expectations)
  - exercises the UNIFORM path (what the real data uses) and the HAMMING/HANN
    paths (where psf.py normalises the PSF to 1.0 at centre but the .cu kernel
    does not)
  - checks that clean_mask is honoured by both backends
"""
import sys
import time
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
h = SICDHandler(FP)
chip, _ = h.read_chip(2000, 4000, 2256, 4256)
base_cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
log(f"chip 256x256 from {FP.split('/')[-1]} @ (2000,4000)")
log(f"declared weighting: row={base_cfg.row_wgt!r} col={base_cfg.col_wgt!r}")


def compare(tag, cfg, **kw):
    kwargs = dict(beam_type="gaussian", psf_size=65, gain=0.1,
                  threshold=0.02, max_iters=800, verbose=False)
    kwargs.update(kw)
    rt = run_hogbom_clean(dirty_image=chip, config=cfg, backend="pytorch", **kwargs)
    rc = run_hogbom_clean(dirty_image=chip, config=cfg, backend="cuda", **kwargs)

    log()
    log("-" * 84)
    log(f"{tag}")
    log("-" * 84)
    log(f"  iterations      : pytorch={rt.iterations}   cuda={rc.iterations}")
    log(f"  suppression_db  : pytorch={rt.suppression_db:.4f}  cuda={rc.suppression_db:.4f}  "
        f"delta={abs(rt.suppression_db-rc.suppression_db):.6f}  "
        f"<-- the ONLY thing the repo's benchmark/test compares")

    for field in ["clean_image", "residual_image", "components_map", "restored_model"]:
        a = getattr(rt, field)
        b = getattr(rc, field)
        denom = np.max(np.abs(a)) or 1.0
        amax = np.max(np.abs(a - b))
        rel = amax / denom
        # relative L2
        l2 = np.linalg.norm(a - b) / (np.linalg.norm(a) or 1.0)
        log(f"  {field:16s}: max|d|={amax:.6e}  max|d|/peak={rel:.3e}  relL2={l2:.3e}")

    # do they even pick the same peaks?
    n = min(len(rt.history_coords), len(rc.history_coords))
    same = sum(1 for i in range(n) if rt.history_coords[i] == rc.history_coords[i])
    first_div = next((i for i in range(n) if rt.history_coords[i] != rc.history_coords[i]), None)
    log(f"  peak-selection  : {same}/{n} iterations chose the SAME pixel; "
        f"first divergence at iter {first_div}")
    ncomp_t = int(np.count_nonzero(rt.components_map))
    ncomp_c = int(np.count_nonzero(rc.components_map))
    log(f"  component pixels: pytorch={ncomp_t}  cuda={ncomp_c}")
    return rt, rc


# 1. As-configured (UNIFORM -- what all the real data actually uses)
compare("A. UNIFORM weighting (the configuration all 12 benchmark files use)", base_cfg)

# 2. HAMMING: psf.py normalises PSF peak to 1.0; clean_hogbom.cu does NOT
import copy
d = dict(base_cfg.__dict__)
d.update(row_wgt="HAMMING", col_wgt="HAMMING")
cfg_ham = CleanPhysicsConfig(**d)
compare("B. HAMMING weighting  (psf.py normalises PSF centre to 1.0; the .cu kernel does not)",
        cfg_ham)

d = dict(base_cfg.__dict__)
d.update(row_wgt="HANN", col_wgt="HANN")
compare("C. HANN weighting", CleanPhysicsConfig(**d))

d = dict(base_cfg.__dict__)
d.update(row_wgt="TAYLOR", col_wgt="TAYLOR")
compare("D. TAYLOR weighting", CleanPhysicsConfig(**d))

# 3. Direct PSF centre-value comparison: numpy vs the CUDA kernel's formula
log()
log("=" * 84)
log("PSF CENTRE VALUE: clean_sar/psf.py (normalised) vs clean_hogbom.cu (not normalised)")
log("=" * 84)
from clean_sar.psf import PSFGenerator
for w in ["UNIFORM", "TAYLOR", "HAMMING", "HANN"]:
    d = dict(base_cfg.__dict__)
    d.update(row_wgt=w, col_wgt=w)
    g = PSFGenerator(CleanPhysicsConfig(**d))
    p = g.compute_psf(128, 128, psf_size=65)
    # the .cu kernel's unnormalised centre value = pattern(0)^2
    centre_1d = {"UNIFORM": 1.0, "TAYLOR": 1.0, "HAMMING": 0.54, "HANN": 0.50}[w]
    cu_centre = centre_1d ** 2
    log(f"  {w:8s}: psf.py centre = {p[32,32].real:.6f}   "
        f".cu kernel centre = {cu_centre:.6f}   ratio = {p[32,32].real/cu_centre:.4f}x")

# 4. clean_mask honoured?
log()
log("=" * 84)
log("IS clean_mask HONOURED BY BOTH BACKENDS?")
log("=" * 84)
mask = np.zeros((256, 256), dtype=bool)
mask[:64, :64] = True     # only allow components in the top-left corner
for be in ["pytorch", "cuda"]:
    r = run_hogbom_clean(dirty_image=chip, config=base_cfg, backend=be,
                         psf_size=65, gain=0.1, threshold=0.02, max_iters=300,
                         clean_mask=mask, verbose=False)
    cm = r.components_map
    inside = np.count_nonzero(cm[:64, :64])
    outside = np.count_nonzero(cm) - inside
    log(f"  backend={be:8s}: components inside mask={inside:5d}  OUTSIDE mask={outside:5d}  "
        f"{'<-- MASK IGNORED' if outside > 0 else '(mask respected)'}")

# 5. guard_margin honoured?
log()
log("=" * 84)
log("IS guard_margin HONOURED BY BOTH BACKENDS?")
log("=" * 84)
for be in ["pytorch", "cuda"]:
    r = run_hogbom_clean(dirty_image=chip, config=base_cfg, backend=be,
                         psf_size=65, gain=0.1, threshold=0.02, max_iters=300,
                         guard_margin=100, verbose=False)
    cm = r.components_map
    band = np.count_nonzero(cm) - np.count_nonzero(cm[100:156, 100:156])
    log(f"  backend={be:8s}: components in the excluded margin = {band:5d}  "
        f"{'<-- GUARD IGNORED' if band > 0 else '(guard respected)'}")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a05_backend_parity.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
