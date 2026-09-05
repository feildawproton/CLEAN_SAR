"""
audit/claude_code_opus_5/a15_regression_bite.py

Proves audit/claude_code_opus_5/test_psf_physics_reference.py actually bites.

A test that passes tells you nothing unless you know it can fail. Here each
historical (or plausible) defect is reintroduced by monkey-patching clean_sar
AT RUNTIME -- no file under clean_sar/ is touched -- and the reference suite is
run against the broken library in a fresh subprocess.

For comparison the two ORIGINAL self-referential tests are run against the same
broken library, to show they do not notice.

Scenarios
  R1  Gaussian sigma reverts to the 2*sqrt(2 ln 2) FWHM convention
      (the earlier audit's Finding 2; beam becomes sqrt(2) too narrow)
  R2  Slant range reverts to the hardcoded 10 km constant
      (the earlier audit's Finding 1)
  R3  Taylor coefficients corrupted (sidelobes no longer -30 dB)
  R4  PSF centre normalisation removed
      (makes psf.py behave like the .cu kernel -- F1)
"""
import os
import subprocess
import sys
import textwrap

PY = "/home/feildaw/mypyenv/bin/python"
ROOT = "/home/feildaw/CLEAN_SAR"
PLUGDIR = os.path.join(ROOT, "audit", "claude_code_opus_5", "_regressions")
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


os.makedirs(PLUGDIR, exist_ok=True)
open(os.path.join(PLUGDIR, "__init__.py"), "w").close()

COMMON = '''
import numpy as np
from clean_sar.psf import PSFGenerator
from clean_sar.config import CleanPhysicsConfig

def _grid(gen, row, col, psf_size):
    _, _, xrow, ycol = gen._get_metric_coords(row, col)
    theta = np.arctan2(ycol, gen.config.scp_slant_range + xrow)
    ct, st = np.cos(theta), np.sin(theta)
    d = np.arange(psf_size) - psf_size // 2
    DR, DC = np.meshgrid(d, d, indexing="ij")
    U = DR * gen.config.row_ss
    V = DC * gen.config.col_ss
    return U * ct + V * st, -U * st + V * ct
'''

SCENARIOS = {
    "R1_gaussian_fwhm": COMMON + '''
def _bad_beam(self, row, col, psf_size=65, beam_type="gaussian"):
    if psf_size % 2 == 0: psf_size += 1
    Up, Vp = _grid(self, row, col, psf_size)
    fwhm_const = 2.0 * np.sqrt(2.0 * np.log(2.0))     # <-- REGRESSION
    sr = self.config.row_wid / fwhm_const
    sc = self.config.col_wid / fwhm_const
    return np.exp(-0.5*((Up/max(sr,1e-6))**2 + (Vp/max(sc,1e-6))**2)).astype(np.complex64)
PSFGenerator.compute_clean_beam = _bad_beam
''',

    "R2_hardcoded_slant_range": COMMON + '''
_orig = CleanPhysicsConfig.from_sicd_handler.__func__
def _bad(cls, handler, chip_start=(0, 0)):
    cfg = _orig(cls, handler, chip_start)
    cfg.scp_slant_range = 10000.0                      # <-- REGRESSION
    return cfg
CleanPhysicsConfig.from_sicd_handler = classmethod(_bad)
''',

    "R3_taylor_coefficients": COMMON + '''
BAD_FM = [0.20, -0.03, 0.01]                           # <-- REGRESSION
def _pat(pos, bw, wgt):
    if wgt in ("UNIFORM","RECT","RECTANGULAR","NONE"): return np.sinc(bw*pos)
    if wgt == "HAMMING":
        return 0.54*np.sinc(bw*pos)+0.23*np.sinc(bw*pos-1)+0.23*np.sinc(bw*pos+1)
    if wgt in ("HANN","HANNING"):
        return 0.5*np.sinc(bw*pos)+0.25*np.sinc(bw*pos-1)+0.25*np.sinc(bw*pos+1)
    if wgt == "TAYLOR":
        p = np.sinc(bw*pos)
        for m, c in enumerate(BAD_FM, start=1):
            p = p + c*(np.sinc(bw*pos-m)+np.sinc(bw*pos+m))
        return p
    return np.sinc(bw*pos)
def _bad_psf(self, row, col, psf_size=65, window_row=None, window_col=None):
    if psf_size % 2 == 0: psf_size += 1
    Up, Vp = _grid(self, row, col, psf_size)
    wr = (window_row or self.config.row_wgt or "UNIFORM").upper()
    wc = (window_col or self.config.col_wgt or "UNIFORM").upper()
    psf = _pat(Up, self.config.row_bw, wr) * _pat(Vp, self.config.col_bw, wc)
    pk = psf[psf_size//2, psf_size//2]
    if abs(pk) > 0: psf = psf / pk
    return psf.astype(np.complex64)
PSFGenerator.compute_psf = _bad_psf
''',

    "R4_no_peak_normalisation": COMMON + '''
def _pat(pos, bw, wgt):
    if wgt == "HAMMING":
        return 0.54*np.sinc(bw*pos)+0.23*np.sinc(bw*pos-1)+0.23*np.sinc(bw*pos+1)
    if wgt in ("HANN","HANNING"):
        return 0.5*np.sinc(bw*pos)+0.25*np.sinc(bw*pos-1)+0.25*np.sinc(bw*pos+1)
    if wgt == "TAYLOR":
        p = np.sinc(bw*pos)
        for m, c in enumerate([0.29265601,-0.01578375,0.00218104], start=1):
            p = p + c*(np.sinc(bw*pos-m)+np.sinc(bw*pos+m))
        return p
    return np.sinc(bw*pos)
def _bad_psf(self, row, col, psf_size=65, window_row=None, window_col=None):
    if psf_size % 2 == 0: psf_size += 1
    Up, Vp = _grid(self, row, col, psf_size)
    wr = (window_row or self.config.row_wgt or "UNIFORM").upper()
    wc = (window_col or self.config.col_wgt or "UNIFORM").upper()
    psf = _pat(Up, self.config.row_bw, wr) * _pat(Vp, self.config.col_bw, wc)
    return psf.astype(np.complex64)                    # <-- REGRESSION: no /peak
PSFGenerator.compute_psf = _bad_psf
''',
}

for name, src in SCENARIOS.items():
    with open(os.path.join(PLUGDIR, f"{name}.py"), "w") as f:
        f.write(src)


def run(target, plugin=None, only=None):
    cmd = [PY, "-m", "pytest", target, "-q", "--no-header",
           "-p", "no:cacheprovider", "--tb=no"]
    if plugin:
        cmd += ["-p", f"_regressions.{plugin}"]
    if only:
        cmd += ["-k", only]
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "audit", "claude_code_opus_5"))
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
    tail = [l for l in p.stdout.strip().splitlines() if "passed" in l or "failed" in l or "error" in l]
    return (tail[-1] if tail else "(no summary)"), p.returncode


NEW = "audit/claude_code_opus_5/test_psf_physics_reference.py"
OLD = "tests/test_psf.py"
OLD_K = "test_slant_range_and_restoring_beam_physics or test_psf_taylor_continuous_window"

log("=" * 96)
log("BASELINE: unmodified library")
log("=" * 96)
s, rc = run(NEW)
log(f"  reference suite            : {s}   [{'PASS' if rc == 0 else 'FAIL'}]")
s, rc = run(OLD, only=OLD_K)
log(f"  the two original tests     : {s}   [{'PASS' if rc == 0 else 'FAIL'}]")

log()
log("=" * 96)
log("WITH EACH DEFECT REINTRODUCED (runtime monkey-patch; clean_sar/ untouched)")
log("=" * 96)
log(f"  {'scenario':<28} {'reference suite':<34} {'original two tests':<30}")
log(f"  {'-'*28} {'-'*34} {'-'*30}")
for name in SCENARIOS:
    s_new, rc_new = run(NEW, plugin=name)
    s_old, rc_old = run(OLD, plugin=name, only=OLD_K)
    v_new = "CAUGHT" if rc_new != 0 else "MISSED"
    v_old = "caught" if rc_old != 0 else "MISSED"
    log(f"  {name:<28} {s_new:<26} {v_new:<7} {s_old:<20} {v_old}")

log()
log("  'CAUGHT' = the suite failed, i.e. the regression was detected.")
log("  The two original tests are expected to MISS R1 (they re-derive the sigma")
log("  formula inline) and R3/R4 (they never call compute_psf for these).")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a15_regression_bite.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
