"""
audit/claude_code_opus_5/a16_recovery_bite.py

Proves audit/claude_code_opus_5/test_clean_recovery_reference.py bites, and that the ORIGINAL
test it replaces does not.

The scenarios patch the PSF the ALGORITHM subtracts, while the synthetic
forward model keeps using the true analytic PSF -- i.e. they simulate a
deconvolution kernel that disagrees with the real IPR. That is the exact
failure mode the original assertion
    max|residual| < max|dirty|
cannot see, because subtracting anything at all satisfies it.

Scenarios
  S1  algorithm's PSF becomes a delta function (no deconvolution model)
  S2  algorithm's PSF is scaled by 0.29 (the unnormalised-Hamming error, F1)
  S3  algorithm's PSF uses 2x the true bandwidth (wrong resolution)
"""
import os
import subprocess
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

# Patch get_psfs_torch (pytorch backend's PSF source) while leaving
# compute_psf -- used by the test's forward model -- untouched.
TEMPLATE = '''
import numpy as np, torch
from clean_sar.psf import PSFGenerator

_orig_psf = PSFGenerator.compute_psf
_orig_beam = PSFGenerator.compute_clean_beam

def _patched(self, row, col, psf_size=65, beam_type="gaussian", device=None):
    dirty = _orig_psf(self, row, col, psf_size=psf_size)
    clean = _orig_beam(self, row, col, psf_size=psf_size, beam_type=beam_type)
{mutation}
    return (torch.as_tensor(dirty, dtype=torch.complex64, device=device),
            torch.as_tensor(clean, dtype=torch.complex64, device=device))

PSFGenerator.get_psfs_torch = _patched
'''

SCENARIOS = {
    "S1_delta_psf": """
    dirty = np.zeros_like(dirty); dirty[psf_size//2, psf_size//2] = 1.0
""",
    "S2_unnormalised_scale": """
    dirty = dirty * 0.2916
""",
    "S3_double_bandwidth": """
    cfg = self.config
    import copy as _c
    c2 = _c.copy(cfg); c2.row_bw = cfg.row_bw*2; c2.col_bw = cfg.col_bw*2
    g2 = PSFGenerator(c2); dirty = _orig_psf(g2, row, col, psf_size=psf_size)
""",
}

for name, mut in SCENARIOS.items():
    body = textwrap.indent(textwrap.dedent(mut).strip("\n"), "    ")
    with open(os.path.join(PLUGDIR, f"{name}.py"), "w") as f:
        f.write(TEMPLATE.format(mutation=body))


def run(target, plugin=None, only=None):
    cmd = [PY, "-m", "pytest", target, "-q", "--no-header",
           "-p", "no:cacheprovider", "--tb=no"]
    if plugin:
        cmd += ["-p", f"_regressions.{plugin}"]
    if only:
        cmd += ["-k", only]
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "audit", "claude_code_opus_5"))
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env=env, timeout=1200)
    tail = [l for l in p.stdout.strip().splitlines()
            if "passed" in l or "failed" in l or "error" in l]
    return (tail[-1] if tail else "(no summary)"), p.returncode


NEW = "audit/claude_code_opus_5/test_clean_recovery_reference.py"
NEW_K = "pytorch"
OLD = "tests/test_clean_algorithm.py"

log("=" * 98)
log("BASELINE: unmodified library")
log("=" * 98)
s, rc = run(NEW, only=NEW_K)
log(f"  recovery reference suite : {s}   [{'PASS' if rc == 0 else 'FAIL'}]")
s, rc = run(OLD)
log(f"  original test            : {s}   [{'PASS' if rc == 0 else 'FAIL'}]")

log()
log("=" * 98)
log("ALGORITHM'S PSF MADE TO DISAGREE WITH THE TRUE IPR (runtime patch)")
log("=" * 98)
log(f"  {'scenario':<24} {'recovery reference':<32} {'original test':<28}")
log(f"  {'-'*24} {'-'*32} {'-'*28}")
for name in SCENARIOS:
    s_new, rc_new = run(NEW, plugin=name, only=NEW_K)
    s_old, rc_old = run(OLD, plugin=name)
    v_new = "CAUGHT" if rc_new != 0 else "MISSED"
    v_old = "caught" if rc_old != 0 else "MISSED"
    log(f"  {name:<24} {s_new:<24} {v_new:<7} {s_old:<19} {v_old}")

log()
log("  'CAUGHT' = suite failed, i.e. the broken deconvolution kernel was detected.")
log("  The original test asserts only max|residual| < max|dirty|, which every")
log("  subtractive procedure satisfies -- so it is expected to MISS all three.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a16_recovery_bite.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
