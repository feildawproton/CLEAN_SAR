"""
audit/claude_code_opus_5/a08_phantom_options.py

Options that are accepted, documented, and echoed in the log, but which the
default (CUDA) backend does not implement:

  - beam_type="mainlobe"  : clean_hogbom.cu only ever evaluates the Gaussian
                            beam. The kernel has no mainlobe branch at all.
  - psf_generator=...     : run_hogbom_cuda_native takes the argument and never
                            reads it; a custom/subclassed PSFGenerator is
                            silently discarded.

Both matter because backend defaults to "auto", which resolves to CUDA whenever
NVRTC is available -- so these are the DEFAULT code paths.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.algorithm import run_hogbom_clean
from clean_sar.backends import resolve_backend

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


log(f"resolve_backend('auto') -> {resolve_backend('auto')!r}   "
    f"(so 'auto' is the CUDA path on this machine)")
log()

FP = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
h = SICDHandler(FP)
chip, _ = h.read_chip(2000, 4000, 2128, 4128)
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
KW = dict(config=cfg, psf_size=65, gain=0.1, threshold=0.02, max_iters=200, verbose=False)

log("=" * 86)
log("(1) DOES beam_type='mainlobe' DO ANYTHING ON THE CUDA BACKEND?")
log("=" * 86)
for be in ["pytorch", "cuda"]:
    rg = run_hogbom_clean(dirty_image=chip, backend=be, beam_type="gaussian", **KW)
    rm = run_hogbom_clean(dirty_image=chip, backend=be, beam_type="mainlobe", **KW)
    d = np.max(np.abs(rg.restored_model - rm.restored_model))
    peak = np.max(np.abs(rg.restored_model)) or 1.0
    log(f"   backend={be:8s}: max|restored(gaussian) - restored(mainlobe)| = {d:.6e}  "
        f"({100*d/peak:.3f}% of peak)   "
        f"{'<-- IDENTICAL: mainlobe silently ignored' if d/peak < 1e-6 else '(beam types differ)'}")

log()
log("   verbose log line printed by the CUDA backend when mainlobe is requested:")
run_hogbom_clean(dirty_image=chip, backend="cuda", beam_type="mainlobe",
                 config=cfg, psf_size=65, gain=0.1, threshold=0.02,
                 max_iters=3, verbose=True)
log("   ^ it reports 'Beam: MAINLOBE' while evaluating the Gaussian beam.")

log()
log("=" * 86)
log("(2) DOES A CUSTOM psf_generator REACH THE CUDA BACKEND?")
log("=" * 86)


class LoudPSFGenerator(PSFGenerator):
    """A PSF generator that returns an obviously different PSF and counts calls."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def compute_psf(self, row, col, psf_size=65, **kw):
        self.calls += 1
        p = np.zeros((psf_size, psf_size), dtype=np.complex64)
        p[psf_size // 2, psf_size // 2] = 1.0  # delta PSF: unmistakably different
        return p

    def get_psfs_torch(self, row, col, psf_size=65, beam_type="gaussian", device=None):
        import torch
        self.calls += 1
        d = np.zeros((psf_size, psf_size), dtype=np.complex64)
        d[psf_size // 2, psf_size // 2] = 1.0
        t = torch.as_tensor(d, dtype=torch.complex64, device=device)
        return t, t


for be in ["pytorch", "cuda"]:
    gen = LoudPSFGenerator(cfg)
    try:
        r = run_hogbom_clean(dirty_image=chip, config=cfg, psf_generator=gen, backend=be,
                             beam_type="gaussian", psf_size=65, gain=0.1,
                             threshold=0.02, max_iters=200, verbose=False)
        log(f"   backend={be:8s}: custom generator called {gen.calls:4d} times, "
            f"suppression={r.suppression_db:.2f} dB  "
            f"{'<-- generator IGNORED' if gen.calls == 0 else '(generator used)'}")
    except RuntimeError as e:
        # Not a flaw in this probe: this is the driver-context conflict, isolated
        # in a09. Reaching a pytorch run AFTER native-CUDA runs in a process where
        # the NVRTC probe created its own context raises here. See a09 scenario S6.
        log(f"   backend={be:8s}: RuntimeError: {e}")
        log(f"              ^ CUDA driver context conflict, not a psf_generator issue;")
        log(f"                isolated separately in a09_context_conflict.py (S6 vs S7).")

log()
log("   (The psf_generator result for the CUDA backend is established by inspection:")
log("    run_hogbom_cuda_native accepts psf_generator at cuda_backend.py:139 and never")
log("    references it again anywhere in the function body.)")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a08_phantom_options.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
