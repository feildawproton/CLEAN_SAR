"""
audit/claude_code_opus_5/a07_cuda_error_handling.py

The CUDA backend checks NONE of the return codes from cuMemAlloc_v2,
cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuLaunchKernel or cuCtxSynchronize.
Output buffers are np.empty() (uninitialised). So if a device allocation or a
copy fails, the function returns UNINITIALISED HOST MEMORY as a "result", and
suppression_db -- computed from host-side history_peaks, not from the arrays --
still looks perfectly healthy.

This script:
  (1) demonstrates the failure mode directly by forcing an over-large allocation
      and showing what the unchecked path returns;
  (2) runs the largest benchmarked scene (72 Mpix) through the exact
      pytorch-then-cuda sequence the benchmark uses, and validates the returned
      arrays are real (finite, consistent with pytorch) rather than garbage;
  (3) checks the config=None path, where the CUDA backend silently substitutes
      placeholder physics instead of raising like the pytorch backend does.
"""
import sys
import ctypes
import time
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean
import clean_sar.backends.cuda_backend as cb

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


# --------------------------------------------------------------- (1) error codes
log("=" * 86)
log("(1) ARE CUDA API FAILURES DETECTED?")
log("=" * 86)
cb._init_cuda_driver()
lib = cb._CUDA_LIB
lib.cuCtxSetCurrent(cb._CUDA_CTX)

free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
lib.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b))
log(f"   GPU memory: free={free_b.value/2**30:.2f} GiB  total={total_b.value/2**30:.2f} GiB")

p = ctypes.c_void_p()
huge = 64 * 2**30  # 64 GiB, guaranteed to fail
rc = lib.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(huge))
log(f"   cuMemAlloc_v2(64 GiB) -> return code {rc} "
    f"({'CUDA_SUCCESS' if rc == 0 else 'FAILURE'}), device ptr = {p.value}")
log(f"   cuda_backend.py ignores this return code entirely "
    f"(no 'if ... != 0' on any cuMemAlloc call).")

# what the caller would then get back
out = np.empty((4, 4), dtype=np.complex64)
rc2 = lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), p, 4 * 4 * 8)
log(f"   cuMemcpyDtoH_v2 from the NULL pointer -> return code {rc2} "
    f"({'CUDA_SUCCESS' if rc2 == 0 else 'FAILURE'}) -- also unchecked")
log(f"   host buffer now contains uninitialised memory, e.g. {out.ravel()[:3]}")
log(f"   -> a failed run returns a plausible-looking array and a healthy "
    f"suppression_db.")

# Also: the size argument is passed as an unprototyped ctypes int
log()
log("   Size arguments are passed without argtypes. Checking >2GiB behaviour:")
try:
    q = ctypes.c_void_p()
    rc3 = lib.cuMemAlloc_v2(ctypes.byref(q), 3 * 2**30)  # bare python int
    log(f"     cuMemAlloc_v2(ptr, 3*2**30) as bare int -> rc={rc3}, ptr={q.value}")
    if q.value:
        lib.cuMemFree_v2(q)
except Exception as e:
    log(f"     raised {type(e).__name__}: {e}")
    log(f"     -> full scenes needing >2 GiB per buffer would crash here.")

# --------------------------------------------------- (2) largest scene validation
log()
log("=" * 86)
log("(2) LARGEST BENCHMARKED SCENE (72 Mpix): ARE THE CUDA RESULTS REAL?")
log("=" * 86)
FP = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
h = SICDHandler(FP)
t0 = time.perf_counter()
img = h.read_full_image()
log(f"   {FP.split('/')[-1]}  {img.shape} = {img.size/1e6:.1f} Mpix, "
    f"{img.nbytes/2**30:.2f} GiB per buffer x4, read in {time.perf_counter()-t0:.1f}s")
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))
KW = dict(config=cfg, beam_type="gaussian", psf_size=65, gain=0.1,
          threshold=0.02, max_iters=200, verbose=False)

# exact benchmark ordering: pytorch first (leaves torch's caching allocator full)
rt = run_hogbom_clean(dirty_image=img, backend="pytorch", **KW)
log(f"   pytorch: iters={rt.iterations} suppr={rt.suppression_db:.4f} dB  "
    f"h2d={rt.h2d_time_sec*1e3:.0f} compute={rt.pure_compute_time_sec*1e3:.0f} "
    f"d2h={rt.d2h_time_sec*1e3:.0f} ms")
lib.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b))
log(f"   GPU free after pytorch run: {free_b.value/2**30:.2f} GiB "
    f"(torch caching allocator still holds its pool)")

rc_ = run_hogbom_clean(dirty_image=img, backend="cuda", **KW)
log(f"   cuda   : iters={rc_.iterations} suppr={rc_.suppression_db:.4f} dB  "
    f"h2d={rc_.h2d_time_sec*1e3:.0f} compute={rc_.pure_compute_time_sec*1e3:.0f} "
    f"d2h={rc_.d2h_time_sec*1e3:.0f} ms")

log()
log("   VALIDATING the CUDA arrays are real numbers and not uninitialised memory:")
for f in ["clean_image", "residual_image", "components_map", "restored_model"]:
    a, b = getattr(rt, f), getattr(rc_, f)
    finite = np.isfinite(b.view(np.float32)).all()
    denom = np.max(np.abs(a)) or 1.0
    log(f"     {f:16s}: all-finite={bool(finite)}  max|torch-cuda|/peak="
        f"{np.max(np.abs(a-b))/denom:.3e}  relL2={np.linalg.norm(a-b)/(np.linalg.norm(a) or 1):.3e}")
log(f"   suppression_db delta = {abs(rt.suppression_db-rc_.suppression_db):.6f} dB")
n = min(len(rt.history_coords), len(rc_.history_coords))
same = sum(1 for i in range(n) if rt.history_coords[i] == rc_.history_coords[i])
log(f"   peak selection agreement: {same}/{n}")

# --------------------------------------------------------- (3) config=None path
log()
log("=" * 86)
log("(3) config=None: pytorch RAISES, cuda SILENTLY SUBSTITUTES PLACEHOLDER PHYSICS")
log("=" * 86)
small, _ = h.read_chip(1000, 1000, 1128, 1128)
for be in ["pytorch", "cuda"]:
    try:
        r = run_hogbom_clean(dirty_image=small, config=None, backend=be,
                             psf_size=65, gain=0.1, threshold=0.02,
                             max_iters=100, verbose=False)
        log(f"   backend={be:8s}: NO ERROR. iterations={r.iterations}, "
            f"suppression={r.suppression_db:.2f} dB "
            f"<-- computed with row_ss=0.5, col_ss=0.5, row_bw=1.5, col_bw=1.5, "
            f"sigma=0.5 placeholders")
    except Exception as e:
        log(f"   backend={be:8s}: raised {type(e).__name__}: {str(e)[:90]}")

log()
log("   The placeholder values are hardcoded in cuda_backend.py lines 205-208 and")
log("   273-276 as the ' if config else <default>' branch. They are not the")
log("   scene's physics, so the run produces a confidently-wrong deconvolution.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a07_cuda_error_handling.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
