"""
audit/claude_code_opus_5/a19_verify_cuda_check.py

Verifies audit/claude_code_opus_5/cuda_check.py fixes all three F8 defects:
  1. size_t arguments no longer truncate at 2**31
  2. failed calls raise CudaError instead of returning garbage
  3. NVRTC compile failures surface the compiler log
and that a real data round-trip through the checked path is bit-exact.
"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR/audit")
sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from cuda_check import CudaDriver, Nvrtc, CudaError, NvrtcError  # noqa: E402
from clean_sar.backends.cuda_backend import NVRTC_CANDIDATE_PATHS  # noqa: E402

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


cu = CudaDriver().init()
dev = cu.device(0)
ctx = cu.primary_context(dev)
free, total = cu.mem_info()
log(f"driver loaded; primary context = {hex(ctx.value or 0)}")
log(f"device free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
log()

# ---------------------------------------------------------------- 1. size_t
log("=" * 88)
log("1. SIZE ARGUMENTS NO LONGER TRUNCATE AT 2**31")
log("=" * 88)
raw = ctypes.CDLL("libcuda.so.1")   # deliberately unprototyped, as the tree does it
log(f"  {'request':>10} {'unprototyped (current code)':>30} {'cuda_check':>22}")
log(f"  {'-'*10} {'-'*30} {'-'*22}")
for gib in (1.5, 2.0, 3.0):
    n = int(gib * 2**30)
    p = ctypes.c_void_p()
    rc = raw.cuMemAlloc_v2(ctypes.byref(p), n)
    if rc == 0:
        raw.cuMemFree_v2(p)
    old = {0: "SUCCESS", 1: "INVALID_VALUE", 2: "OUT_OF_MEMORY"}.get(rc, rc)
    try:
        q = cu.mem_alloc(n)
        cu.mem_free(q)
        new = "SUCCESS"
    except CudaError as e:
        new = e.name
    log(f"  {gib:>7.1f} GiB {old:>30} {new:>22}")

# ------------------------------------------------------------- 2. raising
log()
log("=" * 88)
log("2. FAILURES RAISE INSTEAD OF RETURNING GARBAGE")
log("=" * 88)
try:
    cu.mem_alloc(64 * 2**30)
    log("  ERROR: 64 GiB allocation did not raise")
except CudaError as e:
    log(f"  cu.mem_alloc(64 GiB) raised CudaError:")
    log(f"    {e}")

buf = np.empty(4, dtype=np.complex64)
try:
    cu.memcpy_dtoh(buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(0), 32)
    log("  ERROR: copy from NULL did not raise")
except CudaError as e:
    log(f"  cu.memcpy_dtoh(from NULL) raised CudaError:")
    log(f"    {e}")

log()
log("  For comparison, the current tree ignores both return codes and hands")
log("  back the uninitialised np.empty() buffer as if it were a result.")

# ------------------------------------------------------------- 3. NVRTC log
log()
log("=" * 88)
log("3. NVRTC COMPILE FAILURES SURFACE THE COMPILER LOG")
log("=" * 88)
nv = Nvrtc(NVRTC_CANDIDATE_PATHS)
bad = b"__global__ void k(float* p) { p[0] = undefined_symbol_here; }"
try:
    nv.compile_to_ptx(bad, b"bad.cu", [b"--std=c++14", b"--gpu-architecture=compute_86"])
    log("  ERROR: bad source compiled successfully?")
except NvrtcError as e:
    first = [l for l in e.log.splitlines() if l.strip()][:3]
    log(f"  raised NvrtcError (status {e.code}); compiler log:")
    for l in first:
        log(f"    {l.strip()}")
log("  The current tree returns False here and discards the log entirely.")

good = b"extern \"C\" __global__ void k(float* p) { p[threadIdx.x] = 1.0f; }"
ptx = nv.compile_to_ptx(good, b"good.cu",
                        [b"--std=c++14", b"--gpu-architecture=compute_86"])
log(f"  valid source compiles: {len(ptx)} bytes of PTX")

# ------------------------------------------------------- 4. round-trip check
log()
log("=" * 88)
log("4. DATA ROUND-TRIP THROUGH THE CHECKED PATH IS BIT-EXACT")
log("=" * 88)
rng = np.random.default_rng(0)
for n_mpix in (1, 16):
    a = (rng.standard_normal((n_mpix * 1000, 1000)) +
         1j * rng.standard_normal((n_mpix * 1000, 1000))).astype(np.complex64)
    nbytes = a.nbytes
    d = cu.mem_alloc(nbytes)
    cu.memcpy_htod(d, a.ctypes.data_as(ctypes.c_void_p), nbytes)
    b = np.zeros_like(a)
    cu.memcpy_dtoh(b.ctypes.data_as(ctypes.c_void_p), d, nbytes)
    cu.mem_free(d)
    log(f"  {a.size / 1e6:5.1f} Mpix ({nbytes / 2**20:7.1f} MiB): "
        f"bit-identical = {np.array_equal(a, b)}")

log()
log("  Also verifies the real clean_hogbom.cu still compiles through the")
log("  checked NVRTC path:")
src = open("/home/feildaw/CLEAN_SAR/clean_sar/backends/c_src/clean_hogbom.cu", "rb").read()
ptx = nv.compile_to_ptx(src, b"clean_hogbom.cu",
                        [b"--std=c++14", b"--gpu-architecture=compute_86"])
mod = ctypes.c_void_p()
cu.call("cuModuleLoadData", ctypes.byref(mod), ptx)
names = [b"reduce_max_pass1_kernel", b"reduce_max_pass2_kernel",
         b"fused_clean_sub_add_kernel", b"synthesize_clean_image_kernel"]
found = []
for nm in names:
    k = ctypes.c_void_p()
    cu.call("cuModuleGetFunction", ctypes.byref(k), mod, nm)
    found.append(nm.decode())
log(f"  PTX {len(ptx)} bytes; kernels resolved: {found}")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a19_verify_cuda_check.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
