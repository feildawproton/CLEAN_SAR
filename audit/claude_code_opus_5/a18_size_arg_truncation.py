"""
audit/claude_code_opus_5/a18_size_arg_truncation.py

cuda_backend.py calls the driver through ctypes without setting .argtypes:

    _CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_residual), bytes_complex)

`bytes_complex` is a bare Python int. cuMemAlloc_v2's second parameter is a
size_t (64-bit). Without argtypes, ctypes marshals a Python int as a C int
(32-bit), so any size >= 2**31 is liable to be truncated / sign-extended.

In a07 a 3 GiB allocation returned CUDA_ERROR_OUT_OF_MEMORY on a card with
~7 GiB free, which is consistent with truncation but does not prove it. This
distinguishes the two by allocating the same size both ways.
"""
import ctypes
import sys

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


cu = ctypes.CDLL("libcuda.so.1")
cu.cuInit(0)
dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
cu.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b))
log(f"device free = {free_b.value / 2**30:.2f} GiB of {total_b.value / 2**30:.2f} GiB")
log()

RC = {0: "SUCCESS", 1: "INVALID_VALUE", 2: "OUT_OF_MEMORY"}


def attempt(nbytes, typed):
    p = ctypes.c_void_p()
    arg = ctypes.c_size_t(nbytes) if typed else nbytes
    rc = cu.cuMemAlloc_v2(ctypes.byref(p), arg)
    if rc == 0:
        cu.cuMemFree_v2(p)
    return rc


log(f"  {'request':>10} {'as bare Python int':>26} {'as ctypes.c_size_t':>26}")
log(f"  {'-'*10} {'-'*26} {'-'*26}")
for gib in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    n = int(gib * 2**30)
    fits = n < free_b.value
    a = attempt(n, typed=False)
    b = attempt(n, typed=True)
    log(f"  {gib:>7.1f} GiB {RC.get(a, a):>26} {RC.get(b, b):>26}"
        f"   {'(fits in free memory)' if fits else '(exceeds free memory)'}")

log()
log("  A row where the bare int FAILS but c_size_t SUCCEEDS is direct evidence")
log("  that the unprototyped size argument is being truncated at 32 bits.")
log()
log(f"  2**31 boundary = {2**31 / 2**30:.1f} GiB. Buffers in this codebase are")
log(f"  H*W*8 bytes, so a scene larger than {2**31 // 8 / 1e6:.0f} Mpixels crosses it.")
log(f"  Largest scene benchmarked here is 83.16 Mpixels ({83.16e6 * 8 / 2**30:.2f} GiB) -- under the")
log(f"  limit, which is why this has not yet been hit.")

cu.cuCtxDestroy_v2(ctx)

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a18_size_arg_truncation.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
