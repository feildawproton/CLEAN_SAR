"""
audit/claude_code_opus_5/a17_context_strategy.py

Answers: should cuda_backend RETAIN the primary context, or create/destroy its
own context per run so each implementation is exercised end-to-end?

Every experiment runs in a fresh subprocess so nothing here can corrupt the
parent session.

  E1  Does PyTorch use the device PRIMARY context? Is cuDevicePrimaryCtxRetain
      the same handle?
  E2  Is cuCtxCreate_v2's context a different one, and what does the second
      context cost in device memory?
  E3  Cost of establishing a context: retain vs create.
  E4  DANGER: what happens to a live PyTorch if the primary context is reset,
      i.e. the literal "end the current one" strategy.
  E5  FIX A -- retain the primary context: probe -> torch -> cuda -> torch.
  E6  FIX B -- keep an own context but push/pop around each run so the caller's
      context is always restored: probe -> torch -> cuda -> torch.
  E7  Create/destroy a private context PER RUN, then use torch again.
"""
import subprocess
import textwrap

PY = "/home/feildaw/mypyenv/bin/python"
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


PRE = textwrap.dedent('''
    import ctypes, sys, time
    sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)

    def cur():
        c = ctypes.c_void_p()
        cu.cuCtxGetCurrent(ctypes.byref(c))
        return c.value

    def freemem():
        f, t = ctypes.c_size_t(), ctypes.c_size_t()
        cu.cuMemGetInfo_v2(ctypes.byref(f), ctypes.byref(t))
        return f.value / 2**20

    def small_clean(backend):
        import numpy as np
        from clean_sar.sicd_handler import SICDHandler
        from clean_sar.config import CleanPhysicsConfig
        from clean_sar.algorithm import run_hogbom_clean
        h = SICDHandler("/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf")
        chip, _ = h.read_chip(2000, 4000, 2064, 4064)
        cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
        r = run_hogbom_clean(dirty_image=chip, config=cfg, backend=backend,
                             psf_size=33, gain=0.1, threshold=0.05,
                             max_iters=30, verbose=False)
        return f"{backend} OK ({r.suppression_db:.2f} dB)"
''')

EXPS = {
"E1  torch's context vs cuDevicePrimaryCtxRetain": '''
    import torch
    torch.zeros(1, device="cuda")
    torch_ctx = cur()
    p = ctypes.c_void_p()
    cu.cuDevicePrimaryCtxRetain(ctypes.byref(p), dev)
    print(f"torch current context      = {hex(torch_ctx)}")
    print(f"cuDevicePrimaryCtxRetain   = {hex(p.value)}")
    print(f"SAME CONTEXT: {torch_ctx == p.value}")
''',

"E2  cuCtxCreate_v2 makes a second context; memory cost": '''
    import torch
    torch.zeros(1, device="cuda")
    before = freemem()
    torch_ctx = cur()
    c = ctypes.c_void_p()
    cu.cuCtxCreate_v2(ctypes.byref(c), 0, dev)
    after = freemem()
    print(f"torch context   = {hex(torch_ctx)}")
    print(f"created context = {hex(c.value)}")
    print(f"SAME CONTEXT: {torch_ctx == c.value}")
    print(f"device free memory before/after 2nd context: {before:.0f} / {after:.0f} MiB "
          f"(cost {before-after:.0f} MiB)")
''',

"E3  cost of establishing a context": '''
    import torch
    torch.zeros(1, device="cuda")
    t0 = time.perf_counter()
    p = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(p), dev)
    t_retain = (time.perf_counter()-t0)*1e3
    t0 = time.perf_counter()
    c = ctypes.c_void_p(); cu.cuCtxCreate_v2(ctypes.byref(c), 0, dev)
    t_create = (time.perf_counter()-t0)*1e3
    t0 = time.perf_counter()
    cu.cuCtxDestroy_v2(c)
    t_destroy = (time.perf_counter()-t0)*1e3
    print(f"cuDevicePrimaryCtxRetain : {t_retain:8.3f} ms")
    print(f"cuCtxCreate_v2           : {t_create:8.3f} ms")
    print(f"cuCtxDestroy_v2          : {t_destroy:8.3f} ms")
    print(f"-> create+destroy per run would add {t_create+t_destroy:.1f} ms of setup")
''',

"E4  DANGER: resetting the primary context under a live PyTorch": '''
    import torch
    x = torch.ones(1024, device="cuda")
    print("torch tensor allocated OK")
    cu.cuDevicePrimaryCtxReset_v2(dev)          # "end the current context"
    print("primary context reset")
    try:
        y = (x * 2).sum().item()
        print(f"torch still works: {y}")
    except Exception as e:
        print(f"torch BROKEN: {type(e).__name__}: {str(e)[:80]}")
''',

"E5  FIX A: retain the primary context (probe->torch->cuda->torch)": '''
    import clean_sar.backends.cuda_backend as cb
    _orig = cb._init_cuda_driver
    def patched():
        ok = _orig()
        if ok and cb._CUDA_CTX is not None:
            p = ctypes.c_void_p()
            cb._CUDA_LIB.cuDevicePrimaryCtxRetain(ctypes.byref(p), dev)
            cb._CUDA_CTX = p                     # adopt the PRIMARY context
        return ok
    cb._init_cuda_driver = patched
    from clean_sar.backends import is_cuda_native_available
    print("probe ->", is_cuda_native_available())
    print(small_clean("pytorch"))
    print(small_clean("cuda"))
    print(small_clean("pytorch"))
''',

"E6  FIX B: own context, but restore the caller's after each run": '''
    from clean_sar.backends import is_cuda_native_available
    print("probe ->", is_cuda_native_available())
    import clean_sar.backends.cuda_backend as cb
    _orig_run = cb.run_hogbom_cuda_native
    def wrapped(*a, **kw):
        saved = cur()                            # push
        try:
            return _orig_run(*a, **kw)
        finally:
            if saved:
                cb._CUDA_LIB.cuCtxSetCurrent(ctypes.c_void_p(saved))   # pop
    cb.run_hogbom_cuda_native = wrapped
    import clean_sar.algorithm as alg
    alg.run_hogbom_cuda_native = wrapped
    print(small_clean("pytorch"))
    print(small_clean("cuda"))
    print(small_clean("pytorch"))
''',

"E7  create + destroy a private context per run, then use torch": '''
    from clean_sar.backends import is_cuda_native_available
    print("probe ->", is_cuda_native_available())
    print(small_clean("pytorch"))
    torch_ctx = cur()
    c = ctypes.c_void_p()
    cu.cuCtxCreate_v2(ctypes.byref(c), 0, dev)   # private context for "the run"
    print(f"private context {hex(c.value)} created and current")
    cu.cuCtxDestroy_v2(c)                        # tear it down
    print("private context destroyed; current context is now", hex(cur() or 0))
    try:
        print(small_clean("pytorch"))
    except Exception as e:
        print(f"torch BROKEN after teardown: {type(e).__name__}: {str(e)[:70]}")
''',
}

for name, body in EXPS.items():
    src = PRE + textwrap.dedent(body)
    p = subprocess.run([PY, "-c", src], capture_output=True, text=True, timeout=600)
    log("=" * 88)
    log(name)
    log("=" * 88)
    for ln in (p.stdout or "").strip().splitlines():
        log("   " + ln)
    if p.returncode != 0:
        for ln in (p.stderr or "").strip().splitlines()[-2:]:
            log("   ! " + ln)
        log(f"   exit {p.returncode}  <-- FAILED")
    else:
        log("   exit 0")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a17_context_strategy.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
