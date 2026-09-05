"""
audit/claude_code_opus_5/a17b_context_fixes.py

a17's E5/E6 patches were themselves faulty, so their failures said nothing
about the strategies. Redone here correctly, with context handles printed at
each step so the mechanism is visible rather than inferred.

  F-A  primary-context retain: make _init_cuda_driver adopt the PRIMARY context
       (the one PyTorch uses) instead of calling cuCtxCreate_v2
  F-B  own context + push/pop: keep cuCtxCreate_v2 but restore the caller's
       context after every run
  F-C  create + destroy a private context per run (the "fresh context each
       time" strategy)

All in fresh subprocesses. Each runs the sequence that currently fails:
    probe -> torch -> cuda -> torch
"""
import subprocess
import textwrap

PY = "/home/feildaw/mypyenv/bin/python"
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


PRE = textwrap.dedent('''
    import ctypes, sys
    sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)

    def cur():
        c = ctypes.c_void_p()
        cu.cuCtxGetCurrent(ctypes.byref(c))
        return c.value or 0

    def primary():
        p = ctypes.c_void_p()
        cu.cuDevicePrimaryCtxRetain(ctypes.byref(p), dev)
        return p.value or 0

    def small_clean(backend):
        from clean_sar.sicd_handler import SICDHandler
        from clean_sar.config import CleanPhysicsConfig
        from clean_sar.algorithm import run_hogbom_clean
        h = SICDHandler("/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf")
        chip, _ = h.read_chip(2000, 4000, 2064, 4064)
        cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
        r = run_hogbom_clean(dirty_image=chip, config=cfg, backend=backend,
                             psf_size=33, gain=0.1, threshold=0.05,
                             max_iters=30, verbose=False)
        return r.suppression_db

    EXPECT = 14.01
    def step(label, backend):
        try:
            v = small_clean(backend)
            ok = "OK " if abs(v - EXPECT) < 0.05 else "WRONG VALUE"
            print(f"   {label:<28} {backend:<8} {v:7.2f} dB  {ok}  ctx={hex(cur())}")
            return abs(v - EXPECT) < 0.05
        except Exception as e:
            print(f"   {label:<28} {backend:<8} {'--':>7}     {type(e).__name__}: {str(e)[:44]}")
            return False
''')

FIXES = {
"F-A  primary-context retain (recommended)": '''
    # Simulate the fix: establish the PRIMARY context as current before the
    # probe, so _init_cuda_driver's cuCtxGetCurrent finds it and adopts it
    # instead of calling cuCtxCreate_v2. In a real fix cuda_backend would call
    # cuDevicePrimaryCtxRetain directly.
    p = primary()
    cu.cuCtxSetCurrent(ctypes.c_void_p(p))
    print(f"   primary context = {hex(p)}")

    from clean_sar.backends import is_cuda_native_available
    import clean_sar.backends.cuda_backend as cb
    print("   probe ->", is_cuda_native_available(),
          "  adopted ctx =", hex(cb._CUDA_CTX.value or 0))
    print(f"   adopted IS primary: {(cb._CUDA_CTX.value or 0) == p}")
    ok = all([step("1 torch", "pytorch"), step("2 cuda", "cuda"),
              step("3 torch", "pytorch"), step("4 cuda", "cuda")])
    print(f"   RESULT: {'ALL PASS' if ok else 'FAILED'}")
''',

"F-B  own context + restore caller's after each run": '''
    from clean_sar.backends import is_cuda_native_available
    print("   probe ->", is_cuda_native_available())
    import clean_sar.backends.cuda_backend as cb
    import clean_sar.algorithm as alg
    print(f"   own ctx = {hex(cb._CUDA_CTX.value or 0)}  primary = {hex(primary())}")

    _orig_run = cb.run_hogbom_cuda_native
    def wrapped(*a, **kw):
        saved = cur()
        try:
            return _orig_run(*a, **kw)
        finally:
            cu.cuCtxSetCurrent(ctypes.c_void_p(saved))
            print(f"      [restored ctx -> {hex(saved)}]")
    cb.run_hogbom_cuda_native = wrapped
    alg.run_hogbom_cuda_native = wrapped

    ok = all([step("1 torch", "pytorch"), step("2 cuda", "cuda"),
              step("3 torch", "pytorch"), step("4 cuda", "cuda")])
    print(f"   RESULT: {'ALL PASS' if ok else 'FAILED'}")
''',

"F-C  create + destroy a private context per run": '''
    from clean_sar.backends import is_cuda_native_available
    import clean_sar.backends.cuda_backend as cb
    import clean_sar.algorithm as alg
    print("   probe ->", is_cuda_native_available())

    _orig_run = cb.run_hogbom_cuda_native
    def wrapped(*a, **kw):
        c = ctypes.c_void_p()
        cu.cuCtxCreate_v2(ctypes.byref(c), 0, dev)   # fresh context for this run
        cb._CUDA_CTX = c
        cb._CUDA_MODULE = None                        # module is context-bound
        cb._CUDA_KERNELS.clear()
        cb._init_cuda_driver()                        # recompile into this ctx
        try:
            return _orig_run(*a, **kw)
        finally:
            cu.cuCtxDestroy_v2(c)                     # tear it down
            cb._CUDA_MODULE = None
            cb._CUDA_KERNELS.clear()
            print(f"      [private ctx destroyed; current -> {hex(cur())}]")
    cb.run_hogbom_cuda_native = wrapped
    alg.run_hogbom_cuda_native = wrapped

    ok = all([step("1 torch", "pytorch"), step("2 cuda", "cuda"),
              step("3 torch", "pytorch"), step("4 cuda", "cuda")])
    print(f"   RESULT: {'ALL PASS' if ok else 'FAILED'}")
''',
}

for name, body in FIXES.items():
    src = PRE + textwrap.dedent(body)
    p = subprocess.run([PY, "-c", src], capture_output=True, text=True, timeout=900)
    log("=" * 92)
    log(name)
    log("=" * 92)
    for ln in (p.stdout or "").strip().splitlines():
        log(ln)
    if p.returncode != 0:
        for ln in (p.stderr or "").strip().splitlines()[-2:]:
            log("   ! " + ln)
        log(f"   exit {p.returncode}")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a17b_context_fixes.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
