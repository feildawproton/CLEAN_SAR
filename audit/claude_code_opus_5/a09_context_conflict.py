"""
audit/claude_code_opus_5/a09_context_conflict.py

Isolating the "CUDA driver error: invalid resource handle" seen when the
PyTorch backend runs after the native CUDA backend.

cuda_backend._init_cuda_driver() calls cuCtxCreate_v2() and cuCtxSetCurrent()
when no context is current yet. That creates a SECOND, non-primary context on
the device. PyTorch uses the PRIMARY context. Whether the two coexist depends
on which one got established first.

Scenario matrix, each in a FRESH interpreter:
  S1: torch CUDA initialised first, then native CUDA, then torch again
  S2: native CUDA driver initialised first (via resolve_backend('auto')),
      then torch
  S3: resolve_backend('auto') only (no CUDA run), then torch
"""
import subprocess
import sys
import textwrap

PY = "/home/feildaw/mypyenv/bin/python"
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


PRE = textwrap.dedent("""
    import sys, numpy as np
    sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
    from clean_sar.sicd_handler import SICDHandler
    from clean_sar.config import CleanPhysicsConfig
    from clean_sar.algorithm import run_hogbom_clean
    h = SICDHandler("/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf")
    chip, _ = h.read_chip(2000, 4000, 2064, 4064)
    cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
    KW = dict(config=cfg, psf_size=33, gain=0.1, threshold=0.05,
              max_iters=30, verbose=False)
    def run(be):
        r = run_hogbom_clean(dirty_image=chip, backend=be, **KW)
        return f"{be} OK (suppr={r.suppression_db:.2f} dB)"
""")

SCEN = {
    "S1  torch first, then cuda, then torch": """
        print(run("pytorch"))
        print(run("cuda"))
        print(run("pytorch"))
    """,
    "S2  cuda first, then torch": """
        print(run("cuda"))
        print(run("pytorch"))
    """,
    "S3  resolve_backend('auto') only, then torch": """
        from clean_sar.backends import resolve_backend
        print("resolve_backend('auto') ->", resolve_backend("auto"))
        print(run("pytorch"))
    """,
    "S4  is_cuda_native_available() probe, then torch": """
        from clean_sar.backends import is_cuda_native_available
        print("is_cuda_native_available() ->", is_cuda_native_available())
        print(run("pytorch"))
    """,
    "S5  torch.cuda touched first, then probe, then torch": """
        import torch
        torch.zeros(1, device="cuda")
        from clean_sar.backends import is_cuda_native_available
        print("is_cuda_native_available() ->", is_cuda_native_available())
        print(run("pytorch"))
    """,
}

for name, body in SCEN.items():
    src = PRE + textwrap.dedent(body)
    p = subprocess.run([PY, "-c", src], capture_output=True, text=True, timeout=600)
    log("=" * 84)
    log(name)
    log("=" * 84)
    for ln in p.stdout.strip().splitlines():
        log("   " + ln)
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        log("   STDERR (last 3 lines):")
        for ln in err[-3:]:
            log("      " + ln)
        log(f"   exit code {p.returncode}   <-- FAILED")
    else:
        log("   exit code 0")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a09_context_conflict.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")

# ---- second round: the precise trigger ----
SCEN2 = {
    "S6  probe FIRST (creates its own ctx), then torch, then cuda, then torch": """
        from clean_sar.backends import is_cuda_native_available
        is_cuda_native_available()          # cuCtxCreate_v2: own, non-primary context
        print(run("pytorch"))               # torch lazily inits and makes ITS ctx current
        print(run("cuda"))                  # cuCtxSetCurrent(own ctx) -- never restored
        print(run("pytorch"))               # torch op on the wrong current context
    """,
    "S7  same, but WITHOUT the leading probe": """
        print(run("pytorch"))
        print(run("cuda"))
        print(run("pytorch"))
    """,
}
for name, body in SCEN2.items():
    src = PRE + textwrap.dedent(body)
    p = subprocess.run([PY, "-c", src], capture_output=True, text=True, timeout=600)
    log("=" * 84); log(name); log("=" * 84)
    for ln in p.stdout.strip().splitlines():
        log("   " + ln)
    if p.returncode != 0:
        for ln in p.stderr.strip().splitlines()[-2:]:
            log("      " + ln)
        log(f"   exit code {p.returncode}   <-- FAILED")
    else:
        log("   exit code 0")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a09_context_conflict.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
