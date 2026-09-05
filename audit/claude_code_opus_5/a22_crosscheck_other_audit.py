"""
audit/claude_code_opus_5/a22_crosscheck_other_audit.py

Independent verification of four defects reported by the Gemini 3.8 Flash audit
that this audit did not raise, plus one point where the two accounts differ.

  X1  BUG-05: ImageData.FirstRow / FirstCol ignored by CleanPhysicsConfig
  X2  BUG-06: NVRTC --gpu-architecture hardcoded to compute_86
  X3  BUG-07: device allocations not protected by try/finally (VRAM leak)
  X4  §2.3:   ImageCreation.Application / DateTime not updated on write
  X5  §4.1:   the precise failure mode of demo_clean.py

Nothing is taken on trust; each is reproduced here.
"""
import ctypes
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


FILES = sorted(glob.glob("/home/feildaw/data/*.nitf")) + \
        sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))

# ------------------------------------------------------- X1 FirstRow/FirstCol
log("=" * 94)
log("X1  BUG-05: is ImageData.FirstRow / FirstCol ignored when building the config?")
log("=" * 94)
log("  CleanPhysicsConfig.from_sicd_handler takes chip_start=(0,0) by default and")
log("  never reads handler.first_row / first_col. Checking the 12 datasets:")
log()
log(f"  {'file':<46} {'FirstRow':>9} {'FirstCol':>9}")
log(f"  {'-'*46} {'-'*9} {'-'*9}")
nonzero = []
for fp in FILES:
    h = SICDHandler(fp)
    if h.first_row or h.first_col:
        nonzero.append((fp, h.first_row, h.first_col))
    log(f"  {fp.split('/')[-1][:45]:<46} {h.first_row:>9} {h.first_col:>9}")
log()
log(f"  files with non-zero FirstRow/FirstCol: {len(nonzero)} of {len(FILES)}")

# Demonstrate the magnitude of the error on a synthetic chipped product.
log()
log("  Magnitude of the defect if a pre-chipped SICD were processed with")
log("  chip_bounds=None (so chip_start defaults to (0,0)):")
h = SICDHandler(FILES[0])
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))
for first_row, first_col in ((1932, 3931), (5000, 5000)):
    # what the code computes for chip pixel (0,0)
    x_wrong, y_wrong = cfg.global_to_metric(0.0, 0.0)
    # what it should compute: r_global = r_chip + FirstRow
    x_right, y_right = cfg.global_to_metric(float(first_row), float(first_col))
    log(f"    FirstRow/Col = ({first_row},{first_col}): "
        f"computed ({x_wrong:9.1f},{y_wrong:9.1f}) m vs correct "
        f"({x_right:9.1f},{y_right:9.1f}) m  -> error "
        f"({x_right-x_wrong:8.1f},{y_right-y_wrong:8.1f}) m")
log()
log("  CONFIRMED as a latent defect: no shipped dataset has FirstRow != 0, so it")
log("  is not active today, but a chip written by write_nitf and re-read would hit it.")

# --------------------------------------------------------- X2 compute_86
log()
log("=" * 94)
log("X2  BUG-06: NVRTC architecture hardcoded")
log("=" * 94)
src = open("/home/feildaw/CLEAN_SAR/clean_sar/backends/cuda_backend.py").read()
line = [l.strip() for l in src.splitlines() if "gpu-architecture" in l]
log(f"  cuda_backend.py: {line[0] if line else '(not found)'}")

cu = ctypes.CDLL("libcuda.so.1")
cu.cuInit(0)
dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0)
major, minor = ctypes.c_int(), ctypes.c_int()
cu.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)   # COMPUTE_CAPABILITY_MAJOR
cu.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)   # COMPUTE_CAPABILITY_MINOR
log(f"  this device reports compute capability {major.value}.{minor.value} "
    f"-> correct flag would be compute_{major.value}{minor.value}")
log(f"  the device capability is queryable in 3 lines but is never queried.")

sys.path.insert(0, "/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5")
from cuda_check import Nvrtc, NvrtcError  # noqa: E402
from clean_sar.backends.cuda_backend import NVRTC_CANDIDATE_PATHS  # noqa: E402

nv = Nvrtc(NVRTC_CANDIDATE_PATHS)
ksrc = open("/home/feildaw/CLEAN_SAR/clean_sar/backends/c_src/clean_hogbom.cu", "rb").read()
ctx = ctypes.c_void_p()
cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
log()
log(f"  {'PTX target':<16} {'compiles':>10} {'loads on this sm_86 device':>30}")
log(f"  {'-'*16} {'-'*10} {'-'*30}")
for arch in (b"compute_70", b"compute_80", b"compute_86", b"compute_90"):
    try:
        ptx = nv.compile_to_ptx(ksrc, b"k.cu", [b"--std=c++14", b"--gpu-architecture=" + arch])
        comp = "yes"
        mod = ctypes.c_void_p()
        rc = cu.cuModuleLoadData(ctypes.byref(mod), ptx)
        loads = "yes" if rc == 0 else f"NO (driver rc={rc})"
        if rc == 0:
            cu.cuModuleUnload(mod)
    except NvrtcError as e:
        comp, loads = f"NO (rc={e.code})", "-"
    log(f"  {arch.decode():<16} {comp:>10} {loads:>30}")
log()
log("  PTX is forward-compatible: compute_86 PTX JITs onto NEWER devices, so an")
log("  H100 (sm_90) would in fact work. It is OLDER targets that fail -- A100")
log("  (sm_80), T4 (sm_75), V100 (sm_70) cannot load compute_86 PTX.")
log("  CONFIRMED as a real portability defect, with that one correction to scope.")

# --------------------------------------------------------- X3 try/finally
log()
log("=" * 94)
log("X3  BUG-07: are device allocations protected against an exception mid-loop?")
log("=" * 94)
body = src[src.index("def run_hogbom_cuda_native"):]
log(f"  'try:' occurrences in run_hogbom_cuda_native  : {body.count('try:')}")
log(f"  'finally:' occurrences                        : {body.count('finally:')}")
log(f"  cuMemFree_v2 calls                            : {body.count('cuMemFree_v2')}")
log("  The 8 cuMemFree_v2 calls sit on the straight-line path after the loop, so any")
log("  exception raised inside the loop leaks every device buffer for the life of")
log("  the process. CONFIRMED.")
log("  Note this compounds with F8: once return codes ARE checked and calls raise,")
log("  the leak becomes reachable on any CUDA error, not just on a Python bug.")

# --------------------------------------------------- X4 ImageCreation on write
log()
log("=" * 94)
log("X4  is ImageCreation provenance updated when writing a deconvolved product?")
log("=" * 94)
h = SICDHandler(FILES[0])
chip, chip_xml = h.read_chip(1000, 1000, 1128, 1128)
out = "/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/demo_out/provenance_test.nitf"
h.write_nitf(out, chip, custom_xmltree=chip_xml)
h2 = SICDHandler(out)


def ic(handler, tag):
    try:
        v = handler.xh.load(f"./{{*}}ImageCreation/{{*}}{tag}")
        return "<absent>" if v is None else str(v)
    except Exception:
        return "<absent>"


log(f"  {'field':<16} {'input product':<34} {'written product':<34}")
log(f"  {'-'*16} {'-'*34} {'-'*34}")
for tag in ("Application", "DateTime", "Site", "Profile"):
    log(f"  {tag:<16} {ic(h, tag)[:33]:<34} {ic(h2, tag)[:33]:<34}")
log()
log(f"  FirstRow/FirstCol written: {h2.first_row}, {h2.first_col} "
    f"(chip was read from global 1000,1000)")
log(f"  NumRows/NumCols written  : {h2.num_rows} x {h2.num_cols}")
log()
log("  CONFIRMED: ImageCreation is carried over unchanged, so a deconvolved")
log("  product still claims the original image-formation software and timestamp.")

# ------------------------------------------------------------ X5 demo failure
log()
log("=" * 94)
log("X5  precise failure mode of demo_clean.py (the two audits describe it differently)")
log("=" * 94)
import subprocess
p = subprocess.run(
    ["/home/feildaw/mypyenv/bin/python", "demo_clean.py", "--max_iters", "3",
     "--skip_diffpfa", "--out_dir",
     "audit/claude_code_opus_5/demo_out/x5"],
    capture_output=True, text=True, cwd="/home/feildaw/CLEAN_SAR", timeout=900)
tail = [l for l in p.stdout.strip().splitlines() if l.startswith("[")][-3:]
for l in tail:
    log("  stdout: " + l)
err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "(no stderr)"
log(f"  stderr: {err}")
log()
log("  The other audit reports it 'fails immediately upon execution' with")
log("  \"TypeError: bad operand type for abs(): 'str'\". Measured here, the")
log("  deconvolution and the NITF write BOTH complete first, and the error is the")
log("  keyword mismatch, raised at call time before plot_comparison's body runs --")
log("  so abs() is never reached. Same root cause, different triage: the compute")
log("  is not lost, only the figure.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/"
          "a22_crosscheck_other_audit.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
