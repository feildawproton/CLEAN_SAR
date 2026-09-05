"""
audit/claude_code_opus_5/a06_benchmark_methodology.py

Audits benchmark_pytorch_vs_cuda.py's methodology.

  (1) What is inside the PyTorch backend's "Pure GPU Compute Loop"? Profile how
      much of it is CPU-side numpy PSF synthesis + per-iteration H2D of the
      kernel, versus actual GPU work. If most of it is CPU numpy, the headline
      "CUDA vs PyTorch GPU speedup" is really "fused kernel vs CPU PSF
      synthesis", i.e. misattributed.

  (2) Reproduce the published end-to-end numbers and check whether the summary
      table's headline (compute_speedup) tells the same story as the
      total_gpu_speedup column that the summary omits.

  (3) Warm-up / run-to-run variance: the benchmark takes ONE sample per backend
      with no warm-up, and PyTorch always runs first.
"""
import sys
import time
import cProfile
import pstats
import io
import json
import numpy as np

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


FP = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_SICD.nitf"
h = SICDHandler(FP)
img = h.read_full_image()
cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))
log(f"scene: {FP.split('/')[-1]}  {img.shape}  ({img.size/1e6:.2f} Mpix)")
log()

KW = dict(config=cfg, beam_type="gaussian", psf_size=65, gain=0.1,
          threshold=0.02, max_iters=300, verbose=False)

# ---------------------------------------------------------------- (1) profile
log("=" * 86)
log("(1) WHAT IS THE PYTORCH BACKEND ACTUALLY SPENDING ITS 'PURE GPU COMPUTE' TIME ON?")
log("=" * 86)
pr = cProfile.Profile()
pr.enable()
rt = run_hogbom_clean(dirty_image=img, backend="pytorch", **KW)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
prof = s.getvalue()
keep = [ln for ln in prof.splitlines()
        if any(k in ln for k in ("psf.py", "compute_psf", "get_psfs_torch", "clean_beam",
                                 "pytorch_backend", "{built-in method", "ncalls", "sinc",
                                 "meshgrid", "as_tensor"))]
for ln in keep[:22]:
    log("   " + ln.strip())

log()
log(f"   pytorch: iterations={rt.iterations}  pure_compute={rt.pure_compute_time_sec*1000:.1f} ms")

# Directly time PSF synthesis alone for the same number of calls
from clean_sar.psf import PSFGenerator
g = PSFGenerator(cfg)
coords = rt.history_coords[:rt.iterations]
t0 = time.perf_counter()
for (r, c) in coords:
    g.compute_psf(r, c, psf_size=65)
    g.compute_clean_beam(r, c, psf_size=65, beam_type="gaussian")
t_psf = time.perf_counter() - t0
log(f"   time to synthesise the same {len(coords)} PSF+beam pairs in numpy on CPU: "
    f"{t_psf*1000:.1f} ms")
log(f"   -> CPU PSF synthesis is {100*t_psf/rt.pure_compute_time_sec:.1f}% of the "
    f"PyTorch backend's reported 'Pure GPU Compute' time")
log()
log("   The CUDA backend evaluates the PSF analytically INSIDE the fused kernel,")
log("   so it never pays this cost. The measured 'speedup' is therefore dominated")
log("   by where the PSF is synthesised, not by PyTorch-vs-CUDA GPU efficiency.")

# ---------------------------------------------------------------- (2) reproduce
log()
log("=" * 86)
log("(2) REPRODUCING THE PUBLISHED HEAD-TO-HEAD NUMBERS")
log("=" * 86)
pub = json.load(open("/home/feildaw/CLEAN_SAR/output/benchmarks_comparison_raw/"
                     "pytorch_vs_cuda_benchmark.json"))
row = [r for r in pub if r["filename"] == FP.split("/")[-1]][0]
log(f"   published row for this file (max_iters=1000):")
for k in ["torch_compute_ms", "cuda_compute_ms", "compute_speedup",
          "torch_d2h_ms", "cuda_d2h_ms", "torch_total_gpu_ms", "cuda_total_gpu_ms",
          "total_gpu_speedup", "delta_suppression_db"]:
    log(f"     {k:24s} = {row[k]}")

log()
log("   published table across ALL 12 benchmarked scenes:")
allpub = []
for d in ["benchmarks_comparison_raw", "benchmarks_comparison_diffpfa"]:
    allpub += json.load(open(f"/home/feildaw/CLEAN_SAR/output/{d}/pytorch_vs_cuda_benchmark.json"))
log(f"     {'file':<44} {'compute_speedup':>16} {'total_gpu_speedup':>18}")
for r in allpub:
    flag = "  <-- CUDA SLOWER end-to-end" if r["total_gpu_speedup"] < 1.0 else ""
    log(f"     {r['filename'][:43]:<44} {r['compute_speedup']:>16.2f} "
        f"{r['total_gpu_speedup']:>18.2f}{flag}")
cs = [r["compute_speedup"] for r in allpub]
ts = [r["total_gpu_speedup"] for r in allpub]
log()
log(f"     mean compute_speedup   = {np.mean(cs):.2f}x   <-- THIS is what the summary prints")
log(f"     mean total_gpu_speedup = {np.mean(ts):.2f}x   <-- this column is NOT printed")
log(f"     scenes where CUDA is slower end-to-end: {sum(1 for t in ts if t < 1.0)}/{len(ts)}")

# ---------------------------------------------------------------- (3) variance
log()
log("=" * 86)
log("(3) RUN-TO-RUN VARIANCE AND ORDERING EFFECTS (benchmark takes ONE sample, no warm-up)")
log("=" * 86)
for be in ["pytorch", "cuda"]:
    ts_ = []
    for i in range(5):
        r = run_hogbom_clean(dirty_image=img, backend=be, **KW)
        ts_.append(r.pure_compute_time_sec * 1000)
    a = np.array(ts_)
    log(f"   {be:8s}: {np.array2string(a, precision=1)} ms   "
        f"mean={a.mean():.1f} first/median={a[0]/np.median(a):.2f}x")

log()
log("   D2H isolation check (the published cuda_d2h_ms for big scenes is huge):")
for be in ["pytorch", "cuda"]:
    r = run_hogbom_clean(dirty_image=img, backend=be, **KW)
    log(f"   {be:8s}: h2d={r.h2d_time_sec*1000:8.1f} ms  compute={r.pure_compute_time_sec*1000:9.1f} ms  "
        f"d2h={r.d2h_time_sec*1000:9.1f} ms")
log("   NOTE: the CUDA path copies FOUR full complex64 arrays back "
    "(clean, residual, model, components)")
log("         via pageable cuMemcpyDtoH; the PyTorch path copies four as well but "
    "through torch's")
log("         pinned staging. Neither is wrong, but D2H dominates end-to-end for "
    "large scenes.")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a06_benchmark_methodology.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
