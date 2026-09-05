"""
audit/claude_code_opus_5/a21_warmup_methodology.py

Should the benchmark do a discarded warm-up run?

The current methodology takes ONE sample per backend per file, PyTorch always
first, with no warm-up. Several costs land unevenly on that first sample:

  - torch CUDA context init (~750 ms) lands inside the first file's torch_h2d
  - NVRTC compile (~5 s) is excluded from CUDA timings entirely
  - the GPU idles at a low power state; clocks ramp over the first few launches
  - torch's caching allocator and the driver's allocator both grow on first use

This measures the size of that effect on the reported speedup, comparing:

  A  current methodology: 1 pytorch sample, then 1 cuda sample, cold
  B  warm-up discarded, then 5 interleaved pairs, median

Each condition runs in a fresh subprocess.
"""
import json
import subprocess
import sys
import textwrap

PY = "/home/feildaw/mypyenv/bin/python"
FILE = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_SICD.nitf"
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


PRE = textwrap.dedent(f'''
    import json, sys, numpy as np
    sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
    from clean_sar.sicd_handler import SICDHandler
    from clean_sar.config import CleanPhysicsConfig
    from clean_sar.algorithm import run_hogbom_clean

    h = SICDHandler("{FILE}")
    img = h.read_full_image()
    cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(0, 0))
    KW = dict(config=cfg, beam_type="gaussian", psf_size=65, gain=0.1,
              threshold=0.02, max_iters=300, verbose=False)

    def run(be):
        r = run_hogbom_clean(dirty_image=img, backend=be, **KW)
        return dict(h2d=r.h2d_time_sec*1e3, compute=r.pure_compute_time_sec*1e3,
                    d2h=r.d2h_time_sec*1e3)
''')

COND = {
"A  current methodology (cold, one sample each, pytorch first)": '''
    t = run("pytorch")
    c = run("cuda")
    print(json.dumps({"torch": [t], "cuda": [c]}))
''',
"B  warm-up discarded, then 5 interleaved pairs": '''
    run("pytorch"); run("cuda")            # warm-up, discarded
    ts, cs = [], []
    for _ in range(5):
        ts.append(run("pytorch"))
        cs.append(run("cuda"))
    print(json.dumps({"torch": ts, "cuda": cs}))
''',
}


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


results = {}
for name, body in COND.items():
    p = subprocess.run([PY, "-c", PRE + textwrap.dedent(body)],
                       capture_output=True, text=True, timeout=1800)
    line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        log(f"{name}: FAILED\n{p.stderr[-400:]}")
        continue
    results[name] = json.loads(line[-1])

log("=" * 96)
log(f"WARM-UP EFFECT ON THE REPORTED SPEEDUP")
log(f"scene: {FILE.split('/')[-1]}, 300 iterations")
log("=" * 96)
log(f"  {'condition':<52} {'torch ms':>10} {'cuda ms':>10} {'speedup':>9}")
log(f"  {'-'*52} {'-'*10} {'-'*10} {'-'*9}")
summary = {}
for name, r in results.items():
    tc = med([x["compute"] for x in r["torch"]])
    cc = med([x["compute"] for x in r["cuda"]])
    summary[name] = (tc, cc, tc / cc)
    log(f"  {name:<52} {tc:>10.1f} {cc:>10.1f} {tc/cc:>8.2f}x")

if len(summary) == 2:
    (a_t, a_c, a_s), (b_t, b_c, b_s) = summary.values()
    log()
    log(f"  cold-vs-warm change in reported speedup: {a_s:.2f}x -> {b_s:.2f}x "
        f"({100*(b_s-a_s)/a_s:+.1f}%)")
    log(f"    pytorch compute {a_t:.1f} -> {b_t:.1f} ms ({100*(b_t-a_t)/a_t:+.1f}%)")
    log(f"    cuda    compute {a_c:.1f} -> {b_c:.1f} ms ({100*(b_c-a_c)/a_c:+.1f}%)")

log()
log("=" * 96)
log("PER-SAMPLE SPREAD IN CONDITION B (is one sample enough?)")
log("=" * 96)
for name, r in results.items():
    if "B" not in name:
        continue
    for be in ("torch", "cuda"):
        v = [x["compute"] for x in r[be]]
        log(f"  {be:<6} compute ms: {np.round(v,1).tolist() if False else [round(x,1) for x in v]}")
        log(f"  {'':<6} min={min(v):.1f} median={med(v):.1f} max={max(v):.1f} "
            f"spread={100*(max(v)-min(v))/med(v):.1f}% of median")
    log()
    log("  first-sample h2d vs later (torch CUDA init lands in the first):")
    hs = [x["h2d"] for x in r["torch"]]
    log(f"    torch h2d ms: {[round(x,1) for x in hs]}")

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/"
          "a21_warmup_methodology.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
