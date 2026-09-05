# CLEAN_SAR — Combined Audit Report

**Compiled by:** Claude Opus 5 (Anthropic), running in Claude Code
**Date:** 2026-09-05
**Subject:** `/home/feildaw/CLEAN_SAR`, working tree as-is (uncommitted refactor; last commit `f6734a1`)

**Contributing audits, conducted independently:**

| Audit | Directory | Report |
|---|---|---|
| Claude Opus 5 (Claude Code) | [`claude_code_opus_5/`](claude_code_opus_5/) | [`AUDIT_REPORT.md`](claude_code_opus_5/AUDIT_REPORT.md) |
| Agy (Gemini 3.8 Flash) | [`agy_gemini_3.8_flash/`](agy_gemini_3.8_flash/) | [`AUDIT_REPORT.md`](agy_gemini_3.8_flash/AUDIT_REPORT.md) |

**Method.** Each audit was performed without reference to the other. This document
merges them. Every finding below carries an explicit **verification status**;
claims originating in one audit and reproduced by the other are marked as such.
Test and demonstration code is **not reproduced here** — each finding points to
the script in its originating subdirectory.

**Environment.** Python 3.12.3 (`/home/feildaw/mypyenv`), torch 2.6.0+cu124,
numpy 2.4.1, sarkit 1.8.0; NVIDIA RTX 3070 Laptop (compute 8.6, 8 GiB), driver
555.51, WSL2. Twelve SICD datasets under `/home/feildaw/data/` and
`/home/feildaw/diffpfa/workspace/output/`.

---

## Executive summary

**The project is fundamentally sound.** Both audits independently confirm that the
native CUDA backend is genuine (NVRTC → PTX → Driver API, not a PyTorch shim),
that it agrees with the PyTorch backend to float32 rounding under the weighting
the real data uses, and that the analytic impulse-response mathematics is correct.
Given a correct PSF, CLEAN recovers synthetic point targets to 0.1–0.6 % amplitude
error with 100 % of flux on true pixels. Published benchmark speedups reproduce.
All findings from the earlier (2026-08-29) multi-model audit have been properly
fixed.

**Twenty issues are recorded below**, four at P0. They cluster into three themes:

1. **The two backends have diverged, and nothing measures it.** Hamming/Hann
   weighting makes them produce entirely different images; three documented
   options are accepted and silently dropped by the default backend; a missing
   config yields fabricated physics on one backend and an exception on the other.
2. **The evidence system cannot detect the divergence.** The only cross-backend
   comparison anywhere — in the benchmark, the CSVs, and the test named
   `test_cuda_backend_execution_parity` — is a host-side scalar that a
   delta-function PSF scores within 0.54 dB of correct physics on.
3. **Failures are silent rather than loud.** No CUDA return code is checked and
   output buffers are `np.empty()`, so a failed device run returns
   plausible-looking noise together with a healthy quality figure.

The single highest-value change is replacing the parity test with an array-level,
all-weightings comparison: it catches C01 and C02 together. The cheapest
high-value change is raising `NotImplementedError` wherever a backend cannot
honour a requested option.

**Independent corroboration.** The central physics defect (C01) was found by both
audits by different routes and agrees numerically: 3.426× amplitude inflation is
+10.7 dB, and both measured the same 3.5× convergence penalty.

---

## 1. What is verified correct

Recorded because it is load-bearing: these were checked, not assumed.

| Claim | Verified by | Result |
|---|---|---|
| CUDA backend is native, not a PyTorch shim | both | NVRTC compiles PTX; 4 kernels resolved; driver-API launches |
| Backend parity under UNIFORM/TAYLOR | both | ≤ 4×10⁻⁷ (Opus 5), 5.33×10⁻⁷ (Gemini); **800/800** and **200/200** identical peak selections including on a 72-Mpixel scene |
| Analytic IPR is exact | Opus 5 `a01` | max rel. error vs. independent numerical FT: 5×10⁻⁹ uniform, 7×10⁻¹⁰ Hamming, 2×10⁻¹⁶ Hann, 2×10⁻⁹ Taylor |
| Taylor coefficients | Opus 5 `a01` | match `scipy.signal.windows.taylor` (nbar 4, SLL −30 dB) to 1×10⁻⁹ |
| Gaussian restoring-beam width convention | both | σ = Wid/(2√ln2) gives −3.01 dB half-power width; ratio 1.000000 |
| CLEAN algorithm correctness | Opus 5 `a04` | 5 synthetic targets recovered to 0.10–0.57 %, 100 % of flux on true pixels, exactly 5 component pixels |
| Coordinate transform | Opus 5 `a02` | naive `config.global_to_metric` vs. rigorous `sarkit.rowcol_to_xrowycol`: **0 m** on all 12 files |
| Real-valued PSF assumption | Opus 5 `a03` | Grid Row KCtr = 64.04 cyc/m, but sampled image is **baseband** (measured support centre +0.00000), so a real symmetric IPR is correct |
| Published speedups | Opus 5 `a06` | measured 4.1× vs. published 4.14× |
| Prior audit findings fixed | Opus 5 `a01`,`a02` | hardcoded 10 km slant range, √2 Gaussian error, Taylor bug, unbounded cache, xmltree mutation, version mismatch — all genuinely remediated |

---

## 2. Consolidated findings

**Origin** — which audit first raised it. **Verified** — who reproduced it.
`O5` = Claude Opus 5, `G` = Gemini 3.8 Flash.

| ID | Sev | Finding | Origin | Verified | Evidence |
|---|---|---|---|---|---|
| **C01** | P0 | CUDA does not normalise the Hamming/Hann PSF centre → 3.43×/4.0× amplitude error; backends produce unrelated images | both | both | [`a05`](claude_code_opus_5/a05_backend_parity.txt), [`a11`](claude_code_opus_5/a11_normalization_and_wgt.txt), [`test_hamming_parity.py`](agy_gemini_3.8_flash/scripts/test_hamming_parity.py) |
| **C02** | P0 | `beam_type="mainlobe"`, `clean_mask`, `psf_generator` accepted and silently ignored by the default backend; `MAINLOBE` is logged while a Gaussian runs | both | both | [`a08`](claude_code_opus_5/a08_phantom_options.txt), [`a05`](claude_code_opus_5/a05_backend_parity.txt), [`test_beam_options.py`](agy_gemini_3.8_flash/scripts/test_beam_options.py) |
| **C03** | P0 | No CUDA return code is checked; `np.empty()` outputs; size args truncate at 2³¹; NVRTC compile logs discarded | O5 | O5 | [`a07`](claude_code_opus_5/a07_cuda_error_handling.txt), [`a18`](claude_code_opus_5/a18_size_arg_truncation.txt), [`a19`](claude_code_opus_5/a19_verify_cuda_check.txt) |
| **C04** | P0 | `config=None` → CUDA substitutes placeholder physics (`row_ss=0.5`, `bw=1.5`); PyTorch raises | O5 | O5 | [`a07`](claude_code_opus_5/a07_cuda_error_handling.txt) |
| **C05** | P1 | The backend "parity" test compares a host-side scalar, never the output arrays, and runs only one weighting | O5 | O5 | [`test_backend_parity_reference.py`](claude_code_opus_5/test_backend_parity_reference.py) |
| **C06** | P1 | `suppression_db` measures peak-residual convergence, not sidelobe suppression; a delta PSF scores 33.44 dB vs 33.98 dB for correct physics | both | both | [`a04`](claude_code_opus_5/a04_metric_validity.txt), [`a10`](claude_code_opus_5/a10_real_sidelobe_suppression.txt) |
| **C07** | P1 | `cuCtxCreate_v2` creates a non-primary context and never restores; breaks PyTorch after a CUDA run, and blocks the C05 fix | both | O5 | [`a09`](claude_code_opus_5/a09_context_conflict.txt), [`a17`](claude_code_opus_5/a17_context_strategy.txt), [`a17b`](claude_code_opus_5/a17b_context_fixes.txt) |
| **C08** | P1 | `ImageData.FirstRow`/`FirstCol` ignored when building the config; the tool's own written chips trigger it → 480 m error | **G** | O5 | [`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt) |
| **C09** | P1 | NVRTC `--gpu-architecture` hardcoded to `compute_86`; cannot load on A100/T4/V100 | **G** | O5 | [`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt), [`test_cuda_arch.py`](agy_gemini_3.8_flash/scripts/test_cuda_arch.py) |
| **C10** | P1 | Two tests assert their own inline algebra and never call the library; they miss the very regressions they were written to prevent | O5 | O5 | [`a15`](claude_code_opus_5/a15_regression_bite.txt) |
| **C11** | P1 | No test validates deconvolution against ground truth; the existing assertion is satisfied by any subtraction | O5 | O5 | [`a16`](claude_code_opus_5/a16_recovery_bite.txt) |
| **C12** | P1 | Grid metadata consumed without validation; the six DiffPFA products declare `ImpRespWid = 1/BW` where SICD requires the half-power width `0.886/BW` → restoring beam 1.129× too wide | O5 | O5 | [`a11`](claude_code_opus_5/a11_normalization_and_wgt.txt), [`a12`](claude_code_opus_5/a12_sicd_standard_check.txt), [`a13`](claude_code_opus_5/a13_diffpfa_irw_check.txt) |
| **C13** | P2 | Device allocations not exception-safe; 0 `try`/`finally`, 8 `cuMemFree_v2` on the straight-line path | **G** | O5 | [`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt) |
| **C14** | P2 | `demo_clean.py` (the README quickstart) crashes; `--target` flag documented but absent; output paths wrong | both | both | [`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt) X5 |
| **C15** | P2 | Deconvolved products inherit the original `ImageCreation/Application` and `DateTime` | **G** | O5 | [`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt) X4 |
| **C16** | P2 | Benchmark takes one cold sample, omits the `total_gpu_speedup` column where CUDA loses, and excludes ~5 s of NVRTC startup from CUDA timings only | O5 | O5 | [`a06`](claude_code_opus_5/a06_benchmark_methodology.txt), [`a21`](claude_code_opus_5/a21_warmup_methodology.txt) |
| **C17** | P2 | Mainlobe clean beam is flood-filled to a hard edge — 13 of 4225 pixels, stepping from −11.2 dB straight to zero | **G** | O5 | [`a23`](claude_code_opus_5/a23_verify_remaining_claims.txt) |
| **C18** | P2 | ~600 lines of dead C++/CUDA (`libcleansar.so` path) never built or loaded; error message instructs users to build it | both | both | source inspection, both reports |
| **C19** | P3 | Test suite hardcodes `/home/feildaw/data` and asserts GPU availability; cannot run elsewhere or in CI | O5 | O5 | [`AUDIT_REPORT.md` F14](claude_code_opus_5/AUDIT_REPORT.md) |
| **C20** | P3 | "Exact spatially-varying IPR" is a 0.26 % effect at spaceborne range; correct physics, but mis-scoped as the central differentiator | O5 | both | [`a20`](claude_code_opus_5/a20_spatial_variation_regime.txt), [`test_psf_spatial_variation.py`](agy_gemini_3.8_flash/scripts/test_psf_spatial_variation.py) |

---

## 3. The P0 findings in detail

### C01 — Hamming/Hann PSF normalisation (both audits)

`psf.py::compute_psf` divides by the PSF centre value; `clean_hogbom.cu::eval_1d_window_sinc`
does not. Uniform and Taylor have centre 1.0 either way, so they agree; Hamming's
is 0.54² = 0.2916 and Hann's 0.25.

| Weighting | psf.py centre | `.cu` centre | ratio | rel L2 of `clean_image` | identical peaks |
|---|---|---|---|---|---|
| UNIFORM | 1.000000 | 1.000000 | 1.00× | 3.6e-07 | 800/800 |
| TAYLOR | 1.000000 | 1.000000 | 1.00× | 3.8e-07 | 800/800 |
| **HAMMING** | 1.000000 | 0.291600 | **3.43×** | **1.29** | **62/800** |
| **HANN** | 1.000000 | 0.250000 | **4.00×** | **1.13** | **97/800** |

**Which side is correct is not a matter of taste.** Högbom subtracts `gain·amp·PSF`
but books `gain·amp` as extracted flux; those agree only if `PSF(0,0) = 1`. With
synthetic targets of known amplitude under Hamming:

| Backend | recovered / true | iterations |
|---|---|---|
| pytorch | 0.999×, 0.999×, 0.998× ✅ | 185 |
| cuda | 3.426×, 3.424×, 3.421× ❌ | 657 |

Gemini reached +10.64 dB independently (= 20·log₁₀ 3.426) and the same 3.5×
convergence penalty. **Fix the kernel, not `psf.py`.**

Latent today only because all 12 datasets are uniform-weighted.

### C02 — Silently ignored options (both audits)

`backend="auto"` resolves to CUDA whenever NVRTC is present, so these are the
**default** paths.

| Option | PyTorch | CUDA |
|---|---|---|
| `beam_type="mainlobe"` | 26.50 % of peak difference vs. gaussian | **bit-identical to gaussian (0.000000)** |
| `clean_mask` | 0 components outside mask | **53 components outside mask** |
| `psf_generator` | used | **custom generator called 0 times** |

The aggravating factor is the log line, which prints `Beam: MAINLOBE` while
evaluating a Gaussian. Wrong output is recoverable; wrong output accompanied by
confirmation that the requested setting was applied is not.

**`NotImplementedError` now beats implementing the features later.** `backend="pytorch"`
remains a working fallback for all three.

### C03 — Silent CUDA failure (Opus 5)

No return code checked anywhere in `run_hogbom_cuda_native`; outputs are `np.empty()`:

```
cuMemAlloc_v2(64 GiB)      -> return code 2 (FAILURE), ptr = None    [ignored]
cuMemcpyDtoH_v2 from NULL  -> return code 1 (FAILURE)                [ignored]
host buffer now contains:  [3.1967064e-11+0.j  0.+0.j  3.2856828e-11+0.j]
```

Small plausible floats, returned with a healthy `suppression_db` — because that
value is computed host-side from `history_peaks` and never touches the device result.

A concrete trigger exists: `ctypes` calls carry no `.argtypes`, so size arguments
marshal as **C int (32-bit)**. Measured on a card with 6.95 GiB free — bare int vs.
`c_size_t`: 1.5 GiB SUCCESS/SUCCESS, **2.0 GiB OUT_OF_MEMORY/SUCCESS**, 3.0 GiB
OUT_OF_MEMORY/SUCCESS. Buffers are `H·W·8` bytes, so scenes above **268 Mpixels**
cross the boundary; the largest here is 83 Mpixels.

Third defect in the same place: NVRTC compile errors return `False` without reading
`nvrtcGetProgramLog`, so a syntax error is indistinguishable from "no CUDA present" —
and surfaces as C18's misleading `libcleansar.so` message.

**Today's results are real** — the 72-Mpixel scene was validated end-to-end (all
finite, matching PyTorch to 1.2×10⁻⁷, 200/200 peak agreement). This is about margin.

### C04 — Fabricated physics on a missing config (Opus 5)

```
backend=pytorch : raised ValueError: Either 'config' or 'psf_generator' must be provided.
backend=cuda    : NO ERROR. iterations=100, suppression=0.70 dB
```

`cuda_backend.py` lines 205–208 and 273–276 fall back to `row_ss=col_ss=0.5`,
`row_bw=col_bw=1.5`, `sigma=0.5` — close enough to real values to look believable,
far enough to be wrong. Same call, same arguments: exception on one backend,
confident answer on the other.

---

## 4. Concerns that are not defects

Recorded so they are not lost, and so nobody "fixes" them by mistake.

**Wavefront-curvature defocus is not modelled** *(Gemini §1.2.3)*. Under PFA the
dominant space-variant degradation away from the SCP is quadratic/cubic phase
error, not rigid rotation. CLEAN_SAR models only the rotation. For already-refocused
inputs (DiffPFA) the rigid-sinc assumption is appropriate, so this is a scope
statement for the README rather than a defect — but it is the correct caveat, and
it pairs with C20: the term that *is* modelled is negligible at spaceborne range,
while the term that would not be is absent.

**The spatially-varying IPR is retained by project decision** *(C20)*. It is a
0.26 % effect on this data but reaches 5–25 % for airborne geometry (R₀ of tens of
km), and airborne data is anticipated. Any short-circuit must be an **opt-in fast
path, not a removal**. Regime map: `claude_code_opus_5/a20_spatial_variation_regime.txt`.

**SICD namespace versions are a version spread, not a defect** *(Gemini §2.1,
re-framed)*. Measured: 5 files declare `urn:SICD:1.2.1`, 7 declare `urn:SICD:1.3.0`.
All 12 fail the `0024-4_1.5` schema — including the 1.3.0 ones — because that is a
different version, not because the products are malformed. The seven 1.3.0 files
**PASS** the 1.3.0 schema. The 1.2.1 schema is not present in `schemas/`, so those
five cannot be checked here. No action for CLEAN_SAR beyond tolerating both
namespaces, which it already does. See `a23`.

**Remaining CUDA optimisation headroom** *(Gemini §3.1)*. The loop still performs
three synchronous `cuMemcpyDtoH_v2` per iteration. Keeping loop state on-device
(CUDA Graphs or a persistent kernel) is where the remaining latency sits.

**Architecture: `PSFGenerator` should not be a cross-backend abstraction**
*(Opus 5 §5.2, accepted by the project)*. PyTorch consumes a cached host-side PSF
provider; CUDA cannot, and evaluates the PSF inside the kernel. That mismatch is
the shared root of C01 (window maths re-implemented and diverged), C02
(`psf_generator` ignored) and C18 (implemented a third time in `psf_math.cuh`).
**Decision: `CleanPhysicsConfig` becomes the sole contract; each backend performs
its own PSF calculation from the spec; `psf_generator` leaves the public API.**
Full notes in [`claude_code_opus_5/AUDIT_REPORT.md` §5](claude_code_opus_5/AUDIT_REPORT.md).

---

## 5. Recommendations, in priority order

### Do first — stop the code reporting things that did not happen

Each is small, independent, and removes a way for a wrong result to look right.

1. **Raise `NotImplementedError`** on the CUDA path for `beam_type="mainlobe"`,
   `clean_mask` and `psf_generator`; remove them from the log unless honoured (C02).
   The capability-table form that makes this class of bug structurally impossible is
   in [`claude_code_opus_5/AUDIT_REPORT.md` §5.3](claude_code_opus_5/AUDIT_REPORT.md).
2. **Make `config=None` raise** on CUDA, as PyTorch does; delete the placeholder
   physics (C04).
3. **Adopt checked driver calls** — typed signatures, raising wrappers, NVRTC log
   capture. Drop-in: [`claude_code_opus_5/cuda_check.py`](claude_code_opus_5/cuda_check.py),
   verified by [`a19`](claude_code_opus_5/a19_verify_cuda_check.txt). Allocate outputs
   with `np.zeros`, not `np.empty` (C03).

### Correctness

4. **Normalise the PSF centre in `clean_hogbom.cu`.** Do *not* instead remove the
   normalisation from `psf.py` (C01).
5. **Use `cuDevicePrimaryCtxRetain`**, not `cuCtxCreate_v2`. Measured: 0.08 ms vs
   290 ms, no extra 134 MiB, and it is the only strategy that survives
   probe → torch → cuda → torch. Do **not** create/destroy a context per run — it
   breaks PyTorch. **Do this before #10**, which it blocks (C07).
6. **Default `chip_start` to `(handler.first_row, handler.first_col)`** (C08).
7. **Query device compute capability** for the NVRTC architecture flag (C09).
8. **Wrap allocation-to-cleanup in `try/finally`.** Land with #3, which makes the
   leak reachable (C13).
9. **Validate `ImpRespWid × ImpRespBW`** against the declared/known window on load
   and warn on mismatch — do not silently rewrite a product's stated physics.
   Separately, fix the DiffPFA writer to emit `0.886/BW` (C12).

### Test coverage

Four drop-in files, each demonstrated to **fail** when the corresponding defect is
present rather than merely to pass today.

10. [`test_backend_parity_reference.py`](claude_code_opus_5/test_backend_parity_reference.py)
    — array-level, all four weightings. Catches C01 and C02 together. **Highest
    value change in this report.** Depends on #5 (C05).
11. [`test_clean_recovery_reference.py`](claude_code_opus_5/test_clean_recovery_reference.py)
    — synthetic ground-truth recovery, both backends; verified by
    [`a16`](claude_code_opus_5/a16_recovery_bite.txt) (C11).
12. [`test_psf_physics_reference.py`](claude_code_opus_5/test_psf_physics_reference.py)
    — every assertion reads the library; verified by
    [`a15`](claude_code_opus_5/a15_regression_bite.txt) to catch all four historical
    regressions the originals miss (C10).
13. Parametrise the test data path; `pytest.skip` when NVRTC is absent (C19).

### Reporting, provenance and docs

14. Report mainlobe preservation and integrated sidelobe ratio; rename
    `suppression_db` → `peak_reduction_db`. Drop-in:
    [`quality_metrics.py`](claude_code_opus_5/quality_metrics.py) (C06).
15. Benchmark protocol: one discarded warm-up per backend **per batch**, ≥5
    interleaved pairs per file, report medians, print `total_gpu_speedup`, add
    `cuda_startup_ms`. Cold single-sample measurement currently inflates the
    headline by 8.5 % (C16).
16. Set `ImageCreation/Application` and `DateTime` on write (C15).
17. Fix `demo_clean.py`'s `plot_comparison` call (use the array form already in
    `cli.py`), add the documented `--target` flag, reconcile output paths (C14).
18. Delete the dead `c_src` files and correct the two `libcleansar.so` strings. If
    instead wiring them up as an AOT fallback, first resolve their drift from
    `clean_hogbom.cu` and add that path to the #10 parity matrix (C18).
19. Taper the mainlobe beam, or document the hard cutoff (C17).
20. Scope the spatially-varying IPR in the README: 0.26 % spaceborne, 5–25 %
    airborne; retained by decision (C20).

---

## 6. Where the audits differed, and how it was resolved

| Point | Gemini 3.8 Flash | Claude Opus 5 | Resolution |
|---|---|---|---|
| `demo_clean.py` failure | "fails immediately upon execution"; `TypeError: bad operand type for abs(): 'str'` | deconvolution **and** NITF write complete first; error is the keyword mismatch, raised before the function body runs | **Measured** ([`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt) X5): Opus 5's account. Same root cause, different triage — the compute is not lost, only the figure |
| Compute speedup | 5.09× (512×512 chip, 500 iters, cold single sample) | 4.1× (full scenes); 4.14× published | **Not a conflict** — different conditions. A small chip favours CUDA (PyTorch's fixed per-iteration overhead is a larger fraction), and a cold sample inflates by ~8.5 % ([`a21`](claude_code_opus_5/a21_warmup_methodology.txt)). Both figures are real for their conditions |
| `compute_86` scope | fails on A100, V100, T4, **and H100** | H100 would work | **Measured** ([`a22`](claude_code_opus_5/a22_crosscheck_other_audit.txt) X2): PTX is forward-compatible, so compute_86 JITs onto **newer** devices. A100/T4/V100 fail; H100 does not |
| CUDA context | P2, "clarity" | P1 — reproducible runtime failure that also blocks the parity fix | **Measured** ([`a09`](claude_code_opus_5/a09_context_conflict.txt)): raised to P1 as C07 |
| Metric rename | `residual_reduction_db` | `peak_reduction_db` | Cosmetic; either is accurate |

No finding from either audit was contradicted by the other. The differences above
are scope and severity, all resolved by measurement.

---

## 7. Artifact index

Neither audit's code is reproduced in this document. Both subdirectories are
self-contained and runnable from the repository root with
`/home/feildaw/mypyenv/bin/python`.

### `claude_code_opus_5/`

- **[`AUDIT_REPORT.md`](claude_code_opus_5/AUDIT_REPORT.md)** — 19 findings, architecture notes, cross-check section
- **[`README.md`](claude_code_opus_5/README.md)** — layout and how to run
- `a01`–`a23` — investigation scripts with captured `.txt` output
- **Drop-in deliverables**, each verified to fail on the defect it guards:
  [`quality_metrics.py`](claude_code_opus_5/quality_metrics.py),
  [`cuda_check.py`](claude_code_opus_5/cuda_check.py),
  [`test_backend_parity_reference.py`](claude_code_opus_5/test_backend_parity_reference.py),
  [`test_psf_physics_reference.py`](claude_code_opus_5/test_psf_physics_reference.py),
  [`test_clean_recovery_reference.py`](claude_code_opus_5/test_clean_recovery_reference.py)

### `agy_gemini_3.8_flash/`

- **[`AUDIT_REPORT.md`](agy_gemini_3.8_flash/AUDIT_REPORT.md)** — 10 numbered defects plus standards and architecture analysis
- `scripts/` — `test_hamming_parity.py`, `test_beam_options.py`, `test_cuda_arch.py`,
  `test_point_target_recovery.py`, `test_psf_math.py`, `test_psf_spatial_variation.py`,
  `test_offgrid_target.py`, `test_pfa_polar_angle.py`, `test_sicd_compliance.py`,
  `benchmark_audit.py`, `inspect_datasets.py`

### Scope note

**Neither audit modified anything under `clean_sar/`.** Where a defect had to be
reintroduced to prove a test catches it, it was done by runtime monkey-patching
inside a subprocess. `audit/` is no longer `.gitignore`d (it was until 2026-09-05),
so this directory is now tracked normally.

---

*Combined report compiled by Claude Opus 5 (Anthropic), running in Claude Code, from two
independently conducted audits. Every merged claim carries a verification status; findings
originating in the Gemini 3.8 Flash audit (C08, C09, C13, C15, C17) were reproduced before
being recorded here.*
