# CLEAN_SAR — Independent Audit

**Auditor:** Claude Opus 5 (Anthropic), running in Claude Code
**Audit directory:** `audit/claude_code_opus_5/`
**Date:** 2026-09-03 → 2026-09-05
**Independence:** conducted without reference to any other agent's audit of this repository
**Subject:** `/home/feildaw/CLEAN_SAR` working tree as-is (uncommitted refactor; last commit `f6734a1`)
**Scope:** purpose and approach; mathematical correctness; implementation and test integrity
**Environment:** `/home/feildaw/mypyenv` (Python 3.12.3, torch 2.6.0+cu124, numpy 2.4.1, sarkit 1.8.0); NVIDIA RTX 3070 Laptop (compute 8.6, 8 GiB), driver 555.51, WSL2
**Reproduction code:** `a01`–`a22` (`.py` sources, `.txt` captured outputs), plus five drop-in deliverables: `audit/claude_code_opus_5/quality_metrics.py` (F2), `audit/claude_code_opus_5/test_backend_parity_reference.py` (F4), `audit/claude_code_opus_5/test_psf_physics_reference.py` (F5), `audit/claude_code_opus_5/test_clean_recovery_reference.py` (F6) and `audit/claude_code_opus_5/cuda_check.py` (F7/F8). Nothing under `clean_sar/` was modified.

---

## Executive summary

The physics and the compute path are sound.

The CUDA backend is a genuine native implementation — NVRTC compiles `clean_hogbom.cu` to PTX and four kernels launch through the CUDA Driver API. On the weightings the real data uses, it agrees with the PyTorch backend to float32 rounding and selects **identical peaks on 800/800 and 200/200 iterations**, including on a 72-Mpixel full scene. The published benchmark speedups reproduce: I measured 4.1× where the committed CSV reports 4.14×.

The mathematics verifies independently. The analytic IPR matches a numerical Fourier transform of the windowed aperture to ~10⁻⁹, the Taylor coefficients match `scipy` exactly, and the Gaussian restoring-beam width convention is correct. Given a matching forward model, CLEAN recovers synthetic point targets to **0.1–0.6 % amplitude error with 100 % of flux on true pixels**. Every finding from the earlier multi-model audit has been properly fixed.

The findings concentrate in one area: **the CUDA and PyTorch backends have diverged in ways nothing currently measures.** Hamming/Hann weighting makes the two backends produce entirely different images (F1). `config=None` yields placeholder physics on CUDA where PyTorch raises (F11). Failed CUDA calls return uninitialised memory as results (F8).

**The most serious of these is F3.** Three documented options — `beam_type="mainlobe"`, `clean_mask`, and `psf_generator` — are accepted by the default CUDA backend and silently ignored, and one of them is affirmatively reported as active: the run logs `Beam: MAINLOBE` while evaluating a Gaussian. Wrong output is recoverable; wrong output accompanied by confirmation that the requested setting was applied is not. The remedy is cheap — raise `NotImplementedError` rather than substitute — and should not wait on implementing the features. The underlying rule, which also covers F8 and F11: **never log, return, or record a capability that did not run.**

These are invisible because the only cross-backend comparison in the repo — in the benchmark, the CSVs, and `test_cuda_backend_execution_parity` — is a scalar `suppression_db` computed host-side from the peak history, which never compares the output arrays. That metric is also weakly diagnostic on its own: on the showcase chip, **a delta-function PSF scores 33.44 dB against correct physics' 33.98 dB.**

Fixing the parity test to compare arrays across all four weightings catches most of this in one change. A verified replacement ships as `audit/claude_code_opus_5/test_backend_parity_reference.py`; it depends on F7 being fixed first.

§6 records a later cross-check against a second, independent audit: it corroborates the central physics defect by a different route, and contributed four further defects (F16–F19) that this audit missed and has since reproduced.

---

## 1. What this project is

CLEAN_SAR performs **complex Högbom CLEAN deconvolution** on SAR imagery in NGA NITF SICD format.

Classical radio-astronomy CLEAN assumes a real, positive intensity map and a shift-invariant PSF. SAR breaks both: the data is complex (I+jQ, phase carries interferometric and coherent-scattering information), and under Polar Format Algorithm image formation the effective k-space support rotates slightly with scene position, so the impulse response (IPR) is nominally spatially varying.

```
NITF SICD ──► SICDHandler (sarkit)      metadata + chip/full-scene complex64 I/O
          ──► CleanPhysicsConfig        13 plain scalars, POD-struct shaped for C/CUDA
          ──► PSFGenerator              analytic IPR: modulated-sinc series, rotated by
                                        theta(x,y) = atan2(y_col, R0 + x_row); LRU cached
          ──► run_hogbom_clean ─┬─► pytorch_backend   torch loop, numpy PSF per iteration
                                └─► cuda_backend      NVRTC → PTX → Driver API, 4 kernels
          ──► SICDHandler.write_nitf    compliant SICD NITF out
```

Decoupling the physics into a flat scalar `CleanPhysicsConfig` that maps onto a CUDA POD struct is a good design decision, and it is what makes the two backends comparable at all.

---

## 2. What is correct — verified, not assumed

| # | Claim | Verified by | Result |
|---|---|---|---|
| C1 | CUDA backend is native | `a05`, `a07` | NVRTC compiles PTX; 4 kernels resolved via `cuModuleGetFunction`; driver-API launches |
| C2 | Backends agree | `a05`, `a07` | UNIFORM/TAYLOR: `≤4×10⁻⁷` rel; **800/800** and **200/200** identical peak selections; 72 Mpix scene matches at `1.2×10⁻⁷` |
| C3 | Published speedups are real | `a06` | Measured **4.1×**; published **4.14×**. Low variance (pytorch 625–730 ms, cuda 165–173 ms) |
| C4 | Analytic IPR is exact | `a01` | vs. numerical inverse-FT of windowed aperture: max rel. error **5×10⁻⁹** (uniform), **7×10⁻¹⁰** (Hamming), **2×10⁻¹⁶** (Hann), **2×10⁻⁹** (Taylor) |
| C5 | Taylor coefficients | `a01` | `[0.29265601, -0.01578375, 0.00218104]` = nbar 4, SLL −30 dB, matching `scipy.signal.windows.taylor` and the repo's own `utils.taylor_window_1d` to **1×10⁻⁹** |
| C6 | Gaussian beam width | `a01` | σ = Wid/(2√ln2) gives a −3 dB half-power width ratio of **1.000000**. Correct for an amplitude beam |
| C7 | CLEAN algorithm | `a04` | Synthetic ground truth: 5 targets recovered to **0.10 %–0.57 %** amplitude error, **100 %** of flux on true pixels, exactly 5 non-zero component pixels |
| C8 | Coordinate math | `a02` | Naive `config.global_to_metric` vs. rigorous `sarkit.rowcol_to_xrowycol`: **0 m** disagreement on all 12 files |
| C9 | Real-valued PSF is justified | `a03` | Grid Row KCtr = 64.04 cyc/m (X-band carrier) but the **sampled image is baseband** (measured support centre `+0.00000` cyc/m), so a real symmetric IPR is the right model |
| C10 | Prior audit findings fixed | `a01`, `a02` | Hardcoded 10 km slant range → from `SCPCOA.SlantRange`; √2 Gaussian error → fixed; Taylor bug → fixed; unbounded cache → bounded LRU; `write_nitf` mutation → `deepcopy`; version mismatch → both 0.2.0 |

---

## 3. Findings

Severity reflects impact on trusting a result.

### 🔴 F1 — Hamming/Hann: the backends compute different PSFs, off by 3.43× and 4.0×

`psf.py::compute_psf` normalises the PSF to 1.0 at centre (lines 121–124). `clean_hogbom.cu::eval_1d_window_sinc` does not. For uniform and Taylor the centre value is 1.0 either way, so they agree. For Hamming the 1-D centre is 0.54 → 2-D centre 0.2916; for Hann, 0.25.

| Weighting | psf.py centre | `.cu` centre | ratio | suppression Δ | rel L2 of `clean_image` | same peaks |
|---|---|---|---|---|---|---|
| UNIFORM | 1.000000 | 1.000000 | 1.00× | 0.000001 dB | 3.6e-07 | 800/800 |
| TAYLOR | 1.000000 | 1.000000 | 1.00× | 0.000001 dB | 3.8e-07 | 800/800 |
| **HAMMING** | 1.000000 | 0.291600 | **3.43×** | **4.712 dB** | **1.29** | **62/800** |
| **HANN** | 1.000000 | 0.250000 | **4.00×** | **3.628 dB** | **1.13** | **97/800** |

Under Hamming the backends diverge at **iteration 2** and produce unrelated images (rel L2 ≈ 1.3 means the difference is as large as the signal). CUDA under-subtracts by 3.43×, so CLEAN never converges properly.

All 12 datasets are uniform-weighted, so this path is never exercised. Note the adjacent fragility: 11 of 12 files declare `WgtType = None` and the code defaults to `"UNIFORM"` — correct here, but a default rather than a reading. See F15.

**`psf.py` is right and the kernel is wrong — a correctness question, not a consistency question.** Högbom subtracts `gain·amp·PSF` from the residual but records `gain·amp` in the components map. Those agree only if `PSF(0,0) = 1`. With an unnormalised Hamming PSF the loop removes `0.2916·comp` from the image while booking `comp` of flux, and the restoring beam — whose peak *is* 1.0 — adds the full `comp` back. Photometry breaks.

Synthetic ground truth, Hamming weighting, known amplitudes (`a11`):

| Backend | recovered / true amplitude | iterations |
|---|---|---|
| pytorch | **0.999×, 0.999×, 0.998×** ✅ | 185 |
| cuda | **3.426×, 3.424×, 3.421×** ❌ | 657 |

The CUDA error is exactly the 3.4294 normalisation factor. Recovered amplitudes are inflated 3.4×, and the loop needs 3.5× the iterations because it under-subtracts each step.

**Fix:** divide out the centre value in the `.cu` kernel. Do *not* instead remove the normalisation from `psf.py`.

### 🔴 F2 — `suppression_db` is weakly diagnostic

`CleanResult.suppression_db = 20·log10(history_peaks[0] / history_peaks[-1])` — the ratio of first to last residual peak. It measures how far the greedy peak-chase descended. It is labelled **"Sidelobe Suppression (dB)"** in `run_benchmark.py`, quoted by the demo, CLI, README and both CSVs, and is the sole criterion in the backend parity test.

On the showcase chip (`demo_clean.py::UMBRA_CHIP`), `a10`:

| Variant | reported "suppression" | TRUE sidelobe energy change | mainlobe peak preserved |
|---|---|---|---|
| **Correct physics** | **33.98 dB** | −4.46 dB | **0.988×** ✅ |
| **Delta PSF (no model at all)** | **33.44 dB** | −5.06 dB | **1.506×** ❌ |
| Bandwidth ×2 (wrong) | 31.58 dB | −5.56 dB | 2.703× ❌ |
| Bandwidth ÷2 (wrong) | 7.11 dB | +4.05 dB | 15.05× ❌ |

A delta-function PSF — subtracting a bare spike, no deconvolution — scores within 0.54 dB of correct physics, and removes *more* annulus energy, because it punches holes rather than deconvolving. Broader sweep on a different chip (`a04`): correct 33.98 dB, delta PSF 31.00 dB, sample-spacing×10 30.97 dB.

Note the trap in the middle column: **integrated sidelobe reduction alone is also insufficient** — the delta PSF removes *more* annulus energy (−5.06 vs −4.46 dB), because punching a hole does reduce surrounding energy. It just destroys the target while doing it.

The **pair** is what discriminates: correct physics is the only variant that both reduces sidelobes and leaves the mainlobe intact (0.988×).

#### The metric, defined

Locate a bright scatterer `(r0,c0)` in the **dirty** image and use that same pixel for both images. With `rad` the pixel radius from it:

```
resolution cell (px)   cells = max(row_wid/row_ss, col_wid/col_ss)
mainlobe core          rad <= R_in,  R_in = ceil(1.5 * cells)
sidelobe annulus       R_in < rad <= R_out       (R_out = 40 px)

P_pk(img) = max |img|^2  over core
E_sl(img) = sum |img|^2  over annulus

islr_db(img)          = 10*log10( E_sl / P_pk )
islr_change_db        = islr_db(clean) - islr_db(dirty)      negative is good
mainlobe_preservation = max|clean| / max|dirty|  over core   1.0 is ideal
```

Pass condition: `|mainlobe_preservation - 1| <= 0.10` **and** `islr_change_db < 0`.

`islr_db` here is an integrated-sidelobe-to-peak ratio over a 2-D annulus, not the 1-D cut ISLR of the SICD IPDD. It is used for the *change* between two images of the same scene, where the common normalisation cancels.

**A ready-to-use implementation is `audit/claude_code_opus_5/quality_metrics.py`** — numpy-only, no clean_sar imports, intended to be lifted into `clean_sar/quality.py` roughly verbatim. It provides `ipr_quality()`, `ipr_quality_multi()` (averaged over the brightest isolated scatterers), `find_bright_targets()` and `verdict()`. `a14` verifies it reproduces the table above exactly and that identical images give `mainlobe=1.000000, islr_change=+0.000000`.

Two caveats for whoever integrates it:

- **`ipr_quality_multi` is built for full scenes, not chips.** `find_bright_targets` requires a candidate to exceed 6× the brightest pixel in its surrounding 41×41 neighbourhood (excluding the central 11×11), and to sit ≥40 px from the edge. On the 256×256 showcase chip only **one** target qualifies, so `n_targets=1` and the reported `±std` is meaningless. Either relax `isolation`/`edge` for small chips or use single-target `ipr_quality` there and check `n_targets` before trusting the spread.
- **`verdict()` requires strict improvement: `islr_change_db < 0`.** So `clean == dirty` returns `SIDELOBES NOT REDUCED (+0.00 dB)`, not `OK`. That is deliberate — a no-op must not pass — but it surprises people who run it on an unchanged image as a smoke test.

**Fix:** rename `suppression_db` → `peak_reduction_db` (it is a fine convergence diagnostic, just not a quality metric), and report `mainlobe_preservation` + `islr_change_db` alongside it.

### 🔴 F3 — Three documented options are silently ignored by the default backend, and one is affirmatively reported as active

**This is the most serious class of defect in the codebase.** The others produce wrong numbers; this one produces wrong numbers *and tells the operator its settings were applied*. A user who requests a mainlobe restoring beam, reads `Beam: MAINLOBE` in the log, and archives the product has no signal anywhere that they received a Gaussian instead.

`backend="auto"` resolves to `"cuda"` whenever NVRTC is available, so these are the **default** paths, not an obscure corner.

| Option | PyTorch | CUDA | Evidence |
|---|---|---|---|
| `beam_type="mainlobe"` | 26.50 % of peak difference vs. gaussian | **bit-identical to gaussian (0.000000)** while logging `Beam: MAINLOBE` | `a08` |
| `clean_mask` | 0 components outside mask | **53 components outside mask** | `a05` |
| `psf_generator` | used | **custom generator called 0 times**, run completes and reports 27.40 dB | `a08` |

Causes, all omissions rather than errors:

- `clean_hogbom.cu` has **no mainlobe branch at all** — the fused kernel only ever calls `eval_clean_beam_gaussian`.
- `clean_mask` is never uploaded to the device. (`guard_margin` *is*, and is honoured correctly by both backends — verified.)
- `run_hogbom_cuda_native` accepts `psf_generator` at `cuda_backend.py:139` and never references it again in the function body.

The log line is the part that turns three missing features into a correctness problem:

```
Backend: CUDA (Native C++/NVRTC) | Beam: MAINLOBE | PSF size: 65x65
```

printed while evaluating a Gaussian beam.

**Fix — `NotImplementedError` is strongly preferred over silent substitution, even though it is the less capable option.** Raising costs nothing, is honest, and leaves `backend="pytorch"` as a working fallback for all three options today. Implementing them properly is the better long-term answer, but shipping the raise first is strictly superior to the current behaviour and should not wait on the implementation.

The general rule this violates, and which is worth applying across the codebase: **never log, return, or record a capability that did not run.** Where a backend cannot honour a requested parameter it must refuse, not substitute. This same principle covers F11 (`config=None` substituting placeholder physics rather than raising) and F8 (returning uninitialised memory rather than reporting a CUDA failure).

### 🟠 F4 — The parity test cannot detect a parity failure

`tests/test_backends.py::test_cuda_backend_execution_parity`, docstring *"Verify native C++/CUDA backend produces equivalent results to PyTorch backend"*:

```python
assert abs(res_torch.suppression_db - res_cuda.suppression_db) < 1.0
```

It never touches `clean_image`, `residual_image`, `components_map`, or `restored_model`. Per F1 the backends can differ by rel L2 ≈ 1.3 — and this test's metric *would* flag that at 4.71 dB — but only Hamming/Hann trigger it, and no test exercises them. On the uniform path the assertion is satisfied by a quantity that cannot fail.

The same limitation reaches the benchmark: `delta_suppression_db = 0.0000` in all 12 published rows, printed as "Suppression Parity Delta." That value is host-side and would remain 0.0000 even if the device returned uninitialised memory (F8).

#### The principle

**A parity test must compare the payload, not a summary derived from a different code path.** `suppression_db` is computed on the host from `history_peaks`, alongside rather than from the device result, so it cannot witness a device fault by construction. Any quantity computed before, beside, or independently of the thing under test is unfit to validate it — however reasonable it looks on the result object.

The corollary is that a test must not inherit the assumptions of the code it tests. This test is only convincing *because* it passes, and it passes for a real reason (uniform parity is genuinely excellent). Prior correctness is not evidence of current correctness.

#### The fix

**A runnable replacement is `audit/claude_code_opus_5/test_backend_parity_reference.py`** — lift into `tests/` roughly verbatim. It replaces the scalar assertion with array-level comparison, parametrised over every supported weighting:

```python
@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_backend_parity_arrays(chip_and_config, wgt):
    cfg = _with_weighting(base, wgt)
    rt, rc = _run(chip, cfg, "pytorch"), _run(chip, cfg, "cuda")

    assert rt.iterations == rc.iterations
    # peak selection must match step for step
    first_div = next((i for i in range(n)
                      if rt.history_coords[i] != rc.history_coords[i]), None)
    assert first_div is None
    # the payload itself
    for field in ("clean_image", "residual_image", "components_map", "restored_model"):
        a, b = getattr(rt, field), getattr(rc, field)
        assert np.max(np.abs(a - b)) / (np.max(np.abs(a)) or 1.0) < 1e-5
```

It also adds `test_psf_peak_is_unity` (the F1 root cause) and three option-honouring tests. **Verified against the current tree** — it produces exactly the intended result:

| Case | Result | Catches |
|---|---|---|
| `parity[UNIFORM]`, `parity[TAYLOR]` | PASS | genuine agreement |
| `parity[HAMMING]`, `parity[HANN]` | **FAIL** — `peak selection diverges at iteration 1: pytorch=(65,76) cuda=(96,92)` | **F1** |
| `psf_peak_is_unity[*]` | PASS ×4 | confirms `psf.py` is the correct side |
| `mainlobe_beam` | **FAIL** — `bit-identical to gaussian` | **F3** |
| `clean_mask` | **FAIL** — `140 components placed outside the mask` | **F3** |
| `requires_physics_config` | **FAIL** — did not raise | **F11** |

> ⚠️ **This fix is blocked by F7.** The reference file needs a two-line workaround at import (`torch.zeros(1, device="cuda")` before the NVRTC probe). Without it, three cases fail with `RuntimeError: CUDA driver error: invalid resource handle` — spurious failures that *look* like PSF divergence but are the driver-context bug. Any pytest module that probes for CUDA at import and then alternates backends hits this, which is exactly what a parity test must do. **Fix F7 first, then delete the workaround.**

### 🟠 F5 — Two tests contain no library code in their central assertions

`test_slant_range_and_restoring_beam_physics` re-derives the Gaussian formula inline and asserts its own algebra:

```python
factor_3db = 2.0 * np.sqrt(np.log(2.0))
sigma_r = wid_r / factor_3db                  # not read from psf.py
amp_at_half = np.exp(-0.5 * (u_half / sigma_r) ** 2)
assert np.isclose(power_db, -3.0103, atol=1e-4)
```

`compute_clean_beam` is never called. Reverting `psf.py` to the old buggy `2√(2 ln 2)` still passes this test.

`test_psf_taylor_continuous_window` operates entirely on a locally-defined `fm` list and asserts `0.0 < w_edge < 1.0`. It imports nothing from `psf.py` and would pass against an empty library.

`test_psf_analytic_windowing` is real but weak: it checks Taylor sidelobes < uniform sidelobes in a hardcoded slice, while its docstring cites levels (−13.2 dB vs −30 dB) it never verifies.

Both of the self-referential tests were written to lock in fixes from the *earlier* audit, which is what makes this consequential: the implementation is correct today (verified independently in `a01`), so these tests exist solely to detect a future regression — and they cannot.

#### The principle

**A test must exercise the library and compare against a value derived independently of it** — a physical constant, a closed form, or a separately computed transform. A test that recomputes the implementation and compares it to itself measures nothing. Passing is not evidence; a test is only worth its runtime if you know it can fail.

#### The fix

**`audit/claude_code_opus_5/test_psf_physics_reference.py`** — drop-in replacement, lift into `tests/`. Every assertion reads a value out of `clean_sar` and checks it against physics:

- `test_slant_range_comes_from_scpcoa` — compares `config.scp_slant_range` to an independent read of `SCPCOA/SlantRange` from the XML.
- `test_gaussian_beam_halfpower_width_matches_imprespwid` — calls `compute_clean_beam`, recovers σ from the returned array (`σ = ss / sqrt(-2 ln(beam[c+1]/beam[c]))`, exact for a Gaussian, no fitting), and asserts `2σ√(ln2) == ImpRespWid`.
- `test_dirty_psf_matches_independent_fft_ipr` — compares `compute_psf` against the inverse FFT of a rect aperture, a different construction from the analytic sinc series.
- `test_psf_sidelobe_level_matches_window` — measures the peak sidelobe of the returned array against each window's defining spec (uniform −13.26, Taylor −30, Hamming −42.7, Hann −31.5 dB).
- `test_psf_peak_is_unity` — the F1 photometric requirement, all four windows.

#### Proof that it bites

`audit/claude_code_opus_5/a15_regression_bite.py` reintroduces each defect by monkey-patching `clean_sar` at runtime (no file under `clean_sar/` is touched) and runs both suites against the broken library:

| Reintroduced defect | reference suite | the two original tests |
|---|---|---|
| R1 Gaussian σ reverts to `2√(2 ln 2)` (earlier audit's Finding 2) | **CAUGHT** — 3 failed | **MISSED** — 2 passed |
| R2 slant range reverts to hardcoded 10 km (Finding 1) | **CAUGHT** — 1 failed | caught — 1 failed |
| R3 Taylor coefficients corrupted | **CAUGHT** — 2 failed | **MISSED** — 2 passed |
| R4 PSF centre normalisation removed (F1) | **CAUGHT** — 2 failed | **MISSED** — 2 passed |

Baseline on the unmodified library: reference suite **14 passed**, originals 2 passed.

The originals catch only R2 — the one assertion in either of them that actually reads the library (`config.scp_slant_range > 100_000`). They miss the two regressions they were specifically written to prevent.

### 🟠 F6 — No test validates deconvolution against ground truth

`test_exact_spatially_varying_clean_synthetic` is named "synthetic" but reads a real SICD chip, and its only substantive assertion is:

```python
assert np.max(np.abs(res.residual_image)) < np.max(np.abs(chip))
```

which any subtractive procedure satisfies — including the delta PSF and wrong-bandwidth variants of F2.

Worth emphasising because **the code passes the real test easily**. `a04` builds a noise-free synthetic scene from the library's own forward model and recovers every target:

| True amplitude | Recovered | Relative error |
|---|---|---|
| 1.00 + 0.00j | +0.9990 − 0.0000j | 0.10 % |
| 0.60 − 0.30j | +0.5991 − 0.2996j | 0.15 % |
| 0.35 + 0.20j | +0.3491 + 0.1995j | 0.25 % |
| 0.25 + 0.10j | +0.2492 + 0.0997j | 0.34 % |
| 0.15 − 0.05j | +0.1491 − 0.0497j | 0.57 % |

with **100 % of CLEAN flux on true target pixels** and exactly 5 pixels receiving components — no spurious ones.

#### The fix

**`audit/claude_code_opus_5/test_clean_recovery_reference.py`** — drop-in replacement, lift into `tests/`. Parametrised over both backends:

- `test_recovers_true_complex_amplitudes` — known complex amplitudes recovered to < 2 %.
- `test_flux_lands_on_true_targets` — > 98 % of component flux on true sites.
- `test_no_spurious_components` — exactly one component pixel per scatterer.
- `test_residual_is_driven_to_threshold` — replaces the trivial assertion: residual peak < 2×10⁻³ of the dirty peak *and* residual energy < 5 % on a noise-free scene.
- `test_backends_agree_on_synthetic_scene` — array-level, normalised by the **scene** peak rather than each array's own peak (after convergence the residual is ~10⁻³ of the scene, so self-normalising inflates ordinary float32 rounding into an apparent 8×10⁻⁵ "disagreement").

**Scope, stated deliberately:** the forward model is the library's own PSF, so this validates the **algorithm** — peak selection, complex subtraction, flux bookkeeping, restoration — *given* a correct PSF. It does **not** validate the PSF physics; `test_psf_physics_reference.py` (F5) does that against independent ground truth. Neither file alone is sufficient; together they close the loop.

#### Proof that it bites

`audit/claude_code_opus_5/a16_recovery_bite.py` patches the PSF the *algorithm* subtracts while leaving the forward model on the true analytic PSF — simulating a deconvolution kernel that disagrees with the real IPR:

| Broken deconvolution kernel | recovery reference | original test |
|---|---|---|
| S1 delta PSF (no model at all) | **CAUGHT** — 4 failed | **MISSED** — 1 passed |
| S2 PSF scaled by 0.2916 (the F1 error) | **CAUGHT** — 1 failed | **MISSED** — 1 passed |
| S3 PSF at 2× the true bandwidth | **CAUGHT** — 4 failed | **MISSED** — 1 passed |

Baseline on the unmodified library: both pass. The original misses all three, exactly as its assertion predicts — subtracting *anything* lowers the peak.

### 🟠 F7 — CUDA driver context conflict breaks PyTorch after a CUDA run

`_init_cuda_driver` calls `cuCtxCreate_v2` (a new, non-primary context) when none is current, then `cuCtxSetCurrent` on every CUDA run — and never restores the previous context.

Reproduced deterministically (`a09`, fresh interpreters):

| Scenario | Result |
|---|---|
| S1 torch → cuda → torch | ✅ (probe reuses torch's context) |
| S2 cuda → torch | ✅ |
| S3/S4 probe → torch | ✅ |
| **S6 probe → torch → cuda → torch** | ❌ **`RuntimeError: CUDA driver error: invalid resource handle`** |
| S7 same without leading probe | ✅ |

Trigger: if `resolve_backend("auto")` / `is_cuda_native_available()` runs *before* PyTorch initialises CUDA, the probe creates its own context; a later CUDA run switches to it and leaves it current; the next PyTorch GPU op fails. Mixed-backend use in one process is the failure case.

**This blocks the F4 fix.** A parity test must alternate backends, and a pytest module naturally probes for CUDA at import time to decide whether to skip — which is precisely the trigger. `audit/claude_code_opus_5/test_backend_parity_reference.py` needs a two-line workaround (touch `torch.cuda` before the probe, so `_init_cuda_driver` adopts torch's primary context) or three of its cases fail with `invalid resource handle`, spurious failures that mimic PSF divergence. Fix F7 and the workaround can be deleted. The committed benchmark is safe only because `resolve_backend("pytorch")` returns early without probing, so PyTorch always initialises first.

#### Which fix? Three strategies measured (`a17`, `a17b`)

All in fresh subprocesses, each running the failing sequence `probe → torch → cuda → torch → cuda`:

| Strategy | Result |
|---|---|
| **F-A — adopt the primary context** (`cuDevicePrimaryCtxRetain`) | **ALL PASS.** Every step runs in one context; the adopted handle is byte-identical to PyTorch's |
| F-B — keep an own context, restore the caller's after each run | **ALL PASS.** Correct, but costs a second context |
| F-C — create + destroy a private context per run | **FAILED** — torch dies at step 3 with `invalid resource handle` |

Supporting measurements:

| | |
|---|---|
| PyTorch's context vs `cuDevicePrimaryCtxRetain` | **identical handle** — torch uses the device primary context |
| `cuCtxCreate_v2` context | a *different* context, costing **134 MiB** of device memory |
| `cuDevicePrimaryCtxRetain` | **0.080 ms** |
| `cuCtxCreate_v2` + `cuCtxDestroy_v2` | **290 ms + 95 ms = 385 ms per run** |

**On tearing a context down to get a "fresh" one per implementation.** It is an appealing instinct — a cold start each time would exercise the whole init path — but it does not work in-process:

- **F-C breaks PyTorch.** Destroying the private context leaves a *different* context current from the one torch bound to, and the next torch op raises `invalid resource handle`.
- **Resetting the primary context is worse than an error.** With `cuDevicePrimaryCtxReset_v2` called under a live PyTorch, `(ones(1024) * 2).sum()` returned **1.85×10³³** instead of 2048 — torch kept running and produced silent garbage. That is the failure mode this audit most wants to avoid.
- It costs **385 ms and 134 MiB per run** for no correctness benefit.

**The right tool for cold-start coverage is process isolation, not context teardown.** A fresh subprocess gives a genuinely fresh context, module, and driver state, with no risk to a live PyTorch — which is exactly how `a09`, `a17` and `a17b` obtained reliable results here. Recommended: run the backend matrix in-process under F-A for speed, and add one subprocess-per-backend cold-start test for init coverage.

**Fix:** adopt **F-A** — replace `cuCtxCreate_v2` with `cuDevicePrimaryCtxRetain` so the native backend shares PyTorch's primary context (3 600× cheaper to establish, no extra 134 MiB, and pointer-compatible with torch should the backends ever need to exchange device memory). Restoring the prior context after each run, as in F-B, is a sound belt-and-braces addition.

### 🟠 F8 — No CUDA error checking; failures return uninitialised memory

No return code is checked in `run_hogbom_cuda_native` — not `cuMemAlloc_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuLaunchKernel`, or `cuCtxSynchronize`. Output buffers are `np.empty()`.

Demonstrated (`a07`):
```
cuMemAlloc_v2(64 GiB)          -> return code 2 (FAILURE), ptr = None     [ignored]
cuMemcpyDtoH_v2 from NULL      -> return code 1 (FAILURE)                 [ignored]
host buffer now contains:  [3.1967064e-11+0.j  0.+0.j  3.2856828e-11+0.j]
```

Small floats — a plausible-looking SAR residual. And since `suppression_db` derives from host-side `history_peaks`, a failed device run still reports a healthy number and `delta_suppression_db = 0.0000`.

**Currently latent, not active.** I validated the largest benchmarked scene (72 Mpix, 0.54 GiB × 4 buffers) end-to-end: all arrays finite, matching PyTorch to `1.2×10⁻⁷`, 200/200 peak agreement. The results are real. This is the mechanism by which they could stop being real on a busier GPU, a larger scene, or a smaller card.

#### A concrete trigger: 32-bit truncation of size arguments

The driver is called through `ctypes` without `.argtypes`, so a bare Python int is marshalled as a **C int (32-bit)** where the driver expects `size_t`:

```python
_CUDA_LIB.cuMemAlloc_v2(ctypes.byref(d_residual), bytes_complex)   # bytes_complex is a Python int
```

Allocating the same size both ways on a card with 6.95 GiB free (`a18`):

| Request | as bare Python int | as `ctypes.c_size_t` |
|---|---|---|
| 1.5 GiB | SUCCESS | SUCCESS |
| **2.0 GiB** | **OUT_OF_MEMORY** | **SUCCESS** |
| **3.0 GiB** | **OUT_OF_MEMORY** | **SUCCESS** |
| **4.0 GiB** | **INVALID_VALUE** | **SUCCESS** |

The break is exactly at 2³¹ — the size argument is being truncated at 32 bits.

**The compound failure.** Buffers here are `H·W·8` bytes, so a scene above **268 Mpixels** crosses the boundary. At that point `cuMemAlloc_v2` fails, the return code is not checked, the device pointer stays `NULL`, the copy back fails silently, and the caller receives `np.empty()` buffers — with a healthy `suppression_db` and `delta_suppression_db = 0.0000`. The largest scene benchmarked here is 83 Mpixels (0.62 GiB), which is why this has not been hit.

#### A third defect in the same place: NVRTC compile errors are discarded

`_init_cuda_driver` returns `False` on a compile failure without ever calling `nvrtcGetProgramLog`. The compiler's diagnostics are thrown away and the backend simply reports itself "not available" — so a one-line syntax error in `clean_hogbom.cu` is indistinguishable from a machine without CUDA.

#### The fix

**`audit/claude_code_opus_5/cuda_check.py`** — drop-in replacement for the raw ctypes handles, stdlib-only, lift into `clean_sar/backends/`. Provides `CudaDriver` (typed signatures for all 22 driver entry points, checked calls, ergonomic `mem_alloc`/`memcpy_*`/`launch` helpers, and `primary_context()` implementing the F7 fix), `Nvrtc` (compile with log capture), and `CudaError`/`NvrtcError`.

Verified by `a19`:

| | current tree | `cuda_check` |
|---|---|---|
| 1.5 GiB allocation | SUCCESS | SUCCESS |
| **2.0 GiB allocation** | **OUT_OF_MEMORY** | **SUCCESS** |
| **3.0 GiB allocation** | **OUT_OF_MEMORY** | **SUCCESS** |
| 64 GiB allocation | rc=2 ignored, NULL pointer used | raises `cuMemAlloc_v2 failed: CUDA_ERROR_OUT_OF_MEMORY (2) -- out of memory` |
| copy from NULL | rc=1 ignored, `np.empty()` returned | raises `cuMemcpyDtoH_v2 failed: CUDA_ERROR_INVALID_VALUE (1) -- invalid argument` |
| NVRTC compile error | returns `False`, log discarded | raises `NvrtcError` with `bad.cu(1): error: identifier "undefined_symbol_here" is undefined` |

Non-regression: 1 Mpix and 16 Mpix complex64 round-trips through the checked path are **bit-identical**, and the real `clean_hogbom.cu` still compiles (137 028 bytes of PTX) with all four kernels resolved.

Alongside this, allocate outputs with `np.zeros` rather than `np.empty`, so any path that is still missed yields obvious zeros instead of plausible noise.

### 🟠 F15 — Grid metadata is consumed without validation, and the DiffPFA products are non-conformant

**What the standard requires.** SICD DIDD 1.5 (`references/NGA.STND.0024-1`), p.40/42, defines `Grid/Row|Col/ImpRespWid` as a **Required** field holding the *"Half power impulse response width."* p.174/175 gives the normative relation:

> `Rg_IRW = k_RG / Krg_IRBW` — *"Impulse response broadening factor k_RG is associated with weighting function w_RG(s). For uniform weighting, k_RG = 0.886."*

So `ImpRespWid = k(window) / ImpRespBW`, with k = 0.886 for uniform, 1.125 Taylor(4,−30 dB), 1.303 Hamming, 1.441 Hann. `WgtType` and `WgtFunct` are both `minOccurs="0"` in the XSD — omitting them is legal, so `ImpRespWid` is the only required carrier of this information.

**What the products declare.** Measured across all 12 files (`a11`), k = `ImpRespWid × ImpRespBW`:

| Product group | declared k | Consistent with |
|---|---|---|
| 6 × raw Umbra `_SICD.nitf` | **0.8857 – 0.8859** | uniform ✅ |
| 6 × DiffPFA `_SICDU_*.nitf` | **1.0000 exactly** | no standard window ❌ |

**Which is wrong — the metadata or the aperture?** Decided by measuring the true half-power IRW from the image data, using the raw Umbra products as a control (`a13`):

| Product | declared k | **measured k** | Verdict |
|---|---|---|---|
| RAW Umbra 2023-11-14 | 0.8857 | **0.8734** | declared OK (1.4 % method bias) |
| DiffPFA 2023-11-14 | 1.0000 | **0.8857** | **declared 1.129× too wide** |
| RAW Umbra 2023-09-13 | 0.8857 | **0.8732** | declared OK |
| DiffPFA 2023-09-13 | 1.0000 | **0.8836** | **declared 1.132× too wide** |

The DiffPFA apertures measure k ≈ 0.886 — genuinely **uniform**, matching the raw products. The metadata is what is wrong: it writes `ImpRespWid = 1/ImpRespBW`, which is the **Rayleigh/nominal resolution**, not the half-power IRW the field is defined to hold. The two differ by exactly the 0.886 broadening factor.

**Consequence in CLEAN_SAR.** `row_wid`/`col_wid` set the Gaussian restoring-beam sigma, so on all six DiffPFA products the restoring beam is **1.129× wider than the data supports** — the restored image is ~13 % blurrier than it should be, and the product under-claims its own resolution to every downstream consumer.

**Fix in the DiffPFA writer (primary).** For each of Row and Col:
- `ImpRespWid = 0.886 / ImpRespBW` for an untapered aperture; if a taper is applied, use that window's k.
- Populate `WgtType/WindowName` (e.g. `UNIFORM`) so the product is self-describing rather than relying on a consumer's default.
- Populate `WgtFunct` (sampled amplitude weighting) if any non-uniform taper is applied.

**Fix in CLEAN_SAR (secondary — validate, don't accommodate).** On load, compute `k = ImpRespWid × ImpRespBW` and compare against the declared window, or against the table when `WgtType` is absent. On mismatch, **warn and proceed with the declared values** — CLEAN_SAR should not silently rewrite a product's stated physics, but it should tell the user the SICD is internally inconsistent. That single check would have surfaced this immediately, and would equally catch a Taylor-weighted product that omits `WgtType`.

### 🟡 F9 — The headline feature is real physics of negligible magnitude

`theta(x,y) = atan2(y_col, R0 + x_row)` is correct. At spaceborne slant ranges (646–764 km here) it is very small (`a02`, `a03`):

| Quantity | Across an entire full scene |
|---|---|
| θ span | 0.30° – 0.68° (all 12 files) |
| PSF displacement at 32-px kernel edge | 0.17 – 0.38 **pixels** |
| max \|PSF(corner) − PSF(SCP)\| | **2.6×10⁻³** = **0.26 % of peak** |
| max \|PSF(spatially-varying) − PSF(θ≡0)\| | **2.6×10⁻³** = **0.26 % of peak** |

The mainlobe does not change at all. Setting θ ≡ 0 everywhere — discarding the feature — changes the PSF by 0.26 % of peak.

Not an error, and the README's physics discussion is accurate. But "exact spatially-varying IPR for every detected peak" is the stated central differentiator, and it is what forces a PSF recomputation on every CLEAN iteration.

#### The regime where it does matter

Rather than simply discount the feature, `a20` maps it. Same X-band geometry, varying slant range and scene half-width — `max |PSF(corner) − PSF(SCP)|` as % of peak:

| Slant range | 0.5 km | 1.0 km | 2.5 km | 5.0 km | 10.0 km |
|---|---|---|---|---|---|
| 5 km (airborne, low) | 5.68 % | 10.49 % | 25.36 % | 28.71 % | 24.18 % |
| 20 km (airborne) | 1.43 % | 2.77 % | 7.09 % | 11.84 % | 25.36 % |
| 50 km (airborne standoff) | 0.57 % | 1.15 % | 2.77 % | 5.68 % | 10.49 % |
| 100 km | 0.29 % | 0.57 % | 1.43 % | 2.77 % | 5.68 % |
| 300 km | 0.09 % | 0.19 % | 0.48 % | 0.96 % | 1.88 % |
| **650 km (this data)** | 0.04 % | 0.09 % | **0.22 %** | 0.44 % | 0.88 % |

Thresholds:

| Scene half-width | >1 % variation | >10 % variation |
|---|---|---|
| 1.0 km | R₀ < 57 km | R₀ < 5 km |
| 2.5 km | R₀ < 143 km | R₀ < 13 km |
| 5.0 km | R₀ < 287 km | R₀ < 27 km |
| 10.0 km | R₀ < 573 km | R₀ < 53 km |

The audited data sits at R₀ = 647–764 km with half-widths of 1.4–2.8 km — outside every regime above.

**The feature is worth keeping**: it is correct, and it is what makes the tool right for airborne collects (R₀ of tens of km) and for very large scenes, where the effect reaches 5–25 %. The recommendation is only that the README say so, rather than presenting a quarter-percent spaceborne correction as the reason the tool is needed.

> **Project decision (recorded 2026-09-05):** the full spatially-varying path is to be **retained** — airborne data is anticipated, and the code should stay correct for close collections. This finding is documentation-and-scoping only. Do **not** remove the per-position θ computation.

An *optional* optimisation follows, without touching correctness: the variation across a scene is predictable from `scp_slant_range` and the scene extent before the loop starts (the tables above are exactly that prediction). When it falls below tolerance, θ can be computed once and the PSF reused; the per-iteration path stays for the airborne case. Since PSF synthesis is 13 % of the PyTorch backend's compute time (`a06`), this is free speed on precisely the data where the feature does nothing.

### 🟡 F10 — The benchmark summary omits the column where CUDA loses

`benchmark_pytorch_vs_cuda.py` computes and stores `total_gpu_speedup`, then prints only `compute_speedup`:

```
Average Pure Compute Speedup: 3.46x        ← printed
Average Total GPU Speedup:    2.37x        ← computed, saved to CSV, never printed
```

In **2 of 12 scenes CUDA is slower end-to-end** (0.53× and 0.43×), driven by D2H. Those rows are in the CSV and absent from the summary.

Two mitigating facts: the omitted mean is still favourable (2.37×), and the two outliers **do not reproduce** — the published `cuda_d2h_ms` of 35 953 ms for the largest scene measures 3 456 ms on my run, a 10× discrepancy (`a06`, `a07`). That points at single-sample, no-warmup benchmarking capturing a transient system state (WSL2 pageable transfers under memory pressure).

#### A second asymmetry: NVRTC startup is excluded from the CUDA timings

`run_hogbom_cuda_native` calls `_init_cuda_driver()` **before** it starts `t_pipeline_start`, so NVRTC compilation and module load fall outside every reported CUDA number. Measured on this machine:

| | |
|---|---|
| first `_init_cuda_driver()` (NVRTC compile + `cuModuleLoadData`) | **4 973 ms** |
| subsequent calls (cached) | 0.009 ms |

PyTorch's equivalent startup is *not* excluded — its CUDA context initialisation lands inside `torch_h2d_ms`, which is why the first published row shows `torch_h2d_ms = 754.8` against 21–113 ms for later files.

The 5 s is genuinely one-time and amortises across a batch, so this does not change the per-iteration compute comparison. But it is asymmetric bookkeeping, and it should either be reported as an explicit `cuda_startup_ms` column or be incurred by both backends outside the timed region.

#### Also mislabelled

The PyTorch backend's loop is commented `# 2. Pure GPU Compute Loop`, but it includes CPU-side numpy PSF synthesis and the per-iteration host→device transfer of two 65×65 kernels. Profiling puts that at **13.1 %** of the reported figure (`a06`). The number itself is honest — it is wall-clock for that section — only the label overstates what is on the GPU.

#### How much does a warm-up actually change the answer?

Measured (`a21`), same scene and settings, each condition in a fresh subprocess:

| Condition | torch ms | cuda ms | reported speedup |
|---|---|---|---|
| **A** — current methodology (cold, one sample each, PyTorch first) | 664.7 | 167.7 | **3.96×** |
| **B** — warm-up discarded, then 5 interleaved pairs, median | 557.5 | 153.8 | **3.63×** |

**Cold measurement inflates the headline by 8.5 %**, and it does so *systematically* rather than randomly: PyTorch gains far more from warming (−16.1 %) than CUDA does (−8.3 %), because torch carries more first-run overhead — allocator growth, kernel selection, and GPU clock ramp from the idle power state. Measuring cold therefore favours CUDA every time.

A warm-up alone is not sufficient, though. Per-sample spread within condition B:

| | samples (ms) | spread |
|---|---|---|
| torch | 533.8, 546.8, 557.5, 583.1, 557.5 | **8.8 % of median** |
| cuda | 150.2, 153.8, 146.3, 157.2, 156.2 | **7.1 % of median** |

A single sample carries roughly ±8 % noise, so the per-file variation in the published table (3.04× … 4.14×) is largely indistinguishable from measurement noise. Repeats and a median are needed as well as the warm-up.

**Recommended benchmark protocol**

1. One discarded warm-up run per backend at the **start of the batch** — absorbs NVRTC compile, torch CUDA init, allocator growth and clock ramp in one go.
2. ≥5 **interleaved** pairs per file (A,B,A,B,…), not blocked (A,A,…,B,B,…), so thermal drift hits both backends equally.
3. Report the **median** with min/max, not a single sample.
4. Report `cuda_startup_ms` separately rather than excluding it.
5. Print `total_gpu_speedup` alongside `compute_speedup`.

Expect the headline to fall from **3.96× to 3.63×** on this scene, and to become reproducible — a smaller number that survives being re-run. Note the warm-up belongs **per batch, not per file**: the costs it absorbs (NVRTC compile, torch init, allocator growth, clock ramp) are process-wide, so per-file warm-ups would multiply benchmark runtime for no additional benefit.

**Fix:** the protocol above; and retitle the PyTorch loop comment, which currently reads `# 2. Pure GPU Compute Loop`.

A hypothesis I formed and then disproved: I expected the speedup to be an artifact of PyTorch synthesising PSFs on CPU. Profiling shows that is only **13.1 %** of PyTorch's compute time (the LRU cache absorbs ~59 % of calls). The 3–4× is a genuine GPU-side result — a fused kernel against many small torch ops with three `.item()` syncs per iteration.

### 🟡 F11 — `config=None`: CUDA substitutes placeholder physics

```
backend=pytorch : raised ValueError: Either 'config' or 'psf_generator' must be provided.
backend=cuda    : NO ERROR. iterations=100, suppression=0.70 dB
```

`cuda_backend.py` lines 205–208 and 273–276 fall back to `row_ss=0.5, col_ss=0.5, row_bw=1.5, col_bw=1.5, sigma=0.5` via `... if config else <default>`. Same API, one backend refuses, the other returns a wrong answer.

### 🟡 F12 — README quickstart is broken

```
$ python demo_clean.py --target umbra
demo_clean.py: error: unrecognized arguments: --target umbra      # flag does not exist

$ python demo_clean.py                                            # documented invocation
[+] CLEAN pipeline complete in 0.70s
TypeError: plot_comparison() got an unexpected keyword argument 'chip_bounds'
```

`demo_clean.py:76` calls `plot_comparison(input_path, out_nitf, chip_bounds=…, save_path=…, dynamic_range_db=…)`. The real signature takes **arrays** (`dirty_image`, `clean_image`, `residual_image`, `restored_model`, `output_png`, `dyn_range_db`, `title_suffix`, `ref_val`) and none of those three keywords. It crashes *after* the deconvolution and NITF write, and before the second dataset — so the compute is spent, a valid NITF is produced, and the diffpfa run never happens.

The README also directs output to `output/umbra_20231114/` and `output/diffpfa_20231114/`; the code defaults to `output/demo`. **None of the three directories exists in the tree**, consistent with the demo never having completed.

**The bug is localised to the demo.** The README §4 CLI example was run end-to-end, `--plot` included, and works — it wrote a valid 799 KB six-panel PNG. So `plot_comparison` and `tools/compare_sicd.py` are correct; only `demo_clean.py`'s call site is wrong.

**Why this rates above cosmetic:** it is the front door. Anyone evaluating the repo runs the quickstart first and hits a traceback. It also means the six-panel comparison figure has never been produced by the demo — and per F2 that figure is the one artifact that would let a human *see* whether the deconvolution is behaving, as opposed to reading a `suppression_db` number that cannot tell correct physics from none.

**Fix:** give `demo_clean.py` the array-based call `cli.py` already uses —

```python
plot_comparison(
    dirty_image=dirty_img, clean_image=result.clean_image,
    residual_image=result.residual_image, restored_model=result.restored_model,
    output_png=out_png, dyn_range_db=dyn_range,
    title_suffix=os.path.basename(input_path),
)
```

— add the `--target {umbra,diffpfa,both}` flag the README documents (or drop it from the README), and reconcile the output-directory paths in one direction or the other. Wrapping the plot call so a rendering failure cannot discard a completed deconvolution would also be prudent; the current `except ImportError` in `cli.py` does not catch a `TypeError`.

### 🟡 F13 — 600 lines of dead C++/CUDA

`clean_sar_cuda.cu` (294 lines), `clean_sar_cuda.h`, `psf_math.cuh`, `reduction.cuh` and the `Makefile` producing `libcleansar.so` are never built and never loaded. Nothing references `libcleansar.so` except two strings describing it as the thing in use:

```python
# backends/__init__.py:9
"""Checks if the compiled native C++/CUDA shared library (libcleansar.so) is available."""
# backends/__init__.py:36
"Native C++/CUDA backend (libcleansar.so) is not compiled or not found. "
"Please use backend='pytorch' or compile the CUDA library in clean_sar/backends/c_src/."
```

The real path is NVRTC compiling `clean_hogbom.cu` at runtime.

| File | Lines | Status |
|---|---|---|
| `clean_hogbom.cu` | 263 | **live** — compiled at runtime by NVRTC |
| `clean_sar_cuda.cu` | 294 | dead |
| `clean_sar_cuda.h` | 102 | dead |
| `psf_math.cuh` | 85 | dead |
| `reduction.cuh` | 126 | dead |
| `Makefile` | 20 | dead — builds `libcleansar.so` |

**The instruction in the error message is actively counterproductive.** A user who hits it and runs `make` in `c_src/` would successfully build a shared library that is never opened, and the error would persist unexplained — while the real cause (NVRTC not found, or a compile error whose log is discarded per F8) goes unmentioned.

**The two implementations have drifted**, so this is a stale earlier design rather than a maintained fallback:
- `psf_math.cuh::compute_local_shear` computes θ **on the device** from the config struct.
- `clean_hogbom.cu` takes `cos_t`/`sin_t` as **kernel arguments**, computed on the host.

Keeping them invites someone to fix a bug in the wrong file, or to assume the two agree.

**Fix — recommended: delete** `clean_sar_cuda.{cu,h}`, `psf_math.cuh`, `reduction.cuh` and the `Makefile`, and correct the two strings in `backends/__init__.py` to describe the NVRTC path. An ahead-of-time fallback that is never built and never tested is worse than no fallback; if it is wanted later it remains in git history.

**If instead they are wired up** as a genuine AOT fallback for machines without NVRTC — which has real value, since NVRTC costs ~5 s of startup (F10) and is an extra runtime dependency — then two conditions must be met: the drift with `clean_hogbom.cu` must be resolved, and the AOT path must be added to the F4 parity matrix. A third implementation that nothing compares against would recreate exactly the F1 situation.

### 🟢 F14 — Test suite is machine-specific and GPU-mandatory

All five test modules hardcode `/home/feildaw/data/*.nitf`, and `test_resolve_backend_logic` opens with `assert is_cuda_native_available() is True`. The suite cannot run on another machine, in CI, or without a GPU — awkward against the README's "Designed for headless execution in cloud and Kubernetes environments with zero mandatory graphics dependencies." Three of the eight tests in `test_psf.py` don't guard `get_test_sicd()` returning `None`.

On this machine, `pytest -v tests/` → **16 passed**.


---

## 4. Recommendations, in priority order

**Do first — stop the code reporting things that did not happen.** These are small, independent of each other, and each removes a way for a wrong result to look like a right one.

1. Raise `NotImplementedError` on the CUDA path for `beam_type="mainlobe"`, `clean_mask`, and `psf_generator`, and remove them from the verbose log unless honoured. Do this now; implement the features later (F3). See §5.3 for the capability-table form that makes this class of bug impossible rather than remembered.
2. Make `config=None` raise in CUDA, as PyTorch does; delete the placeholder physics (F11).
3. Replace the raw ctypes handles with `audit/claude_code_opus_5/cuda_check.py` (typed signatures, checked calls, NVRTC log capture; also supplies the F7 `primary_context()`), and allocate outputs with `np.zeros` not `np.empty` (F8).

**Correctness**

4. Normalise the PSF centre to 1.0 in `clean_hogbom.cu`, matching `psf.py`. Do not instead remove the normalisation from `psf.py` (F1).
5. Use `cuDevicePrimaryCtxRetain` instead of `cuCtxCreate_v2` (strategy F-A, measured); optionally also restore the prior context after each run. Do **not** create/destroy a context per run — it breaks PyTorch and costs 385 ms + 134 MiB. **Do this before #11** — it blocks the parity test (F7).
6. Validate `ImpRespWid × ImpRespBW` against the declared/known windows on load and warn on mismatch; separately, fix the DiffPFA writer to emit the half-power IRW (`0.886/BW` uniform) rather than `1/BW` (F15).
7. Default `chip_start` to `(handler.first_row, handler.first_col)`; the tool's own written chips carry `FirstRow != 0` and currently round-trip to a 480 m positional error (F16).
8. Query device compute capability for the NVRTC `--gpu-architecture` flag instead of hardcoding `compute_86`; the current value cannot load on A100/T4/V100 (F17).
9. Wrap the CUDA allocation-to-cleanup span in `try/finally`. Land this together with the error-checking fix, which makes the leak reachable (F18).
10. Set `ImageCreation/Application` and `DateTime` on write so deconvolved products stop claiming the original formation software and collection time (F19).

**Test coverage**
11. Replace `test_cuda_backend_execution_parity` with `audit/claude_code_opus_5/test_backend_parity_reference.py` (array-level, all four weightings). Verified to catch F1, F3 and F11 (F4). Depends on #5.
12. Add `audit/claude_code_opus_5/test_clean_recovery_reference.py` (synthetic ground-truth recovery, both backends). Verified by `a16` to catch a deconvolution kernel that disagrees with the true IPR (F6).
13. Replace the two self-referential tests with `audit/claude_code_opus_5/test_psf_physics_reference.py`. Verified by `a15` to catch all four historical regressions the originals miss (F5).
14. Parametrise the test data path; `pytest.skip` when NVRTC is absent (F14).

**Reporting & docs**
15. Report mainlobe preservation and integrated sidelobe ratio; rename `suppression_db` to `peak_reduction_db`. Implementation ready in `audit/claude_code_opus_5/quality_metrics.py` (F2).
16. Adopt the F10 benchmark protocol: one discarded warm-up per backend per batch, ≥5 interleaved pairs per file, report medians, print `total_gpu_speedup`, and add a `cuda_startup_ms` column. Cold single-sample measurement currently inflates the headline by 8.5 % (F10).
17. Fix `demo_clean.py`'s `plot_comparison` call (use the array form from `cli.py`), add the documented `--target` flag, and reconcile the output-directory paths with the README (F12).
18. Delete the dead `c_src` files (recommended) and correct the two `libcleansar.so` strings to describe NVRTC. If instead wiring them up as an AOT fallback, first resolve the drift with `clean_hogbom.cu` and add that path to the F4 parity matrix (F13).
19. Add a README sentence scoping the spatially-varying IPR: 0.26 % at spaceborne ranges, 5–25 % for airborne. The feature is retained by project decision; any short-circuit must be an opt-in fast path, not a removal (F9).

---

## 5. Architecture notes

Offered separately from the findings: these are design observations, not defects.
The layering is sound — `sicd_handler` (I/O) → `config` (physics) → `psf` (model)
→ `backends` (compute) → `processor` (orchestration) → `cli`. Dependencies flow
one way, formats stay out of the physics, and the physics stays out of the
kernels. What follows is where the seams are rubbing.

### 5.1 Keep

- **`CleanPhysicsConfig` as a flat scalar POD.** Thirteen floats and ints that map
  directly onto a CUDA struct. This is the single best decision in the codebase:
  it is what makes two backends comparable at all, and it is why UNIFORM/TAYLOR
  parity is exact to float32. Do not let objects, tensors or XML back into it.
- **The analytic IPR.** Correct to ~10⁻⁹ against an independent transform, and
  far cheaper than an FFT-per-position approach.
- **Backend selection at the call boundary** rather than inside the loop.

### 5.2 The one structural root cause

**`PSFGenerator` is presented as a shared abstraction but only one backend can use it.**

The PyTorch backend calls `psf_gen.get_psfs_torch(...)` — a cached, host-side,
object-oriented PSF provider. The CUDA backend cannot consume that at all; it
evaluates the PSF arithmetic *inside the kernel* from scalars. So `PSFGenerator`
is not a shared contract, it is the PyTorch backend's implementation detail that
happens to live in a shared module.

Three findings are downstream of exactly this:

| | |
|---|---|
| **F3** | `psf_generator=` is accepted by `run_hogbom_clean` and silently ignored by CUDA — the abstraction does not survive the second backend |
| **F1** | the window math had to be re-implemented in `clean_hogbom.cu`, and the two copies then diverged (centre normalisation) |
| **F13** | and re-implemented a *third* time in `psf_math.cuh`, which has drifted further still |

**The fix in principle:** the shared contract between backends should be the PSF
*specification* — the scalars, which both already agree on — not a PSF *provider*
object. **Each backend owns its own PSF evaluation** and takes only the spec.

> **Project decision (recorded 2026-09-05):** adopt this. `PSFGenerator` ceases to
> be a cross-backend abstraction. `CleanPhysicsConfig` (the spec) is the sole
> contract; each implementation backend performs its own PSF calculation from it —
> PyTorch host-side with caching, CUDA inside the fused kernel, and any future
> backend however suits it. `psf_generator` leaves the public API.

Concretely:

1. Remove `psf_generator` from `run_hogbom_clean`'s public signature; `PSFGenerator`
   becomes an internal detail of the PyTorch backend (move it to
   `backends/pytorch_backend.py`, or keep it in `psf.py` as a helper that only
   that backend imports). `config` becomes the single required input — which also
   eliminates F11's `config=None` ambiguity outright, since there is no longer a
   second way to supply the physics.
2. If user-supplied PSFs are genuinely wanted later, that is a different entry
   point — a backend that accepts a precomputed PSF stack — not an optional
   argument that one backend honours and the other discards.
3. Accept that the window math exists twice (Python and CUDA C) — deriving one
   from the other by codegen is not worth it at this size — but **pin them
   together with a contract test** that evaluates both over a grid of positions
   and weightings and asserts pointwise agreement. That is a dozen lines and it
   makes F1 structurally impossible rather than a thing to remember.

### 5.3 Make "silently ignored" impossible by construction

F3 and F11 were both fixed above by *remembering* to raise. A declaration table
removes the need to remember:

```python
CAPABILITIES = {
    "pytorch": {"beam_type": {"gaussian", "mainlobe"}, "clean_mask": True,
                "guard_margin": True, "psf_generator": True},
    "cuda":    {"beam_type": {"gaussian"},             "clean_mask": False,
                "guard_margin": True, "psf_generator": False},
}
```

`run_hogbom_clean` validates the request against the chosen backend's entry and
raises `NotImplementedError` before dispatching. New options are then unsupported
by default rather than silently dropped by default — the failure mode inverts
from "quietly wrong" to "loudly incomplete". This is the highest-leverage
architectural change in this section.

### 5.4 Single sources of truth to consolidate

| Duplicated | Where | Note |
|---|---|---|
| Window/pattern math | `psf.py`, `clean_hogbom.cu`, `psf_math.cuh` | 3 copies; 2 have diverged (F1, F13) |
| Taylor coefficients | `utils.taylor_window_1d` (computed) vs `psf.py` (hardcoded literals) | verified identical to 1×10⁻⁹ today; nothing keeps them so |
| `global_to_metric` | `CleanPhysicsConfig` (naive) and `SICDHandler` (via sarkit) | verified identical to 0 m; one should delegate to the other |
| chip→global | `CleanPhysicsConfig.chip_to_global`, `SICDHandler.chip_to_global_rowcol` | same operation, two spellings |
| `CleanResult` | defined in `algorithm.py`, imported *inside functions* by both backends to dodge a circular import | move to its own `result.py`; both layers import it cleanly |

### 5.5 Interface simplifications

- **`threshold` is overloaded**: a fraction of the initial peak if `< 1.0`,
  otherwise absolute. A genuine absolute threshold below 1.0 cannot be expressed,
  and the behaviour changes discontinuously at an arbitrary value. Prefer two
  explicit parameters (`threshold_frac` / `threshold_abs`) or a small tagged type.
- **`suppression_db` names a quantity it does not measure** (F2). Rename to
  `peak_reduction_db`; add real quality metrics beside it.
- **`resolve_backend()` is a query with a 5-second side effect.** Asking *which*
  backend to use compiles PTX and creates a CUDA context — the direct cause of F7.
  Separate `probe_cuda_available()` (cheap, no context) from
  `initialise_cuda()` (expensive, explicit).
- **`run_hogbom_clean` forwards 12 identical kwargs down two hand-written
  branches.** They are already at risk of drifting; a dict dispatch with a single
  `**kwargs` pass removes a whole class of future bug.
- **`psf_size` even→odd coercion happens in three places** and not in the CUDA
  entry point. Normalise parameters once, at the public boundary.
- **Dead public API**: `taylor_window_1d`, `get_1d_window`, `get_2d_window` are
  exported from `clean_sar/__init__.py` and used by nothing. Either wire
  `taylor_window_1d` into `psf.py` as the coefficient source (removing a
  duplicate) or drop them from `__all__`.

### 5.6 Sequencing

None of this is urgent, and none of it should precede the correctness fixes in
§4. The natural order once those land: 5.3 (capabilities table) → 5.2 (drop
`psf_generator` from the public signature, add the window contract test) → 5.4
(consolidate) → 5.5 (interface polish). Each is independently shippable.


---

## 6. Cross-check against a second audit

Added 2026-09-05 at the repository owner's request, after this audit was
complete. A second independent audit — **Agy (Gemini 3.8 Flash)**,
`audit/agy_gemini_3.8_flash/` — was read and its unique claims reproduced here.
Findings originating in that audit are marked as such; §3 above remains this
audit's independent output and was not revised in light of it.

### 6.1 Independent corroboration

Both audits reached the same conclusion by different routes on the central
physics defect, which materially raises confidence in it:

| | this audit | Gemini 3.8 Flash |
|---|---|---|
| Hamming CUDA PSF error | 3.426× amplitude inflation | +10.64 dB (= 20·log₁₀ 3.426) |
| Convergence penalty | 657 vs 185 iterations (3.5×) | 156 vs 44 iterations (3.5×) |
| PSF variation, full scene | 2.59×10⁻³ | 2.59×10⁻³ |
| Gaussian σ convention | correct (ratio 1.000000) | correct |
| Backend parity, uniform | ≤4×10⁻⁷ | 5.33×10⁻⁷ |

Also independently agreed: F3 (mainlobe/clean_mask silently dropped), F12 (demo
crash), F13 (dead `libcleansar.so`), F2 (metric misnomer), and F7's mechanism
(`cuCtxCreate_v2` instead of `cuDevicePrimaryCtxRetain`) — though that audit
rates the context issue P2/clarity, whereas this one demonstrates it as a
reproducible runtime failure that also blocks the F4 fix.

### 6.2 Defects that audit found which this one missed

All four reproduced independently before being recorded (`a22`).

#### 🟠 F16 — `ImageData.FirstRow` / `FirstCol` are ignored when building the config

*(originally Gemini BUG-05; verified here)*

`CleanPhysicsConfig.from_sicd_handler` never reads `handler.first_row` /
`first_col`; `chip_start` defaults to `(0,0)`. Per SICD DIDD, `SCPPixel` stays
referenced to the **global** image while a chip's pixels are offset by
`FirstRow`/`FirstCol`, so the two must be combined.

None of the 12 shipped datasets has a non-zero `FirstRow`, which is presumably
why neither the code nor this audit's earlier passes caught it. **But the tool
generates its own trigger.** Writing a chip with `SICDHandler.write_nitf` and
reading it back:

```
Re-read CLEAN_SAR-written chip: FirstRow=1000 FirstCol=1000 SCPPixel=[2780. 6432.]
config.chip_start_row = 0   (should be 1000)
chip pixel (0,0) -> computed (-1335.8, -2650.8) m ; correct (-855.3, -2238.7) m
positional error: (480.5, 412.1) m
theta computed -0.23534 deg vs correct -0.19860 deg
```

So any round-trip — deconvolve a chip, then re-process the output — places the
scene 480 m × 412 m from where it is and evaluates the PSF at the wrong
geometry. Given F9, the θ error itself is negligible at spaceborne range, but
the coordinate error is not, and it would be material for the airborne case the
project intends to support.

**Fix:** default `chip_start` to `(handler.first_row, handler.first_col)` rather
than `(0, 0)`, and add `first_row`/`first_col` to whatever the caller supplies.

#### 🟠 F17 — NVRTC compute architecture is hardcoded to `compute_86`

*(originally Gemini BUG-06; verified here, with a scope correction)*

`cuda_backend.py:95`: `opts = [b"--std=c++14", b"--gpu-architecture=compute_86"]`

The device's capability is queryable in three lines (`cuDeviceGetAttribute`
75/76 → 8.6 on this machine) and is never queried. Measured PTX portability on
this sm_86 device:

| PTX target | compiles | loads on sm_86 |
|---|---|---|
| compute_70 | yes | yes |
| compute_80 | yes | yes |
| compute_86 | yes | yes |
| compute_90 | yes | **NO — driver rc=218 (`CUDA_ERROR_INVALID_PTX`)** |

PTX is *forward*-compatible: lower-targeted PTX JITs onto newer devices. So the
scope is narrower than that audit states — an **H100 (sm_90) would work**, since
compute_86 PTX JITs upward. It is **older** devices that fail: A100 (sm_80),
T4 (sm_75), V100 (sm_70) cannot load compute_86 PTX, exactly as compute_90 PTX
fails here.

Compounding: per F8 the compile log is discarded, so the failure surfaces as the
misleading "libcleansar.so is not compiled" message of F13.

**Fix:** query the device capability and build the flag from it. `cuda_check.py`
already exposes the driver call needed.

#### 🟡 F18 — Device allocations are not exception-safe

*(originally Gemini BUG-07; verified here)*

In `run_hogbom_cuda_native`: `try:` occurrences **0**, `finally:` occurrences
**0**, `cuMemFree_v2` calls **8** — all on the straight-line path after the loop.
Any exception inside the loop leaks every device buffer for the life of the
process (4 × H·W·8 bytes; 2.3 GiB on the largest scene benchmarked).

This compounds with F8 in a way worth noting: **today the leak is nearly
unreachable precisely because nothing raises.** Once return codes are checked as
recommended, every CUDA error becomes a leak path. The two fixes should land
together.

**Fix:** wrap allocation through cleanup in `try/finally`, or use a small
RAII-style context manager.

#### 🟡 F19 — Deconvolved products inherit the original `ImageCreation` provenance

*(originally Gemini §2.3; verified here)*

`write_nitf` sets the NITF header fields (`ostaid="CLEAN_SAR"`,
`isorce="CLEAN_SAR_DECONVOLVED"`) but leaves the SICD XML `ImageCreation` block
untouched:

| field | input product | written product |
|---|---|---|
| `Application` | `Valkyrie Systems Sage \| Umbra Ima…` | `Valkyrie Systems Sage \| Umbra Ima…` |
| `DateTime` | `2023-07-31 06:54:58.938920+00:00` | `2023-07-31 06:54:58.938920+00:00` |

A deconvolved product therefore claims to have been produced by the original
image-formation software at the original collection time. `NumRows`/`NumCols`
and `FirstRow`/`FirstCol` *are* written correctly.

**Fix:** set `ImageCreation/Application` to `CLEAN_SAR <version>` and
`DateTime` to the processing time on write.

### 6.3 Where the two accounts differ

| Point | Gemini 3.8 Flash | This audit (measured) |
|---|---|---|
| `demo_clean.py` failure | "fails immediately upon execution", `TypeError: bad operand type for abs(): 'str'` | deconvolution **and** NITF write complete first; the error is `unexpected keyword argument 'chip_bounds'`, raised at call time before the function body runs, so `abs()` is never reached (`a22` X5). Same root cause, different triage — the compute is not lost, only the figure |
| Compute speedup | 5.09× (512×512 chip, 500 iters, single cold sample) | 4.1× on full scenes; and per F10 a cold single sample inflates the figure ~8.5 %, since PyTorch benefits more from warm-up than CUDA. A small chip also favours CUDA further, because PyTorch's fixed per-iteration overhead is a larger fraction of a smaller image. Not a contradiction — different conditions — but the 5.09× carries the same warm-up caveat |
| `compute_86` impact | fails on A100, V100, T4, **and H100** | H100 (sm_90) would work; PTX JITs forward. Older targets fail |
| Context handling | P2, "clarity" | reproducible runtime failure (`invalid resource handle`) that also blocks the F4 parity fix (F7) |

### 6.4 Findings in §3 not raised by the other audit

Listed for whoever merges the two, not as a scorecard: F1's photometric argument
for *which* normalisation is correct (a11), F4 (the parity test cannot detect a
parity failure), F5 (self-referential tests), F6 (no ground-truth test), F8 (no
error checking, 32-bit size truncation, discarded NVRTC logs), F10 (benchmark
warm-up and the omitted `total_gpu_speedup` column), F11 (`config=None`
placeholder physics), F14 (test-suite portability), F15 (DiffPFA `ImpRespWid`
non-conformance), and the F2 demonstration that a delta PSF scores within 0.54 dB
of correct physics.

### 6.5 Also worth carrying forward

Two observations from that audit that are not defects but are good calls:

- **Wavefront-curvature defocus is not modelled.** Under PFA the dominant
  space-variant degradation away from the SCP is quadratic/cubic phase error,
  not rigid rotation. This audit measured the rotation as negligible at
  spaceborne range (F9); that audit notes the term which is *not* negligible is
  absent from the model altogether. For already-refocused inputs (DiffPFA) the
  rigid-sinc assumption is appropriate, so this is a scope statement for the
  README rather than a defect — but it is the right caveat to state.
- **The CUDA loop still does three synchronous `cuMemcpyDtoH_v2` per iteration.**
  Keeping loop state on-device (CUDA Graphs or a persistent kernel) is the
  obvious next optimisation, and is where the remaining latency sits.
- **Mainlobe beam truncation at |dirty| > 0.1 (−20 dB)** cuts the beam off
  sharply, which will ring. Worth a taper if the mainlobe beam is kept.


---

## 7. Audit artifacts

| Script | What it establishes |
|---|---|
| `a01_psf_math.py` | Taylor coefficients, analytic IPR vs. numerical FT, −3 dB width conventions, PSF is purely real |
| `a02_metadata.py` | Grid metadata for all 12 files; naive vs. sarkit coordinates; θ span |
| `a03_ipr_ground_truth.py` | Empirical k-space support; baseband confirmation; measured point-target IPR; PSF spatial variation |
| `a04_metric_validity.py` | Synthetic ground-truth recovery; suppression_db under deliberately wrong physics |
| `a05_backend_parity.py` | Array-level CUDA↔PyTorch comparison across 4 weightings; PSF centre values; mask/guard honouring |
| `a06_benchmark_methodology.py` | PyTorch compute-time profile; reproduction of published numbers; variance; omitted speedup column |
| `a07_cuda_error_handling.py` | Unchecked API failures returning uninitialised memory; 72 Mpix validation; `config=None` |
| `a08_phantom_options.py` | `beam_type="mainlobe"` and `psf_generator` ignored by CUDA |
| `a09_context_conflict.py` | Seven-scenario isolation of the driver-context failure |
| `a10_real_sidelobe_suppression.py` | True sidelobe/mainlobe metrics vs. reported metric on the showcase chip |
| `a11_normalization_and_wgt.py` | Which PSF normalisation is correct (photometry); weighting identification from Wid×BW |
| `a12_sicd_standard_check.py` | Normative definitions of ImpRespWid/BW/WgtType/WgtFunct from the NGA DIDD and XSD |
| `a13_diffpfa_irw_check.py` | Measured half-power IRW vs. declared, with raw Umbra products as control |
| **`quality_metrics.py`** | **Drop-in F2 metrics: `ipr_quality`, `ipr_quality_multi`, `verdict`. numpy-only, lift into `clean_sar/`** |
| `a14_verify_quality_metrics.py` | Verifies `quality_metrics.py` reproduces the F2 table exactly |
| **`test_backend_parity_reference.py`** | **Drop-in replacement parity test: array-level, all weightings, option-honouring. Lift into `tests/`** |
| **`test_psf_physics_reference.py`** | **Drop-in replacement PSF physics tests: every assertion reads the library. Lift into `tests/`** |
| `a15_regression_bite.py` | Reintroduces 4 historical defects at runtime; proves the reference suite catches all 4 and the originals miss 3 |
| **`test_clean_recovery_reference.py`** | **Drop-in synthetic ground-truth recovery tests, both backends. Lift into `tests/`** |
| `a16_recovery_bite.py` | Breaks the algorithm's PSF 3 ways; proves the recovery suite catches all 3 and the original misses all 3 |
| `a17_context_strategy.py` | Context handles, memory and timing costs; shows a primary-context reset silently corrupts live PyTorch results |
| `a17b_context_fixes.py` | Three F7 fix strategies run end-to-end; F-A and F-B pass, per-run create/destroy fails |
| `a18_size_arg_truncation.py` | Proves unprototyped ctypes size arguments truncate at 2³¹ (2 GiB) |
| **`cuda_check.py`** | **Drop-in checked CUDA/NVRTC driver wrapper: typed signatures, raising calls, compile logs. Lift into `clean_sar/backends/`** |
| `a19_verify_cuda_check.py` | Verifies `cuda_check.py` fixes all three F8 defects with bit-identical round-trips |
| `a20_spatial_variation_regime.py` | Maps PSF spatial variation against slant range and scene size; identifies where the feature matters |
| `a21_warmup_methodology.py` | Quantifies how cold, single-sample benchmarking inflates the reported speedup |
| `a22_crosscheck_other_audit.py` | Independent reproduction of the four defects unique to the Gemini 3.8 Flash audit (F16–F19) |

Each has a corresponding `.txt` with captured output. All are read-only with respect to `clean_sar/`; the only files written outside `audit/` were a short demo run into `audit/claude_code_opus_5/demo_out/`.

---

*Prepared by **Claude Opus 5 (Anthropic), running in Claude Code**. All artifacts live in `audit/claude_code_opus_5/`. Findings are reproducible via the scripts above on the audited machine. Where a hypothesis was contradicted by measurement — a predicted √2 Gaussian error (C6), a predicted CPU-PSF benchmark artifact (F10), and the initial trigger proposed for the context bug (F7) — the measurement is reported, not the hypothesis.*
