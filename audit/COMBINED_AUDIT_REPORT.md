# CLEAN_SAR: Combined Audit Report

**Compiled by:** Claude Opus 4.6  
**Date:** August 29, 2026  
**Independent Auditors:**
- Claude Opus 4.6 — [`audit/claude_opus_4_6/`](claude_opus_4_6/)
- Gemini 3.1 Pro — [`audit/gemini_3_1_pro/`](gemini_3_1_pro/)
- Gemini 3.7 Flash — [`audit/gemini_3_7_flash/`](gemini_3_7_flash/)

---

## 1. Project Overview & Problem Statement

**CLEAN_SAR** is a GPU-accelerated Python framework implementing **Complex Hogbom CLEAN deconvolution** for Synthetic Aperture Radar (SAR) imagery stored in the NGA **NITF SICD** standard format.

Traditional radio astronomy CLEAN implementations operate on real-valued, positive-intensity maps under the assumption of a shift-invariant PSF. SAR imagery breaks these assumptions:

1. **Complex Signals (I + jQ):** SAR data preserves interferometric phase and coherent scattering amplitude. Deconvolution must operate in the complex domain.
2. **Spatially-Varying IPR/PSF:** In Polar Format Algorithm (PFA) image formation, polar spatial frequencies (K_r, θ) are mapped into Cartesian coordinates (K_row, K_col), causing the effective aperture boundaries, resolution, and squint angle to vary continuously as a function of scene position relative to the Scene Center Point (SCP): θ(x_row, y_col) ≈ arctan(y_col / (R₀ + x_row))
3. **Format Standardization:** Transitioning from FITS astronomy files to the official NGA NITF SICD standard via SARkit.

### Architecture

```
                          +-------------------------------+
                          |   Input NITF SICD (.nitf)     |
                          +-------------------------------+
                                          |
                                          v
 +-----------------------------------------------------------------------------------+
 |  sicd_handler.py: SICDHandler (SARkit I/O & Metadata)                             |
 |  - Full image & sub-image (chip) reading with native complex64 buffer management  |
 |  - Bidirectional coordinate mapping: Chip (r,c) -> Global (r_g,c_g) -> Metric (x,y)|
 |  - SICD XML subtree modification and compliant NITF export                        |
 +-----------------------------------------------------------------------------------+
                                          |
                                          v
 +-----------------------------------------------------------------------------------+
 |  psf.py: PSFGenerator (Physics & Beam Synthesis)                                   |
 |  - Option A: K-space Aperture Support (BW_r, BW_c, Windowing) + 2D IFFT           |
 |  - Option B: Analytic Spatial Sinc with Geometric Shear                           |
 |  - Clean Beam: Elliptical Gaussian matched to ImpRespWid or Mainlobe taper        |
 |  - PyTorch on-device tensor cache keyed by (r_g, c_g, size, method/beam)          |
 +-----------------------------------------------------------------------------------+
                                          |
                                          v
 +-----------------------------------------------------------------------------------+
 |  engine.py: run_hogbom_clean (GPU Deconvolution Engine)                           |
 |  - Peak detection on |Residual| tensor (with optional boolean mask)                |
 |  - Coherent component accumulation: comp = gamma * Residual[r0, c0]               |
 |  - Windowed dirty PSF subtraction from Residual on device                         |
 |  - Windowed clean beam accumulation into Restored Model on device                 |
 |  - Clean Image synthesis: Clean = Restored Model + Residual                       |
 +-----------------------------------------------------------------------------------+
                                          |
                                          v
 +-----------------------------------------------------------------------------------+
 |  utils.py & cli.py: Export & Multi-Panel Visualization                            |
 |  - dB scaling: 20 * log10(|image| / peak) clipped to dynamic range                |
 |  - Multi-panel comparison figure (Dirty, Clean, Components, Residual, PSFs)       |
 |  - Output NITF SICD product writing                                               |
 +-----------------------------------------------------------------------------------+
```

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| SICD Handler | `clean_sar/sicd_handler.py` | 226 | NITF I/O, metadata, coordinates |
| PSF Generator | `clean_sar/psf.py` | 348 | Spatially-varying dirty PSF + clean beam |
| CLEAN Engine | `clean_sar/engine.py` | 188 | Hogbom CLEAN loop on PyTorch |
| Utilities | `clean_sar/utils.py` | 217 | Windows, dB scaling, plotting |
| CLI | `clean_sar/cli.py` | 136 | Argument parsing, orchestration |

---

## 2. Test Suite Results

All **9 existing tests pass** against UMBRA SICD data. 2,348 deprecation warnings from sarkit's use of legacy `importlib.resources` APIs (cosmetic, no functional impact).

| Test | Status |
|------|--------|
| `test_exact_spatially_varying_clean_synthetic` | ✅ PASS |
| `test_exact_spatially_varying_clean_analytic` | ✅ PASS |
| `test_psf_kspace_generation` | ✅ PASS |
| `test_psf_analytic_generation` | ✅ PASS |
| `test_clean_beam_generation` | ✅ PASS |
| `test_sicd_handler_init` | ✅ PASS |
| `test_sicd_handler_chip_read` | ✅ PASS |
| `test_sicd_handler_coordinates` | ✅ PASS |
| `test_sicd_handler_write_nitf` | ✅ PASS |

---

## 3. Consolidated Findings

### Finding 1 — 🔴 CRITICAL: Hard-coded Slant Range Constant in PFA Shear Calculation

| | |
|---|---|
| **Location** | `clean_sar/psf.py` lines 129, 211, 278 |
| **Reported by** | Claude Opus 4.6, Gemini 3.7 Flash |
| **Confirmed by** | All three auditors |

```python
theta_local = np.arctan2(ycol, 10000.0 + xrow)  # ← hard-coded magic number
```

The spatially-varying shear angle uses a **hard-coded 10,000 m reference distance** as an approximation of the slant range to SCP (R₀). The actual SICD metadata shows:

| UMBRA Dataset | Actual Slant Range (R₀) | Error Factor |
|---------------|---------------------------|--------------|
| Umbra-05 (2023-09-11) | 764,470 m | **64×** |
| Umbra-05 (2023-07-30) | ~646,692 m | **61×** |

At the edges of a full image (max cross-range offset ~2,758 m from SCP):

| | Hard-coded (10 km) | Correct (~764 km) |
|---|---|---|
| Max shear angle | **13.13°** | **0.21°** |
| PSF spatial variation rate | 64× too fast | Correct |

**Impact:** PSFs near the SCP (center) are correct since θ ≈ 0 there. For **small chips near SCP** (the typical example use case), the error is tolerable. For **full-image CLEAN or off-center chips**, the PSFs at image edges are physically wrong, producing incorrect sidelobe subtraction.

**Recommended fix:**
```python
# In sicd_handler.py _parse_metadata():
self.scp_slant_range = float(self._safe_load("./{*}SCPCOA/{*}SlantRange", 10000.0))

# In psf.py, all three occurrences:
r0 = self.sicd.scp_slant_range
theta_local = np.arctan2(ycol, r0 + xrow)
```

---

### Finding 2 — 🔴 CRITICAL: Gaussian Restoring Beam Width Uses Wrong FWHM Convention

| | |
|---|---|
| **Location** | `clean_sar/psf.py` lines 268–285 |
| **Reported by** | Gemini 3.7 Flash |
| **Confirmed by** | Claude Opus 4.6 (post-verification; originally marked PASS in error) |

The code computes the Gaussian sigma from `ImpRespWid` using:

```python
fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548
sigma_r = wid_r / fwhm_factor
```

This factor corresponds to the **half-amplitude (−6 dB power)** width. However, the SICD standard defines `ImpRespWid` as the **half-power (−3 dB)** impulse response width. The correct derivation:

For a voltage Gaussian E(x) = exp(-x²/(2σ²)), the power is P(x) = |E(x)|² = exp(-x²/σ²).

At half-power (x = W/2):
  exp(-(W/2)²/σ²) = 1/2  →  σ = W / (2√(ln 2)) ≈ W / 1.6651

| | Current Code | Correct |
|---|---|---|
| Factor | 2√(2 ln 2) ≈ 2.3548 | 2√(ln 2) ≈ 1.6651 |
| Power at W/2 | −6.02 dB | **−3.01 dB** |
| Beam width error | **√2 ≈ 1.41× too narrow** | Correct |

This is independently confirmed by the SICD default fallback formula `ImpRespWid = 0.886 / ImpRespBW`, where 0.886 is the half-power sinc² width factor.

**Recommended fix:**
```python
# Replace in psf.py:
factor_3db = 2.0 * np.sqrt(np.log(2.0))  # ≈ 1.6651
sigma_r = wid_r / factor_3db
sigma_c = wid_c / factor_3db
```

**Verification:** Gemini 3.7 Flash's demo at `audit/gemini_3_7_flash/output/restoring_beam_fix_comparison.png` confirms the fixed beam drops to exactly −3.01 dB at W/2.

---

### Finding 3 — 🟡 WARNING: `write_nitf` Mutates `self.xmltree` In-Place

| | |
|---|---|
| **Location** | `clean_sar/sicd_handler.py` lines 204–214 |
| **Reported by** | Claude Opus 4.6 |

When `write_nitf` is called **without** `custom_xmltree`, it modifies the handler's internal `self.xmltree` directly:

```python
xml = custom_xmltree if custom_xmltree is not None else self.xmltree
# ...
num_rows_elem.text = str(complex_image.shape[0])  # mutates self.xmltree!
```

After writing a chip, the handler's `NumRows`/`NumCols` metadata is **silently corrupted** to reflect the chip dimensions instead of the original image. Subsequent operations using the same handler will produce incorrect results.

**Recommended fix:** Use `copy.deepcopy(self.xmltree)` when `custom_xmltree is None`.

---

### Finding 4 — 🟡 WARNING: Taylor Window Bug in `_eval_window_continuous`

| | |
|---|---|
| **Location** | `clean_sar/psf.py` lines 64–69 |
| **Reported by** | Claude Opus 4.6 |

```python
elif name_upper == "TAYLOR":
    u = norm_freq[in_band]
    n_pts = len(u)
    t_w = taylor_window_1d(n_pts, nbar=4, sll=-30.0)
    w[in_band] = t_w
```

This assigns a 1D Taylor window (indexed sequentially) to a **flattened set of 2D frequency samples**. The window values are mapped by flat array index, not by frequency ordering. For a 2D aperture, this produces incorrect, non-physical weighting — the Taylor taper is applied along an arbitrary direction rather than along each frequency axis independently.

**Current impact:** UMBRA data has `WgtType=None` (uniform weighting), so this code path **is never triggered**. If used with Taylor-weighted SICD data, it would produce wrong results.

**Recommended fix:** Apply 1D windows separately along each axis (row and col), then combine as an outer product — consistent with the HAMMING and HANN branches.

---

### Finding 5 — 🟡 WARNING: Unbounded PSF Cache Can Cause OOM

| | |
|---|---|
| **Location** | `clean_sar/psf.py` lines 22–23, 330–345 |
| **Reported by** | Claude Opus 4.6, Gemini 3.7 Flash |

The `PSFGenerator` stores every unique `(row, col)` PSF pair (dirty + clean) in unbounded dictionaries:

| PSF Size | Per-pair Memory | 2,500 iters | 10,000 iters | 50,000 iters |
|----------|----------------|-------------|--------------|--------------|
| 65×65 | 67.6 KB | ~165 MB | ~645 MB | ~3.2 GB |
| 129×129 | 266 KB | ~650 MB | ~2.6 GB | ~13 GB |

There is no cache eviction — only a manual `clear_cache()` method. Long runs on large images risk GPU OOM.

**Recommended fix:** Use `collections.OrderedDict` with a max size cap (e.g., 4,096 entries) implementing LRU eviction, as demonstrated in Gemini 3.7 Flash's `audit/gemini_3_7_flash/fixed_components.py`. Alternatively, compute PSFs on a coarse spatial grid and interpolate for nearby positions.

---

### Finding 6 — 🟢 MINOR: Taylor Window ~1% Deviation from scipy

| | |
|---|---|
| **Location** | `clean_sar/utils.py` lines 7–58 |
| **Reported by** | Claude Opus 4.6 |

The custom `taylor_window_1d` produces values differing from `scipy.signal.windows.taylor` by up to ~1.2%. Both formulations are mathematically valid with slight numerical differences in the coefficient computation. Since the code currently only uses uniform weighting for the available UMBRA data, this is academic.

---

### Finding 7 — 🟢 MINOR: Version Mismatch

| | |
|---|---|
| **Location** | `clean_sar/__init__.py` L16 vs `pyproject.toml` L7 |
| **Reported by** | Claude Opus 4.6 |

`__init__.py` declares `__version__ = "0.2.0"` while `pyproject.toml` declares `version = "0.1.0"`.

---

### Finding 8 — 🟡 WARNING: Unused Windowing in Analytic PSF Generator (`compute_psf_analytic`)

| | |
|---|---|
| **Location** | `clean_sar/psf.py` lines 174–175, 218–219 |
| **Reported by** | Gemini 3.7 Flash |

```python
def compute_psf_analytic(
    self,
    row: Union[int, float],
    col: Union[int, float],
    psf_size: int = 65,
    window_row: Optional[str] = None,
    window_col: Optional[str] = None,
    chip_origin: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    ...
    resp_u = np.sinc(bw_r * U_prime)
    resp_v = np.sinc(bw_c * V_prime)
    psf = (resp_u * resp_v).astype(np.complex128)
```

`compute_psf_analytic` declares `window_row` and `window_col` in its signature, but the function body ignores them entirely and evaluates pure unwindowed sinc responses (first sidelobes fixed at $-13.3\text{ dB}$). If a user specifies `--method analytic` for a Taylor- or Hamming-weighted SICD, the analytic PSF will not match the true aperture weighting, causing over- or under-subtraction of sidelobes.

**Recommended fix:** Either apply spatial-domain window tapering in `compute_psf_analytic` or document Option B as strictly uniform-aperture sinc.

---

### Finding 9 — 🟢 MINOR: Boundary Scatterer Sidelobe Truncation at Chip Edges

| | |
|---|---|
| **Location** | `clean_sar/engine.py` lines 155–164 |
| **Reported by** | Gemini 3.7 Flash |

```python
r_min = max(0, r0 - kh)
r_max = min(H, r0 + kh + 1)
c_min = max(0, c0 - kw)
c_max = min(W, c0 + kw + 1)
```

Slicing arithmetic correctly prevents out-of-bounds indexing when peaks are detected near image boundaries. However, subtracting a truncated PSF kernel near the boundaries can leave uncancelled sidelobe energy outside the active chip region.

**Recommended fix:** Provide an optional `guard_margin` parameter or default `clean_mask` border margin of `psf_size // 2` to avoid selecting point components whose mainlobe/sidelobes fall outside the chip.

---

### Finding 10 — 🟢 MINOR: Radiometric Normalization Scale in Plotting

| | |
|---|---|
| **Location** | `clean_sar/utils.py` line 160, `clean_sar/cli.py` line 120 |
| **Reported by** | Gemini 3.7 Flash |

`plot_clean_comparison` sets `ref_val = np.max(np.abs(dirty_image))`. When processing a sub-image chip containing low-contrast clutter, the $0\text{ dB}$ reference level is anchored to the chip's local maximum rather than the full scene's absolute radiometric scale.

**Recommended fix:** Allow passing an optional user-defined `ref_val` (e.g. from the full scene peak) via CLI `--ref-val` to maintain consistent dB scaling across multiple chips.

---

## 4. Verified Correct Implementations

All three auditors independently confirmed the following are correctly implemented:

| Component | Verification |
|-----------|-------------|
| **CLEAN engine Hogbom loop** | Peak finding (r0 = flat_idx // W, c0 = flat_idx % W), PSF subtraction alignment, clean beam accumulation, and final image composition (clean = restored_model + residual) are all correct. Synthetic point source convergence test passes. |
| **K-space PSF (Option A)** | Frequency sampling dk = 1/(N·SS) is the correct DFT relationship. ifftshift → ifft2 → fftshift pipeline correctly centers the PSF. |
| **Analytic PSF (Option B)** | np.sinc(BW·x) correctly accounts for numpy's sin(πx)/(πx) convention. K-space and analytic PSFs agree in mainlobe width (max diff ~0.015). |
| **SICD Handler I/O** | Metadata extraction matches direct XML queries. Coordinate round-trips achieve ≤ 3.44×10⁻¹³ pixel error across 10,000-point grid. Chip read/write is bitwise lossless. SCP pixel [row, col] ordering is correct. |
| **Cache key logic** | get_psfs_torch correctly computes global coordinates from chip-relative coords + origin, then passes to PSF methods without chip_origin — no double offset. |
| **torch.complex64 processing** | Native complex tensor processing ensures high GPU performance. |

---

## 5. Chip Size Scaling & Performance

From Gemini 3.7 Flash's empirical benchmark (`audit/gemini_3_7_flash/demo_chip_scaling.py`):

| Chip Size | Pixels | Scatterers Extracted | Peak Reduction | GPU Time | Throughput |
|-----------|--------|---------------------|----------------|----------|------------|
| 64×64 | 4,096 | 886 | −27.40 dB | 9.18 s | 544.6 iter/s |
| 128×128 | 16,384 | 977 | −24.40 dB | 8.84 s | 565.4 iter/s |
| 256×256 | 65,536 | 1,120 | −23.63 dB | 9.14 s | 546.9 iter/s |
| 512×512 | 262,144 | 1,557 | −20.61 dB | 10.98 s | 455.5 iter/s |

Scaling figure: `audit/gemini_3_7_flash/output/chip_scaling_analysis.png`

---

## 6. Output Quality

The CLEAN algorithm produces visually reasonable results on UMBRA satellite data, with appropriate sidelobe reduction and component extraction. The dirty PSF shows the expected cross-shaped pattern from rectangular aperture support.

- **UMBRA SICD chip**: 660 point scatterers, residual reduced to 2.3% of peak
  (`output/umbra_20230730/2023-07-30-17-19-39_UMBRA-05_SICD_chip_comparison.png`)
- **diffpfa SICDU chip**: 690 point scatterers, residual reduced to 2.0% of peak
  (`output/diffpfa_20230730/2023-07-30-17-19-39_UMBRA-05_SICDU_X_X_chip_comparison.png`)

---

## 7. Summary

| # | Severity | Finding | Auditor(s) | Impact |
|---|----------|---------|------------|--------|
| 1 | 🔴 Critical | Hard-coded 10,000m slant range constant | Opus 4.6, Flash | PSFs wrong at image edges (64× error in shear) |
| 2 | 🔴 Critical | Gaussian beam uses −6dB width (should be −3dB) | Flash, confirmed Opus 4.6 | Restoring beam √2× too narrow |
| 3 | 🟡 Warning | `write_nitf` mutates `self.xmltree` in-place | Opus 4.6 | Silent metadata corruption |
| 4 | 🟡 Warning | Taylor window applied incorrectly in 2D | Opus 4.6 | Wrong weighting for Taylor-weighted SICDs |
| 5 | 🟡 Warning | Unbounded PSF cache | Opus 4.6, Flash | OOM risk for long iterations |
| 6 | 🟢 Minor | Taylor window ~1% deviation from scipy | Opus 4.6 | Academic |
| 7 | 🟢 Minor | Version mismatch (0.2.0 vs 0.1.0) | Opus 4.6 | Packaging |
| 8 | 🟡 Warning | Analytic PSF ignores window parameters | Flash | Wrong sidelobe levels for windowed SICDs |
| 9 | 🟢 Minor | Boundary scatterer truncation at chip edges | Flash | Uncancelled boundary sidelobes |
| 10 | 🟢 Minor | Plotting dB scale normalized to chip local max | Flash | Radiometric scale inconsistency |

**Key takeaway:** The project's two core differentiators — spatially-varying PSF computation and matched Gaussian restoring beam — both contain physics-level bugs (Findings 1 and 2). These are **masked in practice** because the typical usage is small chips near the SCP (where shear angle ≈ 0) and the beam width error, while systematic, doesn't prevent convergence. Both fixes are straightforward and confined to `psf.py` and `sicd_handler.py`.

---

## 8. Audit Artifact Index

### Claude Opus 4.6 — `audit/claude_opus_4_6/`
| File | Purpose |
|------|---------|
| `AUDIT_REPORT.md` | Individual detailed audit report |
| `audit_psf_physics.py` | PSF math & physics verification |
| `audit_engine.py` | CLEAN engine correctness verification |
| `audit_sicd_handler.py` | SICD I/O & metadata verification (10,000-point grid) |

### Gemini 3.1 Pro — `audit/gemini_3_1_pro/`
| File | Purpose |
|------|---------|
| `audit_summary.md` | Problem statement & approach analysis |
| `audit_demo.py` | End-to-end demo (50×50 chip CLEAN) |
| `output/demo_clean.nitf` | Demo output NITF |

### Gemini 3.7 Flash — `audit/gemini_3_7_flash/`
| File | Purpose |
|------|---------|
| `audit_report.md` | Full technical & physics audit report |
| `fixed_components.py` | Reference implementations with all fixes |
| `demo_fixes.py` | Restoring beam & slant range fix demos |
| `demo_chip_scaling.py` | Chip scaling benchmark |
| `verify_engine_and_physics.py` | 4-part automated verification suite |
| `verify_math_details.py` | Detailed math verification |
| `inspect_datasets.py` | Metadata inspection across all 10 NITF files |
| `benchmark_clean_run.py` | Full benchmark run script |
| `output/restoring_beam_fix_comparison.png` | Beam fix verification plot |
| `output/chip_scaling_analysis.png` | Scaling benchmark plot |
| `output/benchmark_clean_chip.nitf` | Benchmark output NITF |
| `output/benchmark_clean_comparison.png` | Benchmark comparison plot |

### Re-running audit scripts

```bash
source /home/feildaw/mypyenv/bin/activate
cd /home/feildaw/CLEAN_SAR

# Claude Opus 4.6
python audit/claude_opus_4_6/audit_psf_physics.py
python audit/claude_opus_4_6/audit_engine.py
python audit/claude_opus_4_6/audit_sicd_handler.py

# Gemini 3.1 Pro
python audit/gemini_3_1_pro/audit_demo.py

# Gemini 3.7 Flash
python audit/gemini_3_7_flash/verify_engine_and_physics.py
python audit/gemini_3_7_flash/demo_fixes.py
python audit/gemini_3_7_flash/demo_chip_scaling.py
```
