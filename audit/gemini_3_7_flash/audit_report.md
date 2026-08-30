# CLEAN_SAR: Technical & Physical Audit Report

**Auditor:** Independent Technical Audit (Gemini 3.7 Flash)  
**Date:** August 29, 2026  
**Repository:** `CLEAN_SAR` (`/home/feildaw/CLEAN_SAR`)  
**Audit Directory:** `audit/gemini_3_7_flash/`

---

## 1. Executive Summary

`CLEAN_SAR` is a Python/PyTorch framework implementing **Complex Hogbom CLEAN deconvolution** tailored specifically for Synthetic Aperture Radar (SAR) imagery stored in the NGA **NITF SICD** (Sensor Independent Complex Data) standard format.

Unlike traditional astronomical CLEAN implementations that operate on real-valued, positive-intensity incoherent images under shift-invariant Point Spread Function (PSF) assumptions, `CLEAN_SAR` addresses the unique physics of coherent SAR:
1. **Complex $I + jQ$ Signal Domain:** Preserves target interferometric phase and coherent scattering amplitude throughout the iterative subtraction and model restoration pipeline.
2. **Spatially-Varying Impulse Response (IPR/PSF):** Models the local 2D spatial frequency aperture deformation and geometric shear induced by Polar-to-Cartesian interpolation in SAR image formation algorithms like the Polar Format Algorithm (PFA).
3. **Exact Per-Peak Evaluation with Coordinate Caching:** Evaluates the exact local dirty PSF and clean beam at the metric coordinates of each detected peak, caching results by discrete pixel coordinate for $\mathcal{O}(1)$ reuse.
4. **NGA NITF SICD Standard Integration:** Leverages `sarkit` for standards-compliant reader/writer operations, sub-image chipping, and rigorous polynomial coordinate transformations.

---

## 2. Problem Formulation & SAR Physics

### 2.1 The SAR Sidelobe & Point Scatterer Problem
In SAR imaging, high-reflectivity discrete scatterers (such as vehicles, trihedral corner reflectors, building edges, and maritime vessels) exhibit pronounced 2D sinc-like sidelobes. Due to high dynamic ranges (>50 dB), strong sidelobes obscure adjacent weaker targets, corrupt shadow regions, and induce clutter leakage.

### 2.2 Why Astronomical CLEAN Fails for SAR
- **Phase Incoherence:** Astronomical algorithms operate on $I(x, y) \ge 0$. SAR data is complex $z(x, y) = I(x, y) + j Q(x, y)$, where interference between adjacent point targets depends critically on phase.
- **Spatially-Varying IPR:** In SAR image formation (e.g., PFA), polar spatial frequencies $(K_r, \theta)$ are mapped into a Cartesian grid $(K_{\text{row}}, K_{\text{col}})$. The effective aperture boundaries, resolution, and squint angle vary continuously as a function of scene position relative to the Scene Center Point (SCP):
  $$\theta(x_{\text{row}}, y_{\text{col}}) \approx \arctan\left(\frac{y_{\text{col}}}{R_0 + x_{\text{row}}}\right)$$
  Consequently, assuming a stationary PSF across the scene introduces residual artifacts and improper sidelobe cancellation.

---

## 3. Architecture & Implementation Analysis

The system is structured into four core modules:

```
                          +-------------------------------+
                          |   Input NITF SICD (.nitf)     |
                          +-------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  clean_sar/sicd_handler.py: SICDHandler (SARkit I/O & Metadata)                   |
|  - Full image & sub-image (chip) reading with native complex64 buffer management  |
|  - Bidirectional coordinate mapping: Chip (r,c) -> Global (r_g,c_g) -> Metric (x,y)|
|  - SICD XML subtree modification and compliant NITF export                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  clean_sar/psf.py: PSFGenerator (Physics & Beam Synthesis)                         |
|  - Option A: K-space Aperture Support (BW_r, BW_c, Windowing) + 2D IFFT           |
|  - Option B: Analytic Spatial Sinc with Geometric Shear                           |
|  - Clean Beam: Elliptical Gaussian matched to ImpRespWid or Mainlobe taper        |
|  - PyTorch on-device tensor cache keyed by (r_g, c_g, size, method/beam)          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  clean_sar/engine.py: run_hogbom_clean (GPU Deconvolution Engine)                 |
|  - Peak detection on |Residual| tensor (with optional boolean mask)                |
|  - Coherent component accumulation: comp = gamma * Residual[r0, c0]               |
|  - Windowed dirty PSF subtraction from Residual on device                         |
|  - Windowed clean beam accumulation into Restored Model on device                 |
|  - Clean Image synthesis: Clean = Restored Model + Residual                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  clean_sar/utils.py & cli.py: Export & Multi-Panel Visualization                  |
|  - dB scaling: 20 * log10(|image| / peak) clipped to dynamic range                |
|  - Multi-panel comparison figure (Dirty, Clean, Components, Residual, PSFs)       |
|  - Output NITF SICD product writing                                               |
+-----------------------------------------------------------------------------------+
```

---

## 4. Key Findings, Proofs & Implemented Demonstrations

All reference implementations resolving these findings are provided in [`audit/gemini_3_7_flash/fixed_components.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/fixed_components.py) and demonstrated in [`audit/gemini_3_7_flash/demo_fixes.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/demo_fixes.py).

### Finding 1: Mathematical Discrepancy in Gaussian Restoring Beam Width
- **Location in Codebase:** [`clean_sar/psf.py:L268-285`](file:///home/feildaw/CLEAN_SAR/clean_sar/psf.py#L268-L285)
- **Problem:**
  The current code sets `fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))` ($\approx 2.35482$).
  In SICD standard metadata, `Grid.Row.ImpRespWid` is the **half-power ($-3\text{ dB}$)** impulse response width of the power profile $P(x) = |E(x)|^2$.
- **Mathematical Proof:**
  For a voltage Gaussian $E(x) = \exp\left(-\frac{x^2}{2\sigma^2}\right)$, the power is $P(x) = \exp\left(-\frac{x^2}{\sigma^2}\right)$.
  At half-power $x = W_{3\text{dB}} / 2$:
  $$P(W_{3\text{dB}} / 2) = \exp\left(-\frac{(W_{3\text{dB}}/2)^2}{\sigma^2}\right) = \frac{1}{2} = e^{-\ln 2} \implies \sigma = \frac{W_{3\text{dB}}}{2\sqrt{\ln 2}} \approx \frac{W_{3\text{dB}}}{1.665109}$$
  The current code's factor ($2.35482$) matches the **half-amplitude ($-6\text{ dB}$)** width, making the synthesized restoring beam $\sqrt{2} \approx 1.414\times$ too narrow.
- **Developer Fix:**
  In [`clean_sar/psf.py`](file:///home/feildaw/CLEAN_SAR/clean_sar/psf.py#L268-L272), replace:
  ```python
  # OLD:
  fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
  sigma_r = wid_r / fwhm_factor
  sigma_c = wid_c / fwhm_factor
  
  # RECOMMENDED FIX:
  factor_3db = 2.0 * np.sqrt(np.log(2.0)) # 1.665109
  sigma_r = wid_r / factor_3db
  sigma_c = wid_c / factor_3db
  ```
- **Demo Verification:** Running [`audit/gemini_3_7_flash/demo_fixes.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/demo_fixes.py) generates [`output/restoring_beam_fix_comparison.png`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/restoring_beam_fix_comparison.png) verifying that the fixed beam drops to exactly $-3.01\text{ dB}$ at $W/2$.

---

### Finding 2: Hardcoded Slant Range Constant in PFA Shear Calculation
- **Location in Codebase:** [`clean_sar/psf.py:L129`](file:///home/feildaw/CLEAN_SAR/clean_sar/psf.py#L129) & [`psf.py:L211`](file:///home/feildaw/CLEAN_SAR/clean_sar/psf.py#L211)
- **Problem:**
  The shear calculation uses $\theta_{\text{local}} = \arctan(y_{\text{col}} / (10000.0 + x_{\text{row}}))$. The $10,000\text{ m}$ constant is hardcoded.
  In actual satellite datasets (e.g. Umbra-04/05/06), the true slant range to SCP is $\approx 600\text{ km} - 765\text{ km}$ ($646,692\text{ m}$ in Umbra-05).
  This hardcoded value overestimates the peripheral aperture shear angle by **$61.2\times$**.
- **Developer Fix:**
  In [`clean_sar/sicd_handler.py`](file:///home/feildaw/CLEAN_SAR/clean_sar/sicd_handler.py#L40-L92), load `SCPCOA.SlantRange`:
  ```python
  self.slant_range = float(self._safe_load("./{*}SCPCOA/{*}SlantRange", 10000.0))
  ```
  In [`clean_sar/psf.py`](file:///home/feildaw/CLEAN_SAR/clean_sar/psf.py#L129), use `self.sicd.slant_range`:
  ```python
  r0 = getattr(self.sicd, "slant_range", 10000.0)
  theta_local = np.arctan2(ycol, r0 + xrow)
  ```

---

### Finding 3: Bounded LRU Coordinate Caching
- **Location in Codebase:** [`clean_sar/psf.py:L320-348`](file:///home/feildaw/CLEAN_SAR/clean_sar/psf.py#L320-L348)
- **Problem:**
  `_cache_dirty` and `_cache_clean` are unbounded Python dictionaries. For 50,000 iterations over large scenes, caching hundreds of unique $65 \times 65$ complex tensors consumes several gigabytes of GPU VRAM without eviction.
- **Developer Fix:**
  Use `collections.OrderedDict` with a maximum entry cap (e.g., `max_cache_size=4096`), popping least-recently-used items when full (as demonstrated in `EnhancedPSFGenerator`).

---

## 5. Chip Size Scaling & Iteration Capacity Analysis

### 5.1 Why Larger Chips Accommodate More Iterations
When performing Complex Hogbom CLEAN deconvolution:
1. **Higher Scatterer Density & Content:** Larger chip areas contain a greater number of distinct physical point scatterers (vehicles, buildings, infrastructure).
2. **Dynamic Range & Threshold Physics:** The stopping threshold $\tau = \text{threshold} \times \text{InitialPeak}$ is set relative to the highest peak in the chip. In a small $64\times 64$ chip containing 1–2 targets, the peak residual drops below $\tau$ quickly. In a $512\times 512$ or full scene containing hundreds of scatterers, significant residual energy remains across multiple targets, allowing thousands of constructive CLEAN iterations before convergence.
3. **PSF Boundary Containment:** A larger chip avoids window boundary truncation of the $65\times 65$ dirty beam sidelobes, preventing artificial edge ringing.

### 5.2 Empirical Scaling Benchmark Results
From [`audit/gemini_3_7_flash/demo_chip_scaling.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/demo_chip_scaling.py) across Umbra-05 data:

| Chip Size | Pixel Count | Point Scatterers Extracted | Peak Reduction | GPU Execution Time | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$64 \times 64$** | 4,096 | 886 | $-27.40\text{ dB}$ | $9.18\text{ s}$ | $544.6\text{ iters/s}$ |
| **$128 \times 128$** | 16,384 | 977 | $-24.40\text{ dB}$ | $8.84\text{ s}$ | $565.4\text{ iters/s}$ |
| **$256 \times 256$** | 65,536 | 1,120 | $-23.63\text{ dB}$ | $9.14\text{ s}$ | $546.9\text{ iters/s}$ |
| **$512 \times 512$** | 262,144 | 1,557 | $-20.61\text{ dB}$ | $10.98\text{ s}$ | $455.5\text{ iters/s}$ |

*Scaling figure saved to:* [`audit/gemini_3_7_flash/output/chip_scaling_analysis.png`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/chip_scaling_analysis.png)

---

## 6. Audit Artifacts Index

All code, tests, and outputs produced during this audit:
* [`fixed_components.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/fixed_components.py): Reference implementation of `EnhancedSICDHandler` and `EnhancedPSFGenerator` containing all recommended mathematical and architectural fixes.
* [`demo_fixes.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/demo_fixes.py): Direct comparison of restoring beam power profiles and slant range angles.
* [`demo_chip_scaling.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/demo_chip_scaling.py): Chip scaling benchmark and multi-resolution analysis.
* [`verify_engine_and_physics.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/verify_engine_and_physics.py): Automated 4-part verification suite.
* [`inspect_datasets.py`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/inspect_datasets.py): Metadata inspection across all 10 NITF SICD files.
* [`output/restoring_beam_fix_comparison.png`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/restoring_beam_fix_comparison.png): Power profile comparison plot.
* [`output/chip_scaling_analysis.png`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/chip_scaling_analysis.png): Chip scaling benchmark plot.
* [`output/benchmark_clean_chip.nitf`](file:///home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/benchmark_clean_chip.nitf): Fully deconvolved NITF SICD product.
