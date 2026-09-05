# Independent Comprehensive Technical Audit of CLEAN_SAR

**Auditor:** Agy (Gemini 3.8 Flash)  
**Date:** September 2026  
**Target Repository:** `/home/feildaw/CLEAN_SAR`  
**Environment Tested:** Linux (x86_64), Python 3.12.3 (`/home/feildaw/mypyenv`), NVIDIA GeForce RTX 3070 Laptop GPU (Driver 555.97, CUDA 12.5), NVRTC JIT (`libnvrtc.so.12`), SARkit 0.x, PyTorch 2.x.  
**Reference Standards:** NGA.STND.0024-1_1.5 (SICD DIDD), NGA.STND.0024-2_1.5 (SICD FFDD), NGA.STND.0024-3_1.5 (SICD IPDD), XML Schemas (`NGA.STND.0024-4_1.5_Schema.xsd`, `SICD_schema_V1.3.0_2021_11_30.xsd`).

---

## Executive Summary

**CLEAN_SAR** implements Complex Hogbom CLEAN deconvolution on Synthetic Aperture Radar (SAR) imagery stored in NITF SICD format. Its key goals are:
1. Deconvolving complex-valued SAR data ($I + jQ$) to extract point scatterers while preserving coherent phase.
2. Replacing dirty aperture impulse responses (IPR / PSF) with smooth restoring clean beams (Gaussian or mainlobe) to suppress sidelobes.
3. Modeling spatially-varying image formation geometry across the scene based on radar slant range and polar shear angles.
4. Accelerating the computationally demanding 2D argmax and iterative subtraction loops using PyTorch and native CUDA C++ kernels compiled on the fly with `libnvrtc`.

### Key Audit Findings Overview
- **Strengths:** 
  - The core 2D complex CLEAN algorithm in PyTorch and CUDA (for uniform weighting) executes with high numerical parity ($< 6 \times 10^{-7}$ absolute deviation) and exhibits a **5.1x pure compute speedup** in native CUDA over PyTorch.
  - Basic NITF SICD I/O and spatial sub-image chipping via SARkit maintain valid NITF structures and correct coordinate conversions at the Scene Center Point (SCP).
  - Clean beam Gaussian resolution matching accurately achieves the -3.01 dB power half-width specified by `Grid.Row.ImpRespWid` and `Grid.Col.ImpRespWid`.
- **Critical Defects & Discrepancies:**
  - **[P0 - Runtime Crash]** `demo_clean.py` crashes on launch due to mismatched function call signatures with `tools/compare_sicd.py`.
  - **[P0 - Physics & Parity Bug]** CUDA aperture weighting (Hamming and Hann) fails to normalize the dirty PSF center. As a result, CUDA amplifies point target powers by **+10.6 dB (Hamming)** and **+12.0 dB (Hann)**, producing clean images that distort target radar cross-section (RCS) by 340% to 400% and requiring 3.5x more iterations to converge.
  - **[P1 - Silent Feature Drop]** In the CUDA backend, `beam_type="mainlobe"` and `clean_mask` are silently ignored without error or warning. CUDA always falls back to Gaussian beams and ignores spatial masking.
  - **[P1 - Metadata Coordinate Bug]** `CleanPhysicsConfig.from_sicd_handler` ignores `ImageData.FirstRow` and `FirstCol` of the input SICD. Processing an existing chip NITF causes spatial metric coordinates to be offset by kilometers relative to the SCP.
  - **[P1 - Hardware Portability Bug]** NVRTC architecture is hardcoded to `compute_86`. The CUDA backend fails completely on cloud GPUs such as NVIDIA A100 (`compute_80`), V100 (`compute_70`), T4 (`compute_75`), and H100 (`compute_90`). Compilation error logs are swallowed.
  - **[P2 - Metric Misnomer]** The reported `suppression_db` metric measures peak residual convergence ($20 \log_{10}(P_{init} / P_{final})$), not actual point-target sidelobe suppression (PSLR / ISLR).
  - **[P2 - Architectural Duplication]** Two distinct CUDA implementations exist: `clean_sar_cuda.cu` (compiled by `Makefile` to `libcleansar.so`) and `clean_hogbom.cu` (compiled at runtime by `cuda_backend.py`). `libcleansar.so` is never actually used by the Python package, contradicting package documentation.

---

## 1. Mathematical & Physical Modeling Audit

### 1.1 Spatially-Varying IPR Geometry
In `clean_sar/psf.py` and `clean_sar/backends/c_src/clean_hogbom.cu`, the local IPR rotation is calculated as:
$$\theta(x_{row}, y_{col}) = \arctan2\left(y_{col},\, R_0 + x_{row}\right)$$
where $x_{row} = (r_{global} - r_{scp}) \cdot \Delta r$, $y_{col} = (c_{global} - c_{scp}) \cdot \Delta c$, and $R_0 = \text{SCPCOA.SlantRange}$.
The 2D spatial grid coordinates $(U, V)$ are rotated by $\theta$:
$$U' = U \cos\theta + V \sin\theta, \quad V' = -U \sin\theta + V \cos\theta$$

#### Audit Evaluation:
1. **Geometric Coordinate Frame**:
   The transformation matches Step 5 of the NGA SICD Volume 3 (IPDD) projection equations:
   $$\frac{\delta\Phi}{\delta Ka} = rg \cos\theta + az \sin\theta, \quad \frac{\delta\Phi}{\delta Kc} = -rg \sin\theta + az \cos\theta$$
   The sign convention for rotating $(U, V)$ into $(U', V')$ correctly rotates the spatial domain PSF by $+\theta$.
2. **Magnitude of Spatial Variation in Spaceborne Platforms**:
   Our empirical measurements on the Umbra satellite datasets ($R_0 \approx 600\text{--}760\text{ km}$, aperture sizes $5000\text{--}13000$ pixels) revealed:
   - Across a $256 \times 256$ pixel sub-image chip (128 meters), $\Delta\theta \approx 0.012^\circ$. The maximum point-by-point difference in the PSF kernel between opposite corners of the chip is only **$1.49 \times 10^{-4}$** ($-76.5$ dB).
   - Across the entire full scene ($5086 \times 8631$ pixels, $\sim 5\text{ km}$ footprint), $\theta$ reaches at most $\pm 0.235^\circ$, resulting in a maximum PSF kernel difference of **$2.59 \times 10^{-3}$** ($-51.7$ dB).
   - Recomputing the exact rotated PSF on every single iteration inside a localized chip adds significant overhead with negligible physical difference ($< -76$ dB). A block-based or chip-center PSF cache would yield identical numerical results while eliminating LRU lookup overhead.
3. **Absence of Wavefront Curvature Defocusing**:
   In Polar Format Algorithm (PFA) SAR imagery, the dominant degradation away from the SCP is **not** rigid rotation, but **space-variant quadratic and cubic phase error defocusing** caused by wavefront curvature. `CLEAN_SAR` assumes a rigid separable sinc rotation and does not model quadratic phase error defocusing ($e^{j \phi_2(u, v)}$). For refocused imagery (such as DiffPFA outputs), defocusing is already corrected, making the rigid sinc assumption appropriate.

---

### 1.2 Aperture Weighting & Normalization Bug (CRITICAL)

#### The Math:
When a weighting window $w(k)$ is applied to the spatial frequency aperture of bandwidth $B$:
$$w(k) = a_0 + 2 \sum_{m=1}^{M} a_m \cos(2\pi m k / B)$$
The spatial-domain impulse response is the inverse Fourier transform:
$$\text{IPR}(x) = a_0 \text{sinc}(B x) + \sum_{m=1}^{M} a_m \left[\text{sinc}(B(x - m/B)) + \text{sinc}(B(x + m/B))\right]$$
At the target center ($x = 0$):
$$\text{sinc}(0) = 1.0, \quad \text{sinc}(\pm m) = 0 \quad (\forall m \in \mathbb{Z}^+)$$
Consequently:
$$\text{IPR}(0) = a_0$$

#### The Defect:
1. **In Python (`psf.py:L121-125`)**:
   ```python
   psf = psf_r * psf_a
   max_idx = (psf_size // 2, psf_size // 2)
   peak_val = psf[max_idx]
   if np.abs(peak_val) > 0:
       psf = psf / peak_val
   ```
   The Python generator divides by $\text{IPR}(0)^2$, ensuring the dirty PSF has a peak value of **$1.0$**.
2. **In CUDA (`clean_hogbom.cu:L46-52`, `psf_math.cuh:L43-49`)**:
   ```c
   else if (wgt_type == 2) { // HAMMING
       return 0.54f * dev_sinc(x) + 0.23f * (dev_sinc(x - 1.0f) + dev_sinc(x + 1.0f));
   } else if (wgt_type == 3) { // HANN
       return 0.50f * dev_sinc(x) + 0.25f * (dev_sinc(x - 1.0f) + dev_sinc(x + 1.0f));
   }
   ```
   At $x = 0$, `eval_1d_window_sinc` returns **$0.54$** for Hamming and **$0.50$** for Hann.
   In 2D:
   $$h_{dirty}(0, 0) = 0.54 \times 0.54 = \mathbf{0.2916} \quad \text{(Hamming)}$$
   $$h_{dirty}(0, 0) = 0.50 \times 0.50 = \mathbf{0.2500} \quad \text{(Hann)}$$
   In `fused_clean_sub_add_kernel`:
   ```c
   d_residual[idx].real -= comp_re * h_dirty;   // Subtracts 0.2916 * comp
   d_model[idx].real    += comp_re * h_clean;   // Adds 1.0000 * comp
   d_components[idx].real += comp_re;          // Adds 1.0000 * comp
   ```
3. **Impact on Deconvolution Results**:
   We ran an empirical audit on a point target of amplitude $10.0$ with Hamming weighting:
   - **PyTorch**: Extracted component $= 9.903$, Restored Clean Peak $= \mathbf{10.000}$ (Exact).
   - **CUDA**: Extracted component $= 33.955$, Restored Clean Peak $= \mathbf{34.053}$ (**+10.64 dB error**).
   - **Convergence**: CUDA took **156 iterations** vs. PyTorch's **44 iterations** because the effective subtraction factor was reduced by $70.8\%$.

---

### 1.3 Restoring Clean Beam Physics

#### Gaussian Clean Beam:
In `psf.py:L179-183`:
```python
fwhm_const = 2.0 * np.sqrt(np.log(2.0))
sigma_r = self.config.row_wid / fwhm_const
sigma_a = self.config.col_wid / fwhm_const
beam = np.exp(-0.5 * ((U_prime / sigma_r)**2 + (V_prime / sigma_a)**2))
```
- For a Gaussian amplitude profile $E(x) = \exp(-x^2 / (2\sigma^2))$, the power is $P(x) = |E(x)|^2 = \exp(-x^2 / \sigma^2)$.
- At half-power ($P(x) = 0.5$), $x_{half} = \sigma \sqrt{\ln 2}$.
- The full width at half-power (3 dB width) is $W = 2 x_{half} = 2 \sqrt{\ln 2} \sigma$.
- Thus $\sigma = W / (2 \sqrt{\ln 2})$.
- `fwhm_const = 2.0 * np.sqrt(np.log(2.0))` ($\approx 1.6651$) is mathematically correct for complex voltage amplitude where resolution is defined in dB power.

#### Mainlobe Clean Beam Truncation:
In `psf.py:L144-161`:
The flood-fill algorithm collects pixels down to $|dirty(nr, nc)| > 0.1$ ($-20$ dB). Below $-20$ dB, the beam is sharply truncated to zero, introducing high-frequency boundary ring artifacts into the restored image.

#### Mainlobe Beam Omission in CUDA:
In `clean_hogbom.cu`, line 231 unconditionally executes `eval_clean_beam_gaussian`. The kernel has no parameter for `beam_type`. Passing `beam_type="mainlobe"` to the CUDA backend produces a Gaussian beam without notification.

---

## 2. Standards Compliance & Metadata Integrity (NGA SICD 1.5)

### 2.1 Dataset Compliance Verification
We validated all 12 available Umbra and DiffPFA SICD NITF datasets against both `schemas/NGA.STND.0024-4_1.5_Schema.xsd` and `schemas/SICD_schema_V1.3.0_2021_11_30.xsd`:
- **Older Umbra Collections** (`2023-*_UMBRA-*_SICD.nitf`): Use namespace `urn:SICD:1.2.1`. They fail validation against Schema 1.5 and 1.3 due to root tag namespace mismatch.
- **DiffPFA & Newer Umbra Collections** (`*_SICDU_*.nitf`): Use namespace `urn:SICD:1.3.0`. They validate as **100% compliant** against Schema 1.3.
- **Weighting Types**: In all datasets except `2025-10-26_UMBRA-08`, `Grid.Row.WgtType` and `Grid.Col.WgtType` are omitted (`None`). In accordance with NGA SICD Volume 1 (DIDD) Table 3-4, when `WgtType` is absent, the aperture is unweighted (`UNIFORM`), which `CLEAN_SAR` correctly handles as the default.

### 2.2 Sub-Image Chipping Coordinate Bug
In NGA SICD Volume 1 (DIDD) Section 2.4 and Table 3-3:
- For any sub-image (chip), `ImageData.FirstRow` and `FirstCol` denote the global row and column offset of the sub-image relative to the full image collection.
- `ImageData.SCPPixel` **always remains referenced to the global full image coordinate system**.
- Metric coordinates relative to SCP are defined as:
  $$x_{row} = (r_{global} - \text{SCPPixel.Row}) \cdot SS_{row}$$
  $$r_{global} = r_{chip} + \text{FirstRow}$$

#### Defect in `clean_sar/config.py:L28-44`:
```python
@classmethod
def from_sicd_handler(cls, handler, chip_start: Tuple[int, int] = (0, 0)):
    return cls(
        ...
        scp_row=float(handler.scp_pixel[0]),
        scp_col=float(handler.scp_pixel[1]),
        chip_start_row=int(chip_start[0]),
        chip_start_col=int(chip_start[1]),
    )
```
When `chip_bounds=None` is passed to `CLEANProcessor` on an already-chipped NITF file (e.g. `FirstRow = 1932`):
- `chip_start` defaults to `(0, 0)`.
- `config.chip_start_row` is set to `0` instead of `handler.first_row` ($1932$).
- Metric coordinates from SCP are computed using $r_{chip} - scp\_row$ instead of $(r_{chip} + \text{FirstRow}) - scp\_row$.
- This shifts the computed scene location by kilometers, evaluating the PSF with an incorrect geometry.

### 2.3 Output Metadata Provenance
When exporting deconvolved NITFs via `SICDHandler.write_nitf`:
- SARkit updates `ImageData.NumRows`, `NumCols`, `FirstRow`, and `FirstCol` properly.
- However, `ImageCreation.Application` and `ImageCreation.DateTime` are not updated, leaving the deconvolved output tagged with the original SAR image formation software and collection timestamp.
- The standard NITF header fields `ostaid="CLEAN_SAR"` and `isorce="CLEAN_SAR_DECONVOLVED"` are correctly inserted.

---

## 3. Implementation & Architecture Audit

### 3.1 Head-to-Head Performance (PyTorch vs CUDA)
We benchmarked 500 iterations of Hogbom CLEAN on a $512 \times 512$ pixel chip on the NVIDIA GeForce RTX 3070 Laptop GPU:

| Metric | PyTorch Backend | Native CUDA Backend (NVRTC) | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Pure Compute Time** | 938.98 ms | 184.55 ms | **5.09x Speedup** |
| **Latency per Iteration** | 1.878 ms / iter | 0.369 ms / iter | **-1.509 ms / iter** |
| **Total Pipeline Time** | 1.308 s | 0.670 s | **1.95x Faster** |
| **Numerical Max Diff (Clean Image)** | — | — | **$5.33 \times 10^{-7}$** |
| **Reported Peak Suppression** | 27.98 dB | 27.98 dB | Identical |

#### Architectural Analysis:
- **PyTorch Bottlenecks**:
  - `torch.max(torch.abs(residual))` allocates an $H \times W$ float tensor every iteration.
  - Three synchronous host transfers per iteration (`max_val.item()`, `flat_idx.item()`, etc.) stall the GPU pipeline.
  - Python dictionary hashing and LRU cache queries in `get_psfs_torch` run on the CPU during the tight inner loop.
- **CUDA Advantages**:
  - Fused parallel warp/block argmax reduction kernels (`reduce_max_pass1_kernel` and `reduce_max_pass2_kernel`) reduce memory traffic to 64 blocks without full-image temporary allocations.
  - Subtraction and restoration occur in a single fused 2D kernel pass (`fused_clean_sub_add_kernel`).
- **CUDA Optimization Opportunities**:
  - CUDA still performs three synchronous device-to-host `cuMemcpyDtoH_v2` transfers per iteration. Keeping the loop state on-device (via an asynchronous persistent kernel or CUDA Graphs) could reduce latency from 0.37 ms/iter to $< 0.05$ ms/iter.

---

### 3.2 Portability & Error Handling in CUDA Backend

1. **Hardcoded Compute Architecture**:
   In `cuda_backend.py:L95`:
   `opts = [b"--std=c++14", b"--gpu-architecture=compute_86"]`
   The compute architecture is hardcoded to `compute_86` (RTX 30-series). On an NVIDIA A100 (`sm_80`), V100 (`sm_70`), or H100 (`sm_90`), `cuModuleLoadData` fails with `CUDA_ERROR_INVALID_BINARY`.
2. **Silent Failure & Missing Compilation Logs**:
   In `cuda_backend.py:L97-100`:
   If `nvrtcCompileProgram` fails, the code returns `False` without querying `nvrtcGetProgramLog`. The user receives a misleading `NotImplementedError: Native C++/CUDA backend (libcleansar.so) is not compiled`.
3. **GPU Memory Leaks**:
   In `cuda_backend.py:L170-194`:
   Device memory buffers (`d_residual`, `d_model`, `d_components`, `d_clean`) are allocated without a `try...finally` block. If an unhandled exception occurs inside the deconvolution loop, device memory is leaked until the Python process exits.

---

## 4. API & Tooling Defect Analysis

### 4.1 Bug in Quickstart Demo Script (`demo_clean.py`)
In `demo_clean.py:L76-82`:
```python
plot_comparison(
    input_path,
    out_nitf,
    chip_bounds=chip_bounds,
    save_path=out_png,
    dynamic_range_db=dyn_range,
)
```
`plot_comparison` in `tools/compare_sicd.py` has the signature:
```python
def plot_comparison(
    dirty_image: np.ndarray,
    clean_image: np.ndarray,
    residual_image: Optional[np.ndarray] = None,
    restored_model: Optional[np.ndarray] = None,
    output_png: str = "clean_comparison.png",
    dyn_range_db: float = 50.0,
    title_suffix: str = "",
    ref_val: Optional[float] = None,
) -> None:
```
Passing string file paths and unsupported keywords causes `TypeError: bad operand type for abs(): 'str'` and `unexpected keyword argument 'save_path'`. `demo_clean.py` fails immediately upon execution.

### 4.2 Ambiguity in `clean_mask`
`run_hogbom_cuda_native` in `cuda_backend.py` accepts `clean_mask: Optional[np.ndarray] = None`, but never transfers it to GPU or applies it in the reduction kernel. Masks are silently ignored when using the CUDA backend.

---

## 5. Summary of Identified Defects & Recommendations

| ID | Severity | File(s) Affected | Description | Recommended Remediation |
| :---: | :---: | :--- | :--- | :--- |
| **BUG-01** | **P0 (Crash)** | `demo_clean.py:L76-83` | Incorrect argument types and keywords passed to `plot_comparison`. Quickstart script crashes. | Load images via `SICDHandler.read_chip()` and pass arrays with `output_png` and `dyn_range_db`. |
| **BUG-02** | **P0 (Physics)** | `clean_hogbom.cu:L46-52`<br>`psf_math.cuh:L43-49` | CUDA dirty PSF for Hamming/Hann weighting is not normalized to 1.0. Inflates point target power by +10.6 to +12 dB. | Divide `psf_r` and `psf_c` by $a_0$ ($0.54$ for Hamming, $0.50$ for Hann) or normalize $h_{dirty}$ at peak. |
| **BUG-03** | **P1 (Logic)** | `clean_hogbom.cu:L231`<br>`cuda_backend.py:L281` | CUDA backend ignores `beam_type="mainlobe"` and silently defaults to Gaussian. | Add `beam_type` parameter to CUDA kernel or raise `NotImplementedError` when `beam_type="mainlobe"`. |
| **BUG-04** | **P1 (Logic)** | `cuda_backend.py:L145` | `clean_mask` parameter accepted by CUDA entrypoint but silently ignored during reduction. | Pass boolean mask buffer to `reduce_max_pass1_kernel` or check mask in kernel. |
| **BUG-05** | **P1 (Standards)**| `clean_sar/config.py:L28-44` | `ImageData.FirstRow` and `FirstCol` ignored when building config. Pre-chipped input files are geographically misplaced. | Add `handler.first_row` and `handler.first_col` to `chip_start_row` and `chip_start_col`. |
| **BUG-06** | **P1 (Platform)** | `cuda_backend.py:L95` | NVRTC architecture hardcoded to `compute_86`. Incompatible with A100, V100, H100 GPUs. | Query device compute capability via `cuDeviceGetAttribute` and pass `f"compute_{major}{minor}"`. |
| **BUG-07** | **P2 (Memory)** | `cuda_backend.py:L170-397`| CUDA device memory allocations not wrapped in `try...finally`. Leaks VRAM on loop exception. | Wrap deconvolution loop in `try...finally` block ensuring `cuMemFree_v2` is always invoked. |
| **BUG-08** | **P2 (Driver)** | `cuda_backend.py:L77` | Creates separate non-primary CUDA context using `cuCtxCreate_v2` instead of sharing primary context. | Use `cuDevicePrimaryCtxRetain` for seamless interop with PyTorch. |
| **BUG-09** | **P2 (Clarity)** | `algorithm.py:L37-40` | `suppression_db` calculates residual peak reduction, not true point target sidelobe suppression (PSLR). | Rename metric to `residual_reduction_db` and compute true PSLR/ISLR before and after deconvolution. |
| **BUG-10** | **P3 (Cleanup)** | `backends/c_src/Makefile`<br>`backends/__init__.py:L9`| `libcleansar.so` is built by Makefile but unused by Python package, which compiles `clean_hogbom.cu` via NVRTC. | Remove unused `Makefile` and `clean_sar_cuda.cu` or align build documentation with NVRTC runtime. |

