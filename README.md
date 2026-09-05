# CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution with Spatially-Varying IPR

**CLEAN_SAR** is a GPU-accelerated Python framework for performing **Complex Hogbom CLEAN deconvolution** on Synthetic Aperture Radar (SAR) imagery stored in **NITF SICD** (Sensor Independent Complex Data) format.

It computes the **exact spatially-varying Impulse Response (IPR / Point Spread Function)** for every detected peak position across the scene, using **SARkit** for SICD metadata/image handling and **PyTorch / C++ CUDA** for accelerated deconvolution.

Designed for headless execution in cloud and Kubernetes environments with zero mandatory graphics dependencies.

---

## 1. Project Motivation & Goals

### The Problem
Traditional radio astronomy CLEAN implementations operate on real-valued, positive intensity maps under the assumption of a shift-invariant PSF. However, SAR imagery presents unique physics:
1. **Complex Signals ($I + jQ$)**: SAR data contains vital phase information representing target distances, interferometry, and coherent scattering. Deconvolution operates directly in the complex domain.
2. **Spatially-Varying IPR (PSF)**: In SAR image formation (e.g., Polar Format Algorithm / PFA), converting polar spatial frequencies $(K_r, \theta)$ to Cartesian coordinates $(K_{rg}, K_{az})$ causes the effective spatial frequency support, squint angle, and resolution to vary continuously as a function of scene position $(x, y)$ relative to the Scene Center Point (SCP):
   $$\theta(x_{\text{row}}, y_{\text{col}}) = \arctan\left(\frac{y_{\text{col}}}{R_0 + x_{\text{row}}}\right)$$
   where $R_0$ is the dynamic slant range from `SCPCOA.SlantRange`.
   - **Spaceborne Regime**: At spaceborne ranges ($R_0 \sim 500\text{--}800\text{ km}$), angular variation across the scene is a modest $\sim 0.26\%$ effect.
   - **Airborne Regime**: At short airborne ranges ($R_0 \sim 10\text{--}30\text{ km}$), angular variation reaches $5\text{--}25\%$, making exact spatially-varying IPR rotation indispensable.
   - *Note on defocus*: Wide-angle PFA wavefront-curvature phase errors away from the SCP are addressed during formation/refocusing (e.g., DiffPFA); CLEAN_SAR accounts for the exact rigid spatial rotation of the resulting IPR.
3. **Format Standardization**: Native compliance with the official NGA **NITF SICD standard** using SARkit.

---

## 2. Backend Capabilities & Support Matrix

| Feature / Option | PyTorch GPU (`pytorch`) | Native CUDA GPU (`cuda`) | Notes |
| :--- | :---: | :---: | :--- |
| **Gaussian Restoring Beam** | Yes | Yes | Matched to 3 dB half-power width ($\text{FWHM} = \text{ImpRespWid}$) |
| **Mainlobe Restoring Beam** | Yes | *Planned* (`NotImplementedError`) | First-null contour mask; abrupt boundary |
| **Arbitrary Clean Mask** | Yes | *Planned* (`NotImplementedError`) | Constrains component search to ROI |
| **Weighting Windows** | Uniform, Taylor, Hamming, Hann | Uniform, Taylor, Hamming, Hann | Fused on-chip analytic evaluation in CUDA |
| **JIT Architecture** | PyTorch / LibTorch | Dynamic NVRTC (`compute_XX`) | Autodetects GPU compute capability |

---

## 3. Quickstart Demo

You can run the out-of-the-box demo with pre-configured default paths for the **`2023-11-14-03-38-20_UMBRA-04`** dataset (evaluating both raw Umbra and DiffPFA outputs):

```bash
# Run both datasets with defaults
python demo_clean.py

# Or run only one target
python demo_clean.py --target umbra
python demo_clean.py --target diffpfa
```

Output products (deconvolved NITF files and 6-panel dB comparison plots) are saved automatically to:
- `output/umbra_20231114/`
- `output/diffpfa_20231114/`

---

## 4. Python API (`CLEANProcessor`)

The primary interface is **`CLEANProcessor`**, which encapsulates SICD loading, chip/scene management, exact PSF dispatch, deconvolution, and NITF writing:

```python
from clean_sar import CLEANProcessor

# 1. Instantiate processor with input & output NITF paths
processor = CLEANProcessor(
    input_path="/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf",
    output_path="output/clean_image.nitf",
    chip_bounds=(1932, 3931, 2188, 4187),  # Optional: (r_min, c_min, r_max, c_max) or None for full scene
    backend="auto",                         # 'auto', 'pytorch', or 'cuda'
    device="cuda",                          # Optional: 'cuda' or 'cpu' (auto-detected)
)

# 2. Run deconvolution (executes algorithm and writes output NITF automatically)
result = processor.run(
    gain=0.1,
    threshold=0.02,
    max_iters=2500,
    beam_type="gaussian",  # 'gaussian' (matched 3dB width) or 'mainlobe'
    psf_size=65,
    verbose=True,
)

print(f"Iterations: {result.iterations}, Peak reduction: {result.peak_reduction_db:.1f} dB")

# 3. Assess point-target quality (mainlobe preservation and ISLR reduction)
from clean_sar import ipr_quality_multi, verdict

quality = ipr_quality_multi(
    dirty=processor.handler.read_full_image(),
    clean=result.clean_image,
    row_wid=processor.handler.row_wid,
    col_wid=processor.handler.col_wid,
    row_ss=processor.handler.row_ss,
    col_ss=processor.handler.col_ss,
)
print(f"Quality Assessment: {verdict(quality)} (ISLR change: {quality['islr_change_db']:.2f} dB)")
```

---

## 5. General CLI Usage

For processing arbitrary SICD NITF files from the terminal:

```bash
python -m clean_sar.cli \
  -i /home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf \
  --chip "1932,3931,2188,4187" \
  --backend auto \
  --beam gaussian \
  --gain 0.1 \
  --threshold 0.02 \
  --max-iters 2500 \
  -o output/umbra_20231114/custom_clean.nitf \
  --plot output/umbra_20231114/custom_comparison.png
```

#### CLI Parameters
| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path to input NITF SICD file | *Required* |
| `-o, --output` | Path to output deconvolved NITF file | `output/<name>_clean.nitf` |
| `--plot` | Optional path to save comparison PNG figure | None |
| `--chip` | Sub-image bounding box: `"start_row,start_col,stop_row,stop_col"` | Full Scene |
| `--backend` | Compute engine: `auto`, `pytorch`, or `cuda` | `auto` |
| `--beam` | Restoring beam: `gaussian` (matched 3dB width) or `mainlobe` | `gaussian` |
| `--gain` | Loop damping factor $\gamma$ | `0.1` |
| `--threshold` | Stopping threshold (fraction of initial peak if $< 1.0$) | `0.02` |
| `--max-iters` | Maximum CLEAN iterations | `2500` |
| `--psf-size` | PSF kernel dimensions ($N \times N$, odd) | `65` |
| `--guard-margin` | Margin in pixels excluded from peak selection | `0` |
| `--dyn-range` | Dynamic range in dB for visualization | `50.0` |
| `--ref-val` | Reference magnitude for 0 dB normalization | Chip local max |

---

## 6. Tools & Utilities (`tools/`)

| Script | Purpose |
| :--- | :--- |
| **`benchmark_pytorch_vs_cuda.py`** | Head-to-head PyTorch vs. Native CUDA benchmark with warmup and median timing. |
| **`run_benchmark.py`** | Batch full-scene benchmarking across an entire directory of SICDs. |
| **`tools/compare_sicd.py`** | Compares dirty vs. clean SICD NITFs and renders 6-panel dB plots. |
| **`tools/render_benchmark_plots.py`** | Renders multi-panel full scene overview + native resolution zoom plots. |
| **`tools/list_latest_umbra.py`** | Chronological S3 catalog search and downloader for open Umbra CPHD/SICD pairs. |
| **`tools/scan_stepped_chirp.py`** | Fast S3 HTTP byte-range scanner for multi-channel / stepped-chirp CPHDs. |

---

## 7. Installation & Optional Dependencies

Core requirements are minimal for headless environments (no graphics libraries required):
```bash
pip install .
```

To install visualization tools (`matplotlib`, `pillow`):
```bash
pip install ".[viz]"
```

---

## 8. Running Tests

Run the complete test suite using pytest:

```bash
pytest -v tests/
```

To run against a custom directory of NITF SICD files:
```bash
CLEAN_SAR_TEST_DATA=/path/to/sicd/dir pytest -v tests/
```

If no test NITF files or no CUDA GPU are detected, dataset-dependent and CUDA-specific tests automatically skip cleanly with descriptive messages.

---

## 9. License

MIT License
