# CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution with Spatially-Varying IPR

**CLEAN_SAR** is a GPU-accelerated Python framework for performing **Complex Hogbom CLEAN deconvolution** on Synthetic Aperture Radar (SAR) imagery stored in **NITF SICD** (Sensor Independent Complex Data) format.

It computes the **exact spatially-varying Impulse Response (IPR / Point Spread Function)** for every detected peak position across the scene, using **SARkit** for SICD metadata/image handling and **PyTorch** for CUDA-accelerated deconvolution.

---

## 1. Project Motivation & Goals

### The Problem
Traditional radio astronomy CLEAN implementations operate on real-valued, positive intensity maps under the assumption of a shift-invariant PSF. However, SAR imagery presents unique physics:
1. **Complex Signals ($I + jQ$)**: SAR data contains vital phase information representing target distances, interferometry, and coherent scattering. Deconvolution must operate directly in the complex domain.
2. **Spatially-Varying IPR (PSF)**: In SAR image formation (e.g., Polar Format Algorithm / PFA), converting polar spatial frequencies $(K_r, \theta)$ to Cartesian coordinates $(K_{rg}, K_{az})$ causes the effective spatial frequency support, squint angle, and resolution to vary continuously as a function of scene position $(x, y)$ relative to the Scene Center Point (SCP): $\theta(x_{\text{row}}, y_{\text{col}}) = \arctan\left(\frac{y_{\text{col}}}{R_0 + x_{\text{row}}}\right)$ where $R_0$ is the dynamic slant range from `SCPCOA.SlantRange`.
3. **Format Standardization**: Native compliance with the official NGA **NITF SICD standard** using SARkit.

---

## 2. Quickstart Demo

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

## 3. General CLI Usage

For processing arbitrary SICD NITF files:

```bash
python -m clean_sar.cli \
  -i /home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf \
  --chip "1932,3931,2188,4187" \
  --method kspace \
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
| `--plot` | Path to save comparison PNG figure | `output/<name>_comparison.png` |
| `--chip` | Sub-image bounding box: `"start_row,start_col,stop_row,stop_col"` | Full Scene |
| `--method` | Exact PSF generator: `kspace` (Option A) or `analytic` (Option B) | `kspace` |
| `--beam` | Restoring beam: `gaussian` (matched 3dB width) or `mainlobe` | `gaussian` |
| `--gain` | Loop damping factor $\gamma$ | `0.1` |
| `--threshold` | Stopping threshold (fraction of initial peak if $< 1.0$) | `0.02` |
| `--max-iters` | Maximum CLEAN iterations | `2500` |
| `--psf-size` | PSF kernel dimensions ($N \times N$, odd) | `65` |
| `--guard-margin` | Margin in pixels excluded from peak selection | `0` |
| `--dyn-range` | Dynamic range in dB for visualization | `50.0` |
| `--ref-val` | Reference magnitude for 0 dB normalization | Chip local max |

---

## 4. Python API

```python
from clean_sar import SICDHandler, PSFGenerator, run_hogbom_clean, plot_clean_comparison

# 1. Open SICD file and extract sub-image chip
handler = SICDHandler("path/to/image_SICD.nitf")
chip, chip_xml = handler.read_chip(start_row=1932, start_col=3931, stop_row=2188, stop_col=4187)

# 2. Instantiate PSF generator
psf_gen = PSFGenerator(handler)

# 3. Run Complex Hogbom CLEAN with exact per-peak PSF computation
result = run_hogbom_clean(
    dirty_image=chip,
    psf_generator=psf_gen,
    method="kspace",
    beam_type="gaussian",
    psf_size=65,
    gain=0.1,
    threshold=0.02,
    max_iters=2500,
    chip_origin=(1932, 3931),
    device="cuda",
    verbose=True,
)

# 4. Save deconvolved SICD NITF
handler.write_nitf("output/clean_chip.nitf", result.clean_image, custom_xmltree=chip_xml)
```

---

## 5. Running Tests

Run the complete test suite using `pytest`:

```bash
pytest -v tests/
```

---

## 6. License
MIT License
