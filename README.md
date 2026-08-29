# CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution with Spatially-Varying IPR

**CLEAN_SAR** is a GPU-accelerated Python framework for performing **Complex Hogbom CLEAN deconvolution** on Synthetic Aperture Radar (SAR) imagery stored in **NITF SICD** (Sensor Independent Complex Data) format.

It computes the **exact spatially-varying Impulse Response (IPR / Point Spread Function)** for every detected peak position across the scene, using **SARkit** for SICD metadata/image handling and **PyTorch** for CUDA-accelerated deconvolution.

---

## 1. Project Motivation & Goals

### The Problem
Traditional radio astronomy CLEAN implementations operate on real-valued, positive intensity maps under the assumption of a shift-invariant PSF. However, SAR imagery presents unique physics:
1. **Complex Signals ($I + jQ$)**: SAR data contains vital phase information representing target distances, interferometry, and coherent scattering. Deconvolution must operate directly in the complex domain.
2. **Spatially-Varying IPR (PSF)**: In SAR image formation (e.g., Polar Format Algorithm / PFA), converting polar spatial frequencies $(K_r, \theta)$ to Cartesian coordinates $(K_{rg}, K_{az})$ causes the effective spatial frequency support, squint angle, and resolution to vary continuously as a function of scene position $(x, y)$ relative to the Scene Center Point (SCP).
3. **Format Standardization**: Transitioning legacy prototypes from FITS astronomy files to the official NGA **NITF SICD standard**.

### Authoritative Architecture
- **Exact Per-Peak PSF Computation**: On every CLEAN iteration, when a peak $(r_0, c_0)$ is identified in the residual, its exact global pixel and metric $(x_{\text{row}}, y_{\text{col}})$ coordinates are resolved. The exact dirty PSF $h_{(r_0, c_0)}$ and clean beam $h_{\text{clean}, (r_0, c_0)}$ are evaluated for that precise position.
- **Coordinate Caching**: Exact PSFs are cached by pixel coordinate, providing $\mathcal{O}(1)$ lookup on subsequent iterations for bright scatterers.
- **Exact Point-by-Point Clean Beam Restoration**: Each identified component $\Delta c = \gamma A$ accumulates its local Gaussian clean beam directly into the restored model without tiling boundaries or approximation seams.

---

## 2. Architecture & How It Works

```
                        +----------------------------+
                        |       NITF SICD File       |
                        +----------------------------+
                                      |
                                      v
                        +----------------------------+
                        |  SICDHandler (SARkit I/O)  |
                        |  - Full Scene / Chip Read  |
                        |  - Metric Coordinates      |
                        |  - PFA / Grid Metadata     |
                        +----------------------------+
                                      |
                                      v
                        +----------------------------+
                        |  Complex CLEAN Engine      |
                        |  - PyTorch CUDA Execution  |
                        +----------------------------+
                           /                      \
          (Iteration Peak /                        \ (Exact Dirty & Clean
           Global Pixel) /                          \  Beams on Device)
                        v                            v
             +-----------------------------------------------+
             |                PSFGenerator                   |
             | - Option A: k-space 2D Aperture + IFFT        |
             | - Option B: Spatial Analytic Sinc with Shear  |
             | - Matched Gaussian 3dB Restoring Beam         |
             | - Exact Pixel Coordinate Cache                |
             +-----------------------------------------------+
                                      |
                                      v
                        +----------------------------+
                        |       Output Products      |
                        | - Deconvolved SICD NITF    |
                        | - Clean Point Scatterers   |
                        | - Multi-panel dB Plots     |
                        +----------------------------+
```

---

## 3. Installation

Activate your Python virtual environment (e.g., `/home/feildaw/mypyenv`) and install in editable mode:

```bash
source /home/feildaw/mypyenv/bin/activate
cd /home/feildaw/CLEAN_SAR
pip install -e .
```

### Dependencies
- `python >= 3.9`
- `torch` (with CUDA support)
- `sarkit` (NGA SICD / CPHD standard library)
- `numpy`, `scipy`, `matplotlib`, `lxml`

---

## 4. Usage

### Command Line Interface (CLI)

```bash
python run_clean.py \
  -i /home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf \
  --chip "2777,6410,3033,6666" \
  --method kspace \
  --beam gaussian \
  --gain 0.1 \
  --threshold 0.02 \
  --max-iters 2500 \
  -o output/umbra_20230730/2023-07-30-17-19-39_UMBRA-05_SICD_chip_clean.nitf \
  --plot output/umbra_20230730/2023-07-30-17-19-39_UMBRA-05_SICD_chip_comparison.png
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
| `--dyn-range` | Dynamic range in dB for visualization | `50.0` |

---

### Python API

```python
from clean_sar import SICDHandler, PSFGenerator, run_hogbom_clean, plot_clean_comparison

# 1. Open SICD file and extract sub-image chip
handler = SICDHandler("path/to/image_SICD.nitf")
chip, chip_xml = handler.read_chip(start_row=2777, start_col=6410, stop_row=3033, stop_col=6666)

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
    chip_origin=(2777, 6410),
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

Test coverage includes:
- `test_exact_spatially_varying_clean_synthetic`: Exact per-peak spatially varying deconvolution on GPU.
- `test_exact_spatially_varying_clean_analytic`: Analytic exact per-peak deconvolution.
- `test_psf_kspace_generation`: Option A $K$-space aperture support, symmetry, and DC alignment.
- `test_psf_analytic_generation`: Option B spatial sinc response.
- `test_clean_beam_generation`: Gaussian and mainlobe restoring beams.
- `test_sicd_handler_*`: SARkit NITF reader/writer, chipping, and metric coordinate mapping.

---

## 6. License
MIT License
