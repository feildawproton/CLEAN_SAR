# CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution with Spatially-Varying IPR

**CLEAN_SAR** is a GPU-accelerated and C-fallback Python framework for performing **Complex Hogbom CLEAN deconvolution** on Synthetic Aperture Radar (SAR) imagery stored in **NITF SICD** (Sensor Independent Complex Data) format.

It computes the **exact spatially-varying Impulse Response (IPR / Point Spread Function)** for every detected peak position across the scene, using **SARkit** for SICD metadata/image handling and **Native CUDA (with C fallback)** for deconvolution.

Designed for headless execution in cloud, HPC, and embedded environments with zero mandatory graphics or heavyweight ML framework dependencies.

---

## 1. Key Features & Architecture

- **PyTorch-Free**: Completely decoupled from PyTorch; relies solely on NumPy, SciPy, and SARkit for Python runtime.
- **Native CUDA Acceleration**: Compiled directly with `nvcc` ahead-of-time on installation or compiled just-in-time via NVIDIA Runtime Compiler (`libnvrtc`) and launched via the CUDA Driver API.
- **Identical C Fallback**: If CUDA is not available, CLEAN_SAR falls back to a high-performance C implementation mirroring the CUDA math and logic to float32 precision.
- **Automatic PSF Grid Sizing**: The PSF grid size is calculated automatically based on the integrated energy/power of the continuous analytic dirty PSF, bounded by the incoming image dimensions.
- **Simple, Clean API**: No need to instantiate or pass `PSFGenerator` objects to the algorithm; simply pass the complex dirty image and physics configuration.
- **Standard SICD Input**: Input images are expected to be proper SICD files (full scenes or caller-prepared chips), eliminating internal chipping complexity.

---

## 2. Backend Capabilities & Support Matrix

| Feature / Option | Native CUDA GPU (`cuda`) | Native C CPU (`c`) | Notes |
| :--- | :---: | :---: | :--- |
| **Gaussian Restoring Beam** | Yes | Yes | Matched to 3 dB half-power width ($\text{FWHM} = \text{ImpRespWid}$) |
| **Weighting Windows** | Uniform, Taylor, Hamming, Hann | Uniform, Taylor, Hamming, Hann | Analytic continuous evaluation in C and CUDA |
| **Compilation** | `nvcc` AOT or `libnvrtc` JIT | Compiled C `.so` with NumPy fallback | Automatic runtime resolution |
| **Parity** | Baseline | < 1e-5 difference vs CUDA | Peak selections match step-for-step |

---

## 3. Python API (`CLEANProcessor` & `run_hogbom_clean`)

### High-Level SICD Pipeline (`CLEANProcessor`)

The primary interface is **`CLEANProcessor`**, which handles SICD loading, scalar physics extraction, automatic PSF grid sizing, deconvolution, and compliant NITF writing:

```python
from clean_sar import CLEANProcessor

# 1. Instantiate processor with input & output NITF paths
processor = CLEANProcessor(
    input_path="/path/to/input_chip_SICD.nitf",
    output_path="output/clean_image.nitf",
    backend="auto",  # 'auto' (CUDA if available, else C), 'cuda', or 'c'
)

# 2. Run deconvolution (PSF grid size is automatically calculated from radar physics)
result = processor.run(
    gain=0.1,
    threshold=0.02,
    max_iters=2500,
    verbose=True,
)

print(f"Iterations: {result.iterations}, Peak reduction: {result.peak_reduction_db:.1f} dB")
```

### Direct Algorithm Execution (`run_hogbom_clean`)

For direct processing of NumPy complex arrays:

```python
from clean_sar import run_hogbom_clean, CleanPhysicsConfig

# config can be built directly or from a SICDHandler
result = run_hogbom_clean(
    dirty_image=complex_arr,
    config=config,
    backend="auto",
    gain=0.1,
    threshold=0.02,
    max_iters=2500,
)
```

---

## 4. General CLI Usage

For processing arbitrary SICD NITF files from the terminal:

```bash
clean-sar \
  -i /path/to/input_SICD.nitf \
  -o output/deconvolved_clean.nitf \
  --backend auto \
  --gain 0.1 \
  --threshold 0.02 \
  --max-iters 2500
```

#### CLI Parameters
| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path to input NITF SICD file | *Required* |
| `-o, --output` | Path to output deconvolved NITF file | `output/<name>_clean.nitf` |
| `--backend` | Compute engine: `auto`, `cuda`, or `c` | `auto` |
| `--gain` | Loop damping factor $\gamma$ | `0.1` |
| `--threshold` | Stopping threshold (fraction of initial peak if $< 1.0$) | `0.02` |
| `--max-iters` | Maximum CLEAN iterations | `2500` |
| `--guard-margin` | Margin in pixels excluded from peak selection | `0` |
| `--plot` | Optional path to save comparison PNG figure | None |
| `--dyn-range` | Dynamic range in dB for visualization | `50.0` |

---

## 5. Demo Runner (`demo_clean.py`)

Run an end-to-end deconvolution demo on sample SAR imagery (such as Umbra or DiffPFA datasets). By default, it exercises **both CUDA and C fallback backends**, verifies numerical parity to float32 precision, and renders comparison plots:

```bash
# Run both CUDA and C backends with automatic PSF power-based grid sizing
python demo_clean.py

# Run on a custom caller-provided SICD chip
python demo_clean.py --chip-input /path/to/chip.nitf --max-iters 1000

# Select a specific backend ('both', 'cuda', or 'c')
python demo_clean.py --backend cuda --max-iters 2500
```

Outputs generated in `output/demo/`:
- `<name>_clean_cuda.nitf` & `<name>_clean_c.nitf`: Compliant deconvolved NITF SICD files.
- `<name>_comparison.png`: 6-panel diagnostic plot (Dirty, Clean, Residual, Restored Model, and 1D Cuts).
- `<name>_cuda_vs_c_difference.png`: 4-panel backend parity plot verifying $< -140$ dB numerical difference and profile cut overlay.

---

## 6. Installation & Compiler Setup

### Quick Installation

```bash
pip install .
```

During installation:
1. `setup.py` checks for `nvcc` and compiles the CUDA kernels ahead-of-time to PTX. If `nvcc` is not found, installation completes smoothly and falls back to `libnvrtc` at runtime.
2. `setup.py` checks for a C compiler (`gcc`/`clang`) to compile `libclean_c.so`. If none is found, the C backend transparently uses the built-in vectorized engine.

### Installing NVCC and Compilers (Ubuntu / WSL2)

To enable ahead-of-time compilation with `nvcc` and native C compilation:

```bash
# 1. Install build-essential (gcc, g++, make)
sudo apt update
sudo apt install -y build-essential

# 2. Install NVIDIA CUDA Toolkit (provides nvcc matching Ubuntu 24.04 CUDA 12.x)
sudo apt install -y nvidia-cuda-toolkit
```

Verify that `nvcc` is installed:
```bash
nvcc --version
```

Then re-install CLEAN_SAR to precompile kernels:
```bash
pip install -e .
```

---

## 7. Running Tests

Run the complete test suite using pytest:

```bash
pytest -v tests/
```

To run against a custom directory of NITF SICD files:
```bash
CLEAN_SAR_TEST_DATA=/path/to/sicd/dir pytest -v tests/
```

---

## 8. License

MIT License
