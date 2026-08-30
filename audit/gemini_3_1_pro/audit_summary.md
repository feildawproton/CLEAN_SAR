# CLEAN_SAR Audit Summary

## 1. Problem Statement
Traditional CLEAN algorithms from radio astronomy were designed for real-valued, positive-intensity images under the assumption of a shift-invariant Point Spread Function (PSF). Synthetic Aperture Radar (SAR) imagery, however, breaks these assumptions:
1. **Complex Signals:** SAR data represents complex signals ($I + jQ$), encapsulating vital phase information necessary for interferometry and coherent scattering analysis.
2. **Spatially-Varying PSF:** Due to image formation algorithms like the Polar Format Algorithm (PFA), the effective spatial frequency support and resolution vary continuously depending on the distance from the Scene Center Point (SCP). This means the PSF (or Impulse Response, IPR) is **spatially-varying** and changes shape across the image.
3. **Format Support:** Standardized SAR analysis requires handling NGA's **NITF SICD** format rather than standard astronomy FITS files.

## 2. Solution & Approach
The **CLEAN_SAR** project solves these challenges by implementing a **Complex Hogbom CLEAN deconvolution engine** tailored for SAR:
- **Complex Domain Operation:** The Hogbom CLEAN loop directly operates on and reconstructs complex tensors using PyTorch for CUDA-accelerated performance.
- **Exact Per-Peak Spatially-Varying PSF:** Instead of assuming a static PSF or interpolating from a grid of PSFs, the engine dynamically generates the **exact** PSF for the global coordinate of every identified peak at each iteration. 
- **Two Generation Methods:** The `PSFGenerator` supports computing the local dirty PSF via an exact 2D K-space inverse FFT (Option A), or an analytic spatial-domain formulation incorporating geometric shear (Option B).
- **Coordinate Caching:** It mitigates the computational cost of exact PSF generation by aggressively caching PSFs by their exact integer pixel coordinates. This ensures that for bright scatterers requiring multiple CLEAN iterations, the expensive PSF calculation is only performed once.
- **Standards Integration:** The `SICDHandler` class elegantly wraps `SARkit` to natively read, transform (local chip to global metric), and write NITF SICD files without losing crucial metadata.

## 3. Initial Audit Findings
- The architecture correctly addresses the physics of SAR deconvolution.
- The use of `torch.complex64` native processing ensures high performance on GPU.
- The caching layer in `PSFGenerator` is critical for $\mathcal{O}(1)$ lookups on subsequent hits.
- The code structure (`engine.py`, `psf.py`, `sicd_handler.py`) separates concerns (I/O, Physics, Execution) cleanly.

Overall, the project is a sophisticated, correct application of the CLEAN algorithm to the complex, spatially-varying nature of SAR imagery.
