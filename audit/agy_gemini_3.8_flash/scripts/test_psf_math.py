import numpy as np
import math
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator

# Test config with standard values
config = CleanPhysicsConfig(
    row_ss=0.5,
    col_ss=0.5,
    row_bw=1.5,
    col_bw=1.5,
    row_wid=0.886 / 1.5,
    col_wid=0.886 / 1.5,
    scp_slant_range=600000.0,
    scp_row=1000.0,
    scp_col=1000.0,
    row_wgt="UNIFORM",
    col_wgt="UNIFORM",
)

psf_gen = PSFGenerator(config)

windows = ["UNIFORM", "TAYLOR", "HAMMING", "HANN"]

print("=== PSF PEAK AND SIDELOBE ANALYSIS (Python PSFGenerator) ===")
for w in windows:
    psf = psf_gen.compute_psf(1000, 1000, psf_size=65, window_row=w, window_col=w)
    center = 32
    peak = psf[center, center]
    # Sidelobes along row
    cut = np.abs(psf[:, center])
    # find sidelobes
    # local maxima excluding mainlobe (e.g. outside +- 2 pixels)
    sidelobes = []
    for i in range(1, len(cut) - 1):
        if abs(i - center) > 2 and cut[i] > cut[i-1] and cut[i] > cut[i+1]:
            sidelobes.append(cut[i])
    max_sll_db = 20.0 * np.log10(max(sidelobes) / np.abs(peak)) if sidelobes else -999.0
    print(f"Window: {w:8s} | Peak: {peak.real:.4f} + {peak.imag:.4f}j | Max SLL: {max_sll_db:.2f} dB")

print("\n=== CUDA eval_1d_window_sinc PEAK VALUE AT ZERO ===")
# Emulate clean_hogbom.cu eval_1d_window_sinc at pos = 0
def dev_sinc(x):
    return 1.0 if abs(x) < 1e-7 else math.sin(math.pi * x) / (math.pi * x)

def eval_1d_cuda(pos, bw, wgt_type):
    x = bw * pos
    if wgt_type == 0: # UNIFORM
        return dev_sinc(x)
    elif wgt_type == 1: # TAYLOR
        f1 = 0.29265601
        f2 = -0.01578375
        f3 = 0.00218104
        return (dev_sinc(x) +
                f1 * (dev_sinc(x - 1.0) + dev_sinc(x + 1.0)) +
                f2 * (dev_sinc(x - 2.0) + dev_sinc(x + 2.0)) +
                f3 * (dev_sinc(x - 3.0) + dev_sinc(x + 3.0)))
    elif wgt_type == 2: # HAMMING
        return 0.54 * dev_sinc(x) + 0.23 * (dev_sinc(x - 1.0) + dev_sinc(x + 1.0))
    elif wgt_type == 3: # HANN
        return 0.50 * dev_sinc(x) + 0.25 * (dev_sinc(x - 1.0) + dev_sinc(x + 1.0))

for code, w in enumerate(windows):
    val_1d = eval_1d_cuda(0.0, 1.5, code)
    h_dirty_cuda = val_1d * val_1d
    print(f"CUDA code {code} ({w:8s}): 1D val at 0 = {val_1d:.4f} | 2D h_dirty at (0,0) = {h_dirty_cuda:.4f}")
