import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")

from audit.gemini_3_7_flash.fixed_components import EnhancedSICDHandler, EnhancedPSFGenerator
from clean_sar.engine import run_hogbom_clean

def test_chip_scaling():
    print("=" * 75)
    print("DEMO: Chip Size vs CLEAN Iteration Capacity & Scatterer Extraction")
    print("=" * 75)
    
    data_path = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    handler = EnhancedSICDHandler(data_path)
    psf_gen = EnhancedPSFGenerator(handler)
    
    center_r, center_c = 2800, 6450
    chip_sizes = [64, 128, 256, 512]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Center: ({center_r}, {center_c})")
    
    results = []
    
    for size in chip_sizes:
        half = size // 2
        r_start = center_r - half
        c_start = center_c - half
        r_stop = r_start + size
        c_stop = c_start + size
        
        chip, chip_xml = handler.read_chip(r_start, c_start, r_stop, c_stop)
        
        init_peak = np.max(np.abs(chip))
        
        t0 = time.time()
        res = run_hogbom_clean(
            dirty_image=chip,
            psf_generator=psf_gen,
            method="kspace",
            beam_type="gaussian",
            psf_size=65,
            gain=0.1,
            threshold=0.02, # Clean down to 2% (-34 dB) of peak
            max_iters=5000,
            chip_origin=(r_start, c_start),
            device=device,
            verbose=False,
        )
        t_elapsed = time.time() - t0
        
        final_resid = np.max(np.abs(res.residual_image))
        num_scatterers = np.count_nonzero(np.abs(res.components_map) > 0)
        reduction_db = 20.0 * np.log10(final_resid / init_peak)
        
        entry = {
            "size": size,
            "pixels": size * size,
            "iters": res.iterations,
            "scatterers": num_scatterers,
            "time_sec": t_elapsed,
            "iters_per_sec": res.iterations / t_elapsed if t_elapsed > 0 else 0,
            "reduction_db": reduction_db,
            "init_peak": init_peak,
            "final_resid": final_resid,
        }
        results.append(entry)
        
        print(f"Chip {size:4d}x{size:4d} ({size*size:7d} px): "
              f"Iterations={res.iterations:4d} | "
              f"Scatterers={num_scatterers:4d} | "
              f"Reduction={reduction_db:6.2f} dB | "
              f"Time={t_elapsed:5.2f}s ({res.iterations/t_elapsed:5.1f} it/s)")
    
    # Plot results
    out_plot = "/home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output/chip_scaling_analysis.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    sizes = [r["size"] for r in results]
    iters = [r["iters"] for r in results]
    scatts = [r["scatterers"] for r in results]
    times = [r["time_sec"] for r in results]
    
    # Plot 1: Iterations vs Chip Size
    axes[0].plot(sizes, iters, "o-", color="#1f77b4", linewidth=2, markersize=8)
    axes[0].set_title("CLEAN Iterations vs Chip Dimension", fontweight="bold")
    axes[0].set_xlabel("Chip Dimension (pixels, N x N)")
    axes[0].set_ylabel("Iterations to Convergence (threshold=0.02)")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Unique Scatterers vs Chip Size
    axes[1].plot(sizes, scatts, "s-", color="#2ca02c", linewidth=2, markersize=8)
    axes[1].set_title("Point Scatterers Extracted vs Chip Dimension", fontweight="bold")
    axes[1].set_xlabel("Chip Dimension (pixels, N x N)")
    axes[1].set_ylabel("Extracted Point Scatterers")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Execution Time vs Chip Size
    axes[2].plot(sizes, times, "^-", color="#d62728", linewidth=2, markersize=8)
    axes[2].set_title("Total Runtime (seconds)", fontweight="bold")
    axes[2].set_xlabel("Chip Dimension (pixels, N x N)")
    axes[2].set_ylabel("Runtime (s)")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_plot, dpi=180)
    plt.close()
    print(f"\n[+] Saved scaling analysis plot: {out_plot}")
    
    return results

if __name__ == "__main__":
    test_chip_scaling()
