import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/home/feildaw/CLEAN_SAR")

from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator
from clean_sar.engine import run_hogbom_clean
from clean_sar.utils import plot_clean_comparison

def run_benchmark():
    input_file = "/home/feildaw/data/2023-07-30-17-19-39_UMBRA-05_SICD.nitf"
    out_dir = "/home/feildaw/CLEAN_SAR/audit/gemini_3_7_flash/output"
    os.makedirs(out_dir, exist_ok=True)
    
    out_nitf = os.path.join(out_dir, "benchmark_clean_chip.nitf")
    out_plot = os.path.join(out_dir, "benchmark_clean_comparison.png")
    
    print(f"Loading input SICD: {input_file}")
    handler = SICDHandler(input_file)
    psf_gen = PSFGenerator(handler)
    
    # 256x256 chip around bright feature
    start_r, start_c = 2700, 6350
    stop_r, stop_c = 2956, 6606
    
    chip, chip_xml = handler.read_chip(start_r, start_c, stop_r, stop_c)
    print(f"Extracted chip shape: {chip.shape} (dtype: {chip.dtype})")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing CLEAN on device: {device}")
    
    t0 = time.time()
    result = run_hogbom_clean(
        dirty_image=chip,
        psf_generator=psf_gen,
        method="kspace",
        beam_type="gaussian",
        psf_size=65,
        gain=0.1,
        threshold=0.02,
        max_iters=1500,
        chip_origin=(start_r, start_c),
        device=device,
        verbose=True,
    )
    t1 = time.time()
    
    print(f"\nExecution summary:")
    print(f"  Time: {t1 - t0:.2f}s ({result.iterations} iters, {result.iterations/(t1 - t0):.1f} iters/s)")
    print(f"  Dirty peak: {np.max(np.abs(chip)):.4e}")
    print(f"  Final residual peak: {np.max(np.abs(result.residual_image)):.4e}")
    print(f"  Points extracted: {np.count_nonzero(np.abs(result.components_map) > 0)}")
    print(f"  Cache entries: {len(psf_gen._cache_dirty)}")
    
    # Write NITF
    print(f"Writing clean NITF to: {out_nitf}")
    handler.write_nitf(out_nitf, result.clean_image, custom_xmltree=chip_xml)
    assert os.path.exists(out_nitf) and os.path.getsize(out_nitf) > 0
    
    # Generate Plot
    mid_r, mid_c = chip.shape[0] // 2, chip.shape[1] // 2
    psf_dirty = psf_gen.compute_psf_kspace(mid_r, mid_c, psf_size=65, chip_origin=(start_r, start_c))
    psf_clean = psf_gen.compute_clean_beam(mid_r, mid_c, psf_size=65, beam_type="gaussian", chip_origin=(start_r, start_c))
    
    print(f"Writing comparison plot to: {out_plot}")
    plot_clean_comparison(
        dirty_image=chip,
        clean_image=result.clean_image,
        residual_image=result.residual_image,
        components_map=result.components_map,
        output_path=out_plot,
        title="CLEAN_SAR Audit Benchmark: Umbra-05 Scene",
        dyn_range_db=50.0,
        psf_dirty=psf_dirty,
        psf_clean=psf_clean,
    )
    assert os.path.exists(out_plot) and os.path.getsize(out_plot) > 0
    print("[+] Benchmark run completed successfully!")

if __name__ == "__main__":
    run_benchmark()
