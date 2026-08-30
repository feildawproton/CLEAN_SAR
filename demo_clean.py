#!/usr/bin/env python3
"""
demo_clean.py: Demo and runner for Complex SAR Hogbom CLEAN Deconvolution
with Exact Spatially-Varying IPR for NITF SICDs.

Default Dataset: 2023-11-14-03-38-20_UMBRA-04 (both Raw and DiffPFA outputs)
"""

import os
import argparse
import numpy as np
import torch
from clean_sar import SICDHandler, PSFGenerator, run_hogbom_clean, plot_clean_comparison

# Default paths for 2023-11-14-03-38-20_UMBRA-04
DEFAULT_UMBRA_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
DEFAULT_DIFFPFA_PATH = "/home/feildaw/diffpfa/workspace/output/2023-11-14-03-38-20_UMBRA-04_SICDU_X_X.nitf"

# Default chip regions around prominent scatterers near SCP (256x256)
DEFAULT_UMBRA_CHIP = (1932, 3931, 2188, 4187)      # centered around peak at (2060, 4059)
DEFAULT_DIFFPFA_CHIP = (4613, 3737, 4869, 3993)    # centered around peak at (4741, 3865)


def run_single_dataset(
    input_path: str,
    output_dir: str,
    label: str,
    chip_coords: tuple,
    method: str = "kspace",
    beam_type: str = "gaussian",
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    psf_size: int = 65,
    dyn_range: float = 50.0,
    device: str = None,
):
    if not os.path.exists(input_path):
        print(f"[!] Warning: File not found: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_nitf = os.path.join(output_dir, f"{base_name}_clean.nitf")
    out_plot = os.path.join(output_dir, f"{base_name}_comparison.png")

    print("\n" + "=" * 75)
    print(f"  [DEMO] Processing {label}")
    print(f"  File:   {input_path}")
    print(f"  Chip:   rows [{chip_coords[0]}:{chip_coords[2]}], cols [{chip_coords[1]}:{chip_coords[3]}]")
    print(f"  Method: {method.upper()} | Beam: {beam_type.upper()} | PSF: {psf_size}x{psf_size}")
    print(f"  Gain:   {gain} | Thresh: {threshold} | Max Iters: {max_iters}")
    print("=" * 75)

    handler = SICDHandler(input_path)
    psf_gen = PSFGenerator(handler)

    r_start, c_start, r_stop, c_stop = chip_coords
    dirty_img, chip_xml = handler.read_chip(r_start, c_start, r_stop, c_stop)
    chip_origin = (r_start, c_start)

    H, W = dirty_img.shape
    init_peak = float(np.max(np.abs(dirty_img)))
    print(f"[*] Loaded chip shape: {H}x{W} (Initial peak magnitude: {init_peak:.4e})")

    res = run_hogbom_clean(
        dirty_image=dirty_img,
        psf_generator=psf_gen,
        method=method,
        beam_type=beam_type,
        psf_size=psf_size,
        gain=gain,
        threshold=threshold,
        max_iters=max_iters,
        chip_origin=chip_origin,
        device=device,
        verbose=True,
    )

    final_resid = float(np.max(np.abs(res.residual_image)))
    suppression_db = 20.0 * np.log10(init_peak / final_resid) if final_resid > 0 else 999.0

    print(f"[+] Finished in {res.execution_time_sec:.2f}s ({res.iterations} iterations)")
    print(f"    Initial peak:        {init_peak:.4e}")
    print(f"    Final residual peak: {final_resid:.4e} ({suppression_db:.1f} dB suppression)")
    print(f"    Point scatterers:    {np.count_nonzero(np.abs(res.components_map) > 0)}")

    # Write output NITF SICD
    print(f"[*] Saving deconvolved SICD NITF to: {out_nitf}")
    handler.write_nitf(out_nitf, res.clean_image, custom_xmltree=chip_xml)

    # Reference PSF for plot
    mid_r, mid_c = H // 2, W // 2
    if method == "kspace":
        ref_dirty = psf_gen.compute_psf_kspace(mid_r, mid_c, psf_size=psf_size, chip_origin=chip_origin)
    else:
        ref_dirty = psf_gen.compute_psf_analytic(mid_r, mid_c, psf_size=psf_size, chip_origin=chip_origin)
    ref_clean = psf_gen.compute_clean_beam(mid_r, mid_c, psf_size=psf_size, beam_type=beam_type, chip_origin=chip_origin)

    # Generate comparison plot
    print(f"[*] Saving comparison plot to: {out_plot}")
    plot_clean_comparison(
        dirty_image=dirty_img,
        clean_image=res.clean_image,
        residual_image=res.residual_image,
        components_map=res.components_map,
        output_path=out_plot,
        title=f"{label} - Exact Spatially-Varying CLEAN",
        dyn_range_db=dyn_range,
        psf_dirty=ref_dirty,
        psf_clean=ref_clean,
    )
    print(f"[+] {label} completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Demo CLEAN deconvolution on 2023-11-14-03-38-20 UMBRA and DiffPFA datasets."
    )
    parser.add_argument(
        "--target",
        choices=["both", "umbra", "diffpfa"],
        default="both",
        help="Which dataset to process: 'both', 'umbra' (raw), or 'diffpfa' (default: both).",
    )
    parser.add_argument("--umbra-path", default=DEFAULT_UMBRA_PATH, help="Path to Umbra raw SICD NITF.")
    parser.add_argument("--diffpfa-path", default=DEFAULT_DIFFPFA_PATH, help="Path to DiffPFA SICD NITF.")
    parser.add_argument("--method", choices=["kspace", "analytic"], default="kspace", help="PSF method (default: kspace).")
    parser.add_argument("--beam", choices=["gaussian", "mainlobe"], default="gaussian", help="Restoring beam (default: gaussian).")
    parser.add_argument("--gain", type=float, default=0.1, help="Loop gain (default: 0.1).")
    parser.add_argument("--threshold", type=float, default=0.02, help="Stopping threshold fraction (default: 0.02).")
    parser.add_argument("--max-iters", type=int, default=2500, help="Maximum iterations (default: 2500).")
    parser.add_argument("--psf-size", type=int, default=65, help="PSF kernel size (default: 65).")
    parser.add_argument("--device", default=None, help="Device: 'cuda' or 'cpu' (default: auto).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))

    if args.target in ["both", "umbra"]:
        out_dir_umbra = os.path.join(project_root, "output", "umbra_20231114")
        run_single_dataset(
            input_path=args.umbra_path,
            output_dir=out_dir_umbra,
            label="UMBRA Raw SICD (2023-11-14)",
            chip_coords=DEFAULT_UMBRA_CHIP,
            method=args.method,
            beam_type=args.beam,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            psf_size=args.psf_size,
            device=args.device,
        )

    if args.target in ["both", "diffpfa"]:
        out_dir_diff = os.path.join(project_root, "output", "diffpfa_20231114")
        run_single_dataset(
            input_path=args.diffpfa_path,
            output_dir=out_dir_diff,
            label="DiffPFA Reconstructed SICD (2023-11-14)",
            chip_coords=DEFAULT_DIFFPFA_CHIP,
            method=args.method,
            beam_type=args.beam,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            psf_size=args.psf_size,
            device=args.device,
        )

    print("\n" + "=" * 75)
    print("  [DEMO COMPLETE] All requested datasets deconvolved and saved to output/")
    print("=" * 75)


if __name__ == "__main__":
    main()
