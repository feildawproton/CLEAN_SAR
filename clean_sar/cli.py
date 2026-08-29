import os
import argparse
import numpy as np
import torch
from .sicd_handler import SICDHandler
from .psf import PSFGenerator
from .engine import run_hogbom_clean
from .utils import plot_clean_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complex SAR Hogbom CLEAN Deconvolution with Exact Spatially-Varying IPR for NITF SICDs."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input NITF SICD file.")
    parser.add_argument("-o", "--output", default=None, help="Path to output deconvolved NITF SICD file.")
    parser.add_argument("--plot", default=None, help="Path to output comparison PNG plot.")
    parser.add_argument(
        "--chip",
        default=None,
        help="Optional ROI bounding box as 'start_row,start_col,stop_row,stop_col' (e.g. '500,500,1012,1012').",
    )
    parser.add_argument(
        "--method",
        choices=["kspace", "analytic"],
        default="kspace",
        help="Exact PSF generation method: 'kspace' (Option A: 2D Aperture + IFFT) or 'analytic' (Option B: Spatial Sinc).",
    )
    parser.add_argument(
        "--beam",
        choices=["gaussian", "mainlobe"],
        default="gaussian",
        help="Restoring clean beam type: 'gaussian' (matched 3dB width) or 'mainlobe'.",
    )
    parser.add_argument("--gain", type=float, default=0.1, help="CLEAN loop damping factor / gain (default: 0.1).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Stopping threshold (fraction of peak if < 1.0, or absolute magnitude) (default: 0.02).",
    )
    parser.add_argument("--max-iters", type=int, default=2500, help="Maximum CLEAN iterations (default: 2500).")
    parser.add_argument("--psf-size", type=int, default=65, help="Kernel size for local PSF (default: 65).")
    parser.add_argument("--device", default=None, help="Computation device: 'cuda' or 'cpu' (default: auto).")
    parser.add_argument("--dyn-range", type=float, default=50.0, help="Dynamic range in dB for plotting (default: 50.0).")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.abspath(args.input)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(os.path.dirname(input_path), "output")
    os.makedirs(out_dir, exist_ok=True)

    out_nitf = args.output or os.path.join(out_dir, f"{base_name}_clean.nitf")
    out_plot = args.plot or os.path.join(out_dir, f"{base_name}_comparison.png")

    print("=" * 70)
    print("  CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution")
    print("  (Exact Spatially-Varying PSF Computed Per Peak Position)")
    print(f"  Input: {input_path}")
    print(f"  Method: {args.method.upper()} | Beam: {args.beam.upper()} | PSF Size: {args.psf_size}x{args.psf_size}")
    print(f"  Gain: {args.gain} | Threshold: {args.threshold} | Max Iters: {args.max_iters}")
    print("=" * 70)

    handler = SICDHandler(input_path)
    psf_gen = PSFGenerator(handler)

    chip_origin = None
    custom_xml = None

    if args.chip:
        coords = [int(c.strip()) for c in args.chip.split(",")]
        r_start, c_start, r_stop, c_stop = coords
        print(f"[*] Reading chip region: rows [{r_start}:{r_stop}], cols [{c_start}:{c_stop}]...")
        dirty_img, custom_xml = handler.read_chip(r_start, c_start, r_stop, c_stop)
        chip_origin = (r_start, c_start)
    else:
        print(f"[*] Reading full image ({handler.num_rows}x{handler.num_cols})...")
        dirty_img = handler.read_full_image()

    H, W = dirty_img.shape
    print(f"[*] Target image shape: {H}x{W} (Peak magnitude: {np.max(np.abs(dirty_img)):.4e})")

    res = run_hogbom_clean(
        dirty_image=dirty_img,
        psf_generator=psf_gen,
        method=args.method,
        beam_type=args.beam,
        psf_size=args.psf_size,
        gain=args.gain,
        threshold=args.threshold,
        max_iters=args.max_iters,
        chip_origin=chip_origin,
        device=args.device,
        verbose=True,
    )

    print(f"[+] CLEAN finished in {res.execution_time_sec:.2f}s (Iterations: {res.iterations})")
    print(f"    Initial peak: {np.max(np.abs(dirty_img)):.4e}")
    print(f"    Final residual peak: {np.max(np.abs(res.residual_image)):.4e}")
    print(f"    Extracted point scatterers: {np.count_nonzero(np.abs(res.components_map) > 0)}")

    # Write output NITF SICD
    print(f"[*] Saving deconvolved SICD NITF to: {out_nitf}")
    handler.write_nitf(out_nitf, res.clean_image, custom_xmltree=custom_xml)

    # Reference PSF for plot
    mid_r, mid_c = H // 2, W // 2
    if args.method == "kspace":
        ref_dirty = psf_gen.compute_psf_kspace(mid_r, mid_c, psf_size=args.psf_size, chip_origin=chip_origin)
    else:
        ref_dirty = psf_gen.compute_psf_analytic(mid_r, mid_c, psf_size=args.psf_size, chip_origin=chip_origin)
    ref_clean = psf_gen.compute_clean_beam(mid_r, mid_c, psf_size=args.psf_size, beam_type=args.beam, chip_origin=chip_origin)

    # Plot comparison
    print(f"[*] Generating comparison plot to: {out_plot}")
    plot_clean_comparison(
        dirty_image=dirty_img,
        clean_image=res.clean_image,
        residual_image=res.residual_image,
        components_map=res.components_map,
        output_path=out_plot,
        title=f"Exact Spatially-Varying CLEAN ({base_name})",
        dyn_range_db=args.dyn_range,
        psf_dirty=ref_dirty,
        psf_clean=ref_clean,
    )
    print("[+] All done successfully!")


if __name__ == "__main__":
    main()
