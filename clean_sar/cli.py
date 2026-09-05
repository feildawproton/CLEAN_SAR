import os
import argparse
from .processor import CLEANProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complex SAR Hogbom CLEAN Deconvolution with Exact Spatially-Varying IPR for NITF SICDs."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input NITF SICD file.")
    parser.add_argument("-o", "--output", default=None, help="Path to output deconvolved NITF SICD file.")
    parser.add_argument("--plot", default=None, help="Optional path to output comparison PNG plot.")
    parser.add_argument(
        "--chip",
        default=None,
        help="Optional ROI bounding box as 'start_row,start_col,stop_row,stop_col' (e.g. '500,500,1012,1012').",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pytorch", "cuda"],
        default="auto",
        help="Compute backend: 'auto' (default), 'pytorch', or 'cuda'.",
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
    parser.add_argument("--guard-margin", type=int, default=0, help="Border margin in pixels excluded from peak selection (default: 0).")
    parser.add_argument("--device", default=None, help="Computation device: 'cuda' or 'cpu' (default: auto).")
    parser.add_argument("--dyn-range", type=float, default=50.0, help="Dynamic range in dB for plotting (default: 50.0).")
    parser.add_argument("--ref-val", type=float, default=None, help="Reference magnitude for 0 dB scale normalization in plot.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = os.path.abspath(args.input)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(os.path.dirname(input_path), "output")
    os.makedirs(out_dir, exist_ok=True)

    out_nitf = args.output or os.path.join(out_dir, f"{base_name}_clean.nitf")

    chip_bounds = None
    if args.chip:
        coords = [int(c.strip()) for c in args.chip.split(",")]
        chip_bounds = (coords[0], coords[1], coords[2], coords[3])

    print("=" * 70)
    print("  CLEAN_SAR: Complex SAR Hogbom CLEAN Deconvolution")
    print(f"  Input:   {input_path}")
    print(f"  Output:  {out_nitf}")
    if chip_bounds:
        print(f"  Chip:    rows [{chip_bounds[0]}:{chip_bounds[2]}], cols [{chip_bounds[1]}:{chip_bounds[3]}]")
    else:
        print("  Target:  Full Scene")
    print(f"  Backend: {args.backend.upper()} | Device: {args.device or 'auto'}")
    print(f"  Beam:    {args.beam.upper()} | PSF Size: {args.psf_size}x{args.psf_size}")
    print(f"  Gain:    {args.gain} | Threshold: {args.threshold} | Max Iters: {args.max_iters}")
    print("=" * 70)

    processor = CLEANProcessor(
        input_path=input_path,
        output_path=out_nitf,
        chip_bounds=chip_bounds,
        backend=args.backend,
        device=args.device,
    )

    result = processor.run(
        gain=args.gain,
        threshold=args.threshold,
        max_iters=args.max_iters,
        beam_type=args.beam,
        psf_size=args.psf_size,
        guard_margin=args.guard_margin,
        verbose=True,
    )

    if args.plot:
        try:
            from tools.compare_sicd import plot_comparison
            # Read dirty chip / image for comparison plot
            if chip_bounds:
                dirty_img, _ = processor.handler.read_chip(*chip_bounds)
            else:
                dirty_img = processor.handler.read_full_image()

            plot_comparison(
                dirty_image=dirty_img,
                clean_image=result.clean_image,
                residual_image=result.residual_image,
                restored_model=result.restored_model,
                output_png=args.plot,
                dyn_range_db=args.dyn_range,
                title_suffix=os.path.basename(input_path),
                ref_val=args.ref_val,
            )
        except ImportError:
            print("[!] Matplotlib not available; skipping plot generation.")


if __name__ == "__main__":
    main()
