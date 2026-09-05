#!/usr/bin/env python3
"""
demo_clean.py: Out-of-the-box demo runner for CLEAN_SAR.

Runs Complex Hogbom CLEAN deconvolution with exact spatially-varying PSF synthesis
on the 2023-11-14 UMBRA-04 dataset (both raw Umbra and DiffPFA outputs).
"""

import os
import argparse
from clean_sar.processor import CLEANProcessor
from tools.compare_sicd import plot_comparison

DEFAULT_UMBRA_PATH = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
DEFAULT_DIFFPFA_PATH = "/home/feildaw/diffpfa/workspace/output/2023-11-14-03-38-20_UMBRA-04_SICDU_X_X.nitf"

UMBRA_CHIP = (1932, 3931, 2188, 4187)
DIFFPFA_CHIP = (4613, 3737, 4869, 3993)


def run_single_demo(
    name: str,
    input_path: str,
    chip_bounds,
    out_dir: str,
    backend: str = "auto",
    beam_type: str = "gaussian",
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    dyn_range: float = 50.0,
):
    print("\n" + "=" * 75)
    print(f"  [DEMO] Processing {name}")
    print(f"  File:    {input_path}")
    print(f"  Chip:    rows [{chip_bounds[0]}:{chip_bounds[2]}], cols [{chip_bounds[1]}:{chip_bounds[3]}]")
    print(f"  Backend: {backend.upper()}")
    print(f"  Beam:    {beam_type.upper()} | PSF: {psf_size}x{psf_size}")
    print(f"  Gain:    {gain} | Thresh: {threshold} | Max Iters: {max_iters}")
    print("=" * 75)

    if not os.path.exists(input_path):
        print(f"[!] Target file does not exist: {input_path}")
        return

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_nitf = os.path.join(out_dir, f"{base_name}_clean.nitf")
    out_png = os.path.join(out_dir, f"{base_name}_comparison.png")

    # 1. Instantiate processor
    processor = CLEANProcessor(
        input_path=input_path,
        output_path=out_nitf,
        chip_bounds=chip_bounds,
        backend=backend,
    )

    # 2. Run deconvolution
    result = processor.run(
        gain=gain,
        threshold=threshold,
        max_iters=max_iters,
        beam_type=beam_type,
        psf_size=psf_size,
        verbose=True,
    )

    print(f"[*] Deconvolution completed in {result.execution_time_sec:.3f}s ({result.iterations} iterations)")
    print(f"    Sidelobe suppression: {result.suppression_db:.2f} dB")
    print(f"    Saved clean SICD: {out_nitf}")

    # 3. Render side-by-side diagnostic plot
    print(f"[*] Rendering diagnostic comparison plot: {out_png}")
    plot_comparison(
        input_path,
        out_nitf,
        chip_bounds=chip_bounds,
        save_path=out_png,
        dynamic_range_db=dyn_range,
    )


def main():
    parser = argparse.ArgumentParser(description="CLEAN_SAR Quick Demo Runner.")
    parser.add_argument("--out_dir", type=str, default="output/demo", help="Output directory")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cuda", "pytorch"], help="Compute backend")
    parser.add_argument("--beam_type", type=str, default="gaussian", choices=["gaussian", "mainlobe"])
    parser.add_argument("--psf_size", type=int, default=65)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--max_iters", type=int, default=2500)
    parser.add_argument("--dyn_range", type=float, default=50.0)
    parser.add_argument("--skip_umbra", action="store_true")
    parser.add_argument("--skip_diffpfa", action="store_true")

    args = parser.parse_args()

    if not args.skip_umbra:
        run_single_demo(
            name="Raw Umbra 2023-11-14 (Strong Scatterer Point Target)",
            input_path=DEFAULT_UMBRA_PATH,
            chip_bounds=UMBRA_CHIP,
            out_dir=args.out_dir,
            backend=args.backend,
            beam_type=args.beam_type,
            psf_size=args.psf_size,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            dyn_range=args.dyn_range,
        )

    if not args.skip_diffpfa:
        run_single_demo(
            name="DiffPFA 2023-11-14 (Refocused Point Target)",
            input_path=DEFAULT_DIFFPFA_PATH,
            chip_bounds=DIFFPFA_CHIP,
            out_dir=args.out_dir,
            backend=args.backend,
            beam_type=args.beam_type,
            psf_size=args.psf_size,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            dyn_range=args.dyn_range,
        )


if __name__ == "__main__":
    main()
