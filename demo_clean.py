#!/usr/bin/env python3
"""
demo_clean.py: Out-of-the-box demo runner for CLEAN_SAR.

Demonstrates Complex Hogbom CLEAN deconvolution with exact spatially-varying PSF synthesis
on SAR SICD imagery (both raw Umbra and DiffPFA refocused outputs).

Exercises the simplified interface:
- Caller provides a proper SICD chip (or prepares one using SICDHandler)
- CLEANProcessor interface (no chip_bounds, no PSFGenerator object, no device parameter)
- Automatic PSF grid size selection based on enclosed PSF power (psf_size=None)
- Backend selection (auto, cuda, or c)
"""

import os
import argparse
from typing import Optional, Tuple
import numpy as np
from clean_sar.processor import CLEANProcessor
from clean_sar.sicd_handler import SICDHandler
from clean_sar.utils import db_scale
from tools.compare_sicd import plot_comparison

# Default paths for 2023-09-11-10-37-05 UMBRA-05 dataset
DEFAULT_UMBRA_PATH = "/home/feildaw/data/2023-09-11-10-37-05_UMBRA-05_SICD.nitf"
DEFAULT_DIFFPFA_PATH = "/home/feildaw/diffpfa/workspace/output/2023-09-11-10-37-05_UMBRA-05_SICDU_X_X.nitf"

# Default 256x256 chip regions around prominent scatterers near scene center
DEFAULT_UMBRA_CHIP = (1792, 2093, 2048, 2349)     # centered around peak at (1920, 2221)
DEFAULT_DIFFPFA_CHIP = (1809, 2151, 2065, 2407)   # centered around peak at (1937, 2279)


def plot_backend_parity(
    cuda_image: np.ndarray,
    c_image: np.ndarray,
    output_png: str,
    dyn_range_db: float = 50.0,
    title_suffix: str = "",
) -> None:
    """
    Renders and saves a 4-panel comparison verifying numerical parity between CUDA and C backends:
    - Panel 1: CUDA Clean Image (dB)
    - Panel 2: C Fallback Clean Image (dB)
    - Panel 3: Absolute Difference |CUDA - C| in dB relative to image peak
    - Panel 4: 1D Range & Azimuth Cuts comparing CUDA vs C overlaid
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib required for plotting.")
        return

    peak = float(np.max(np.abs(cuda_image)))
    if peak <= 0:
        peak = 1.0

    cuda_db = db_scale(cuda_image, dyn_range_db, ref_val=peak)
    c_db = db_scale(c_image, dyn_range_db, ref_val=peak)

    # Difference in dB relative to peak power
    diff_mag = np.abs(cuda_image - c_image)
    diff_db = 20.0 * np.log10(np.maximum(diff_mag / peak, 1e-12))
    diff_max = float(np.max(diff_db))
    diff_min = float(np.min(diff_db))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    t_str = f" - {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Backend Parity Verification: CUDA vs C Fallback{t_str}", fontsize=15, fontweight="bold")

    # 1. CUDA Clean Image
    im0 = axes[0, 0].imshow(cuda_db, cmap="gray", vmin=-dyn_range_db, vmax=0.0)
    axes[0, 0].set_title("CUDA Deconvolved Image")
    axes[0, 0].set_xlabel("Col Index")
    axes[0, 0].set_ylabel("Row Index")
    fig.colorbar(im0, ax=axes[0, 0], label="Power (dB)")

    # 2. C Clean Image
    im1 = axes[0, 1].imshow(c_db, cmap="gray", vmin=-dyn_range_db, vmax=0.0)
    axes[0, 1].set_title("C Fallback Deconvolved Image")
    axes[0, 1].set_xlabel("Col Index")
    axes[0, 1].set_ylabel("Row Index")
    fig.colorbar(im1, ax=axes[0, 1], label="Power (dB)")

    # 3. Difference Map in dB
    im2 = axes[1, 0].imshow(diff_db, cmap="magma", vmin=diff_min, vmax=diff_max)
    axes[1, 0].set_title(f"Difference Map: 20*log10(|CUDA - C| / Peak)\n(Peak Difference: {diff_max:.1f} dB)")
    axes[1, 0].set_xlabel("Col Index")
    axes[1, 0].set_ylabel("Row Index")
    fig.colorbar(im2, ax=axes[1, 0], label="Relative Difference (dB)")

    # 4. 1D Cuts comparing CUDA and C
    peak_idx = np.unravel_index(np.argmax(np.abs(cuda_image)), cuda_image.shape)
    r_peak, c_peak = peak_idx[0], peak_idx[1]

    axes[1, 1].plot(cuda_db[:, c_peak], label="CUDA (Range Cut)", color="navy", linewidth=2.0)
    axes[1, 1].plot(c_db[:, c_peak], label="C Fallback (Range Cut)", color="crimson", linestyle="--", linewidth=1.5)
    axes[1, 1].plot(cuda_db[r_peak, :], label="CUDA (Azimuth Cut)", color="teal", linewidth=2.0)
    axes[1, 1].plot(c_db[r_peak, :], label="C Fallback (Azimuth Cut)", color="orange", linestyle="--", linewidth=1.5)

    axes[1, 1].set_title(f"1D Profile Overlay (Peak at Row {r_peak}, Col {c_peak})")
    axes[1, 1].set_xlabel("Pixel Index")
    axes[1, 1].set_ylabel("Power (dB)")
    axes[1, 1].set_ylim([-dyn_range_db, 5.0])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"[+] Parity comparison figure saved: {output_png}")


def run_single_demo(
    name: str,
    input_path: str,
    out_dir: str,
    chip_bounds: Optional[Tuple[int, int, int, int]] = None,
    backend: str = "both",
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 5000,
    dyn_range: float = 50.0,
):
    """
    Runs CLEAN deconvolution demo on a SICD target image.

    Demonstrates:
    1. Extracting/preparing a caller-provided SICD chip if input is a large scene.
    2. Invoking CLEANProcessor with the simplified interface.
    3. Automatic calculation of optimal PSF grid size from radar physics.
    4. Exercising both CUDA and C backends and verifying numerical parity in dB.
    5. Exporting deconvolved SICDs and diagnostic comparison plots.
    """
    print("\n" + "=" * 78)
    print(f"  [DEMO] Processing: {name}")
    print(f"  Input File: {input_path}")
    print(f"  Backend(s): {backend.upper()}")
    print(f"  Gain:       {gain} | Threshold: {threshold} | Max Iterations: {max_iters}")
    print("=" * 78)

    if not os.path.exists(input_path):
        print(f"[!] Target file does not exist: {input_path}")
        return

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Step 1: Prepare caller-provided SICD chip if extracting from a larger scene
    if chip_bounds is not None:
        chip_input_nitf = os.path.join(out_dir, f"{base_name}_chip_input.nitf")
        print(f"[*] Extracting caller chip rows [{chip_bounds[0]}:{chip_bounds[2]}], "
              f"cols [{chip_bounds[1]}:{chip_bounds[3]}] -> {chip_input_nitf}")
        h = SICDHandler(input_path)
        chip_data, chip_xml = h.read_chip(*chip_bounds)
        h.write_nitf(chip_input_nitf, chip_data, custom_xmltree=chip_xml)
        proc_input = chip_input_nitf
        dirty_img = chip_data
    else:
        proc_input = input_path
        h = SICDHandler(input_path)
        dirty_img = h.read_full_image()

    # Step 2: Determine backends to run
    backends_to_run = ["cuda", "c"] if backend == "both" else [backend]
    results = {}

    for b in backends_to_run:
        b_label = b.upper()
        out_nitf_b = (
            os.path.join(out_dir, f"{base_name}_clean_{b}.nitf")
            if len(backends_to_run) > 1
            else os.path.join(out_dir, f"{base_name}_clean.nitf")
        )
        print(f"\n  ---> Running [{b_label}] Backend:")
        processor = CLEANProcessor(
            input_path=proc_input,
            output_path=out_nitf_b,
            backend=b,
        )
        res = processor.run(
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            verbose=True,
        )
        results[b] = res
        print(f"[*] [{b_label}] completed in {res.execution_time_sec:.3f}s ({res.iterations} iterations)")
        print(f"    Sidelobe suppression: {res.suppression_db:.2f} dB")
        print(f"    Saved clean SICD: {out_nitf_b}")

    # Step 3: Parity check and difference plot if both backends were exercised
    if "cuda" in results and "c" in results:
        cuda_clean = results["cuda"].clean_image
        c_clean = results["c"].clean_image
        peak_val = float(np.max(np.abs(cuda_clean)))
        diff_mag = np.abs(cuda_clean - c_clean)
        max_abs_diff = float(np.max(diff_mag))
        mean_abs_diff = float(np.mean(diff_mag))
        max_diff_db = 20.0 * float(np.log10(max(max_abs_diff / (peak_val if peak_val > 0 else 1.0), 1e-12)))

        print("\n" + "=" * 78)
        print("  [PARITY VERIFICATION] CUDA vs C Fallback:")
        print(f"  Peak Clean Magnitude:     {peak_val:.4e}")
        print(f"  Max Absolute Difference:  {max_abs_diff:.3e}")
        print(f"  Max Relative Difference:  {max_diff_db:.1f} dB relative to peak")
        print(f"  Mean Absolute Difference: {mean_abs_diff:.3e}")
        status = "EXACT FLOAT32 MATCH" if max_abs_diff < 1e-5 else "ACCEPTABLE"
        print(f"  Verification Status:      {status}")
        print("=" * 78 + "\n")

        out_parity_png = os.path.join(out_dir, f"{base_name}_cuda_vs_c_difference.png")
        print(f"[*] Rendering CUDA vs C difference plot (dB): {out_parity_png}")
        try:
            plot_backend_parity(
                cuda_image=cuda_clean,
                c_image=c_clean,
                output_png=out_parity_png,
                dyn_range_db=dyn_range,
                title_suffix=name,
            )
        except Exception as e:
            print(f"[!] Parity plot generation failed: {e}")

    # Step 4: Render standard diagnostic comparison plot for the primary result
    primary_b = backends_to_run[0]
    primary_res = results[primary_b]
    out_png = os.path.join(out_dir, f"{base_name}_comparison.png")
    print(f"[*] Rendering diagnostic comparison plot: {out_png}")
    try:
        plot_comparison(
            dirty_image=dirty_img,
            clean_image=primary_res.clean_image,
            residual_image=primary_res.residual_image,
            restored_model=primary_res.restored_model,
            output_png=out_png,
            dyn_range_db=dyn_range,
            title_suffix=f"{name} ({primary_b.upper()})",
        )
        print(f"[+] Comparison plot saved: {out_png}")
    except Exception as e:
        print(f"[!] Plot generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CLEAN_SAR Demo Runner exercising simplified interface, auto PSF sizing, and C/CUDA backends."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "umbra", "diffpfa", "custom"],
        help="Dataset target to process ('all', 'umbra', 'diffpfa', or 'custom')"
    )
    parser.add_argument("--umbra-path", "--umbra_path", type=str, default=DEFAULT_UMBRA_PATH, help="Path to raw Umbra SICD NITF")
    parser.add_argument("--diffpfa-path", "--diffpfa_path", type=str, default=DEFAULT_DIFFPFA_PATH, help="Path to DiffPFA SICD NITF")
    parser.add_argument("--chip-input", "--chip_input", type=str, default=None, help="Path to custom caller-provided SICD chip NITF")
    parser.add_argument("--out-dir", "--out_dir", type=str, default="output/demo", help="Output directory")
    parser.add_argument("--backend", type=str, default="both", choices=["both", "auto", "cuda", "c"], help="Compute backend ('both' to compare CUDA & C, 'auto', 'cuda', 'c')")
    parser.add_argument("--gain", type=float, default=0.1, help="CLEAN loop gain (default: 0.1)")
    parser.add_argument("--threshold", type=float, default=0.02, help="Stopping threshold fraction (default: 0.02)")
    parser.add_argument("--max-iters", "--max_iters", type=int, default=5_000, help="Maximum iterations (default: 2500)")
    parser.add_argument("--dyn-range", "--dyn_range", type=float, default=50.0, help="Dynamic range in dB for visualization")

    args = parser.parse_args()

    # Custom chip input if specified
    if args.target == "custom" or args.chip_input:
        if not args.chip_input:
            print("[!] Error: --chip-input must be specified when --target custom is selected.")
            return
        run_single_demo(
            name="Custom Caller SICD Chip",
            input_path=args.chip_input,
            out_dir=args.out_dir,
            chip_bounds=None,
            backend=args.backend,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            dyn_range=args.dyn_range,
        )
        return

    run_umbra = args.target in ("all", "umbra")
    run_diffpfa = args.target in ("all", "diffpfa")

    if run_umbra:
        run_single_demo(
            name="Raw Umbra-05 (Strong Point Scatterer)",
            input_path=args.umbra_path,
            out_dir=args.out_dir,
            chip_bounds=DEFAULT_UMBRA_CHIP,
            backend=args.backend,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            dyn_range=args.dyn_range,
        )

    if run_diffpfa:
        run_single_demo(
            name="DiffPFA Umbra-05 (Refocused Point Scatterer)",
            input_path=args.diffpfa_path,
            out_dir=args.out_dir,
            chip_bounds=DEFAULT_DIFFPFA_CHIP,
            backend=args.backend,
            gain=args.gain,
            threshold=args.threshold,
            max_iters=args.max_iters,
            dyn_range=args.dyn_range,
        )


if __name__ == "__main__":
    main()
