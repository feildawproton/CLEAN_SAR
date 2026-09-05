import argparse
import os
import sys
import time
import json
import csv
from glob import glob
from typing import Dict, Any, List
import numpy as np
import torch

from clean_sar.processor import CLEANProcessor
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean, CleanResult


def benchmark_single_sicd(
    input_path: str,
    output_dir: str,
    max_iters: int = 2500,
    gain: float = 0.1,
    threshold: float = 0.02,
    backend: str = "auto",
    beam_type: str = "gaussian",
    psf_size: int = 65,
    guard_margin: int = 0,
    device: str = None,
) -> Dict[str, Any]:
    """
    Executes full-scene CLEAN deconvolution on a single NITF SICD and records granular timings and metrics.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_clean.nitf")

    print(f"\n{'='*75}")
    print(f"  [BENCHMARK] Processing full scene: {os.path.basename(input_path)}")
    print(f"  Output:    {out_path}")
    print(f"  Backend:   {backend.upper()} | Beam: {beam_type.upper()}")
    print(f"  Gain:      {gain} | Threshold: {threshold} | Max Iters: {max_iters}")
    print(f"{'='*75}")

    # 1. Setup & Read Full Scene Image
    t0_setup = time.perf_counter()
    processor = CLEANProcessor(
        input_path=input_path,
        output_path=out_path,
        chip_bounds=None,  # Full scene, no chipping
        backend=backend,
        device=device,
    )

    t0_read = time.perf_counter()
    dirty_image = processor.handler.read_full_image()
    read_time = time.perf_counter() - t0_read

    setup_and_read_time = time.perf_counter() - t0_setup
    H, W = dirty_image.shape
    print(f"[*] Loaded full scene: {H} rows x {W} cols ({H * W / 1e6:.2f} Mpixels) in {setup_and_read_time:.3f}s")

    # 2. Extract Physics Config
    config = CleanPhysicsConfig.from_sicd_handler(processor.handler, chip_start=(0, 0))

    # 3. Execute CLEAN Deconvolution
    t0_clean = time.perf_counter()
    result = run_hogbom_clean(
        dirty_image=dirty_image,
        config=config,
        backend=backend,
        beam_type=beam_type,
        psf_size=psf_size,
        gain=gain,
        threshold=threshold,
        max_iters=max_iters,
        guard_margin=guard_margin,
        device=processor.device,
        verbose=True,
    )
    clean_proc_time = time.perf_counter() - t0_clean
    print(f"[+] Deconvolution complete in {clean_proc_time:.3f}s "
          f"({result.iterations} iters, {result.suppression_db:.1f} dB suppression)")

    # 4. Write Output NITF SICD
    t0_write = time.perf_counter()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    processor.handler.write_nitf(
        output_path=out_path,
        complex_image=result.clean_image,
    )
    write_time = time.perf_counter() - t0_write
    print(f"[*] Written clean SICD NITF in {write_time:.3f}s")

    total_time = setup_and_read_time + clean_proc_time + write_time
    print(f"[TIMING] Total pipeline took {total_time:.3f}s (Read: {setup_and_read_time:.3f}s, Clean: {clean_proc_time:.3f}s, Write: {write_time:.3f}s)")

    return {
        "filename": os.path.basename(input_path),
        "height": H,
        "width": W,
        "mpixels": round(H * W / 1e6, 2),
        "iterations": result.iterations,
        "initial_peak": float(result.initial_peak),
        "final_peak": float(result.final_peak),
        "suppression_db": round(float(result.suppression_db), 2),
        "setup_and_read_time": setup_and_read_time,
        "clean_proc_time": clean_proc_time,
        "write_time": write_time,
        "total_time": total_time,
        "ms_per_iter": round((clean_proc_time * 1000.0) / max(result.iterations, 1), 3),
    }


def print_statistics_table(all_results: List[Dict[str, Any]]):
    """Prints aggregated summary statistics across all processed datasets."""
    n = len(all_results)
    if n == 0:
        return

    print("\n" + "=" * 75)
    print(f"  BENCHMARK TIMING & PERFORMANCE STATISTICS (N = {n} Full Scenes)")
    print("=" * 75)

    metrics_keys = [
        ("setup_and_read_time", "Setup & Read Time (s)"),
        ("clean_proc_time",     "Deconvolution Time (s)"),
        ("write_time",          "NITF Write Time (s)"),
        ("total_time",          "Total Pipeline Time (s)"),
        ("ms_per_iter",         "Latency per Iteration (ms/iter)"),
        ("suppression_db",      "Sidelobe Suppression (dB)"),
    ]

    for key, label in metrics_keys:
        data = [r[key] for r in all_results if key in r]
        arr = np.array(data, dtype=np.float64)
        print(f"\n{label.upper()}:")
        print(f"  Mean:   {np.mean(arr):.4f}")
        print(f"  Median: {np.median(arr):.4f}")
        print(f"  StdDev: {np.std(arr):.4f}")
        print(f"  Min:    {np.min(arr):.4f}")
        print(f"  Max:    {np.max(arr):.4f}")
        print(f"  25th %: {np.percentile(arr, 25):.4f}")
        print(f"  75th %: {np.percentile(arr, 75):.4f}")
        print("-" * 50)


def save_reports(all_results: List[Dict[str, Any]], output_dir: str):
    """Saves benchmark results in JSON and CSV formats."""
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved detailed benchmark JSON report to: {json_path}")

    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[+] Saved benchmark summary CSV table to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch full-scene CLEAN deconvolution benchmarks across a directory of SICD NITFs."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .nitf SICD files.")
    parser.add_argument("--output_dir", type=str, default="output/benchmarks", help="Directory to save clean SICDs and benchmark metrics.")
    parser.add_argument("--pattern", type=str, default="*.nitf", help="Glob pattern for selecting files (default: '*.nitf').")
    parser.add_argument("--backend", choices=["auto", "pytorch", "cuda"], default="auto", help="Compute backend (default: 'auto').")
    parser.add_argument("--beam", choices=["gaussian", "mainlobe"], default="gaussian", help="Restoring beam type (default: 'gaussian').")
    parser.add_argument("--gain", type=float, default=0.1, help="CLEAN loop gain gamma (default: 0.1).")
    parser.add_argument("--threshold", type=float, default=0.02, help="Stopping threshold fraction (default: 0.02).")
    parser.add_argument("--max_iters", type=int, default=2500, help="Maximum iterations per scene (default: 2500).")
    parser.add_argument("--psf_size", type=int, default=65, help="Kernel dimension for local PSF (default: 65).")
    parser.add_argument("--guard_margin", type=int, default=0, help="Border exclusion margin in pixels (default: 0).")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files to process.")

    args = parser.parse_args()

    input_files = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not input_files:
        print(f"[!] No files matching '{args.pattern}' found in {args.input_dir}")
        sys.exit(1)

    if args.limit:
        input_files = input_files[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(input_files)} SICD NITF files in {args.input_dir} to benchmark.")

    all_results = []
    for file_path in input_files:
        try:
            res = benchmark_single_sicd(
                input_path=file_path,
                output_dir=args.output_dir,
                max_iters=args.max_iters,
                gain=args.gain,
                threshold=args.threshold,
                backend=args.backend,
                beam_type=args.beam,
                psf_size=args.psf_size,
                guard_margin=args.guard_margin,
                device=args.device,
            )
            all_results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed processing {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()

    print_statistics_table(all_results)
    save_reports(all_results, args.output_dir)


if __name__ == "__main__":
    main()
