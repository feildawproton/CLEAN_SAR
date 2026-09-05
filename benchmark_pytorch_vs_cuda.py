#!/usr/bin/env python3
"""
benchmark_pytorch_vs_cuda.py: Side-by-side performance benchmarking comparing
PyTorch GPU vs. Native C++/CUDA GPU backends on full scenes.

Pulls out Disk Read, Host-to-Device (H2D), Pure GPU Compute, Device-to-Host (D2H),
and Disk Write as separate metrics to accurately measure raw compute speedup.
"""

import os
import sys
import time
import json
import csv
import argparse
from glob import glob
from typing import Dict, Any, List
import numpy as np

from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean


def benchmark_file(
    file_path: str,
    max_iters: int = 1000,
    gain: float = 0.1,
    threshold: float = 0.02,
    psf_size: int = 65,
    beam_type: str = "gaussian",
    runs: int = 3,
    warmup: bool = True,
) -> Dict[str, Any]:
    base_name = os.path.basename(file_path)
    print(f"\n{'='*85}")
    print(f"  [BENCHMARK] Comparing PyTorch vs Native CUDA: {base_name}")
    print(f"  Iterations: {max_iters} | Gain: {gain} | Threshold: {threshold} | PSF Size: {psf_size}x{psf_size} | Runs: {runs}")
    print(f"{'='*85}")

    # 1. Disk Read Time
    t0_read = time.perf_counter()
    handler = SICDHandler(file_path)
    dirty_image = handler.read_full_image()
    disk_read_sec = time.perf_counter() - t0_read

    H, W = dirty_image.shape
    mpixels = round((H * W) / 1e6, 2)
    print(f"[*] Loaded full scene: {H}x{W} ({mpixels} Mpixels) in {disk_read_sec:.3f}s")

    config = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(0, 0))

    # Optional Warm-up run (discarded)
    if warmup:
        print("[*] Running warm-up iteration on 128x128 crop (discarded)...")
        crop_h = min(128, H)
        crop_w = min(128, W)
        crop = dirty_image[:crop_h, :crop_w]
        crop_cfg = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(0, 0))
        run_hogbom_clean(
            dirty_image=crop, config=crop_cfg, backend="pytorch",
            beam_type=beam_type, psf_size=min(psf_size, 33), gain=gain,
            threshold=threshold, max_iters=min(max_iters, 20), verbose=False,
        )
        run_hogbom_clean(
            dirty_image=crop, config=crop_cfg, backend="cuda",
            beam_type=beam_type, psf_size=min(psf_size, 33), gain=gain,
            threshold=threshold, max_iters=min(max_iters, 20), verbose=False,
        )

    torch_comp_list: List[float] = []
    torch_h2d_list: List[float] = []
    torch_d2h_list: List[float] = []
    torch_total_list: List[float] = []
    torch_suppr_list: List[float] = []

    cuda_comp_list: List[float] = []
    cuda_h2d_list: List[float] = []
    cuda_d2h_list: List[float] = []
    cuda_total_list: List[float] = []
    cuda_suppr_list: List[float] = []

    res_torch = None
    res_cuda = None

    for run_idx in range(runs):
        run_lbl = f" (run {run_idx + 1}/{runs})" if runs > 1 else ""
        # 2. PyTorch GPU Execution
        print(f"[1/2] Executing PyTorch GPU Backend{run_lbl}...")
        res_t = run_hogbom_clean(
            dirty_image=dirty_image,
            config=config,
            backend="pytorch",
            beam_type=beam_type,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            verbose=False,
        )
        t_comp = res_t.pure_compute_time_sec * 1000.0
        t_h2d = res_t.h2d_time_sec * 1000.0
        t_d2h = res_t.d2h_time_sec * 1000.0
        t_tot = (res_t.h2d_time_sec + res_t.pure_compute_time_sec + res_t.d2h_time_sec) * 1000.0
        torch_comp_list.append(t_comp)
        torch_h2d_list.append(t_h2d)
        torch_d2h_list.append(t_d2h)
        torch_total_list.append(t_tot)
        torch_suppr_list.append(float(res_t.suppression_db))
        res_torch = res_t

        # 3. Native CUDA GPU Execution
        print(f"[2/2] Executing Native CUDA GPU Backend{run_lbl}...")
        res_c = run_hogbom_clean(
            dirty_image=dirty_image,
            config=config,
            backend="cuda",
            beam_type=beam_type,
            psf_size=psf_size,
            gain=gain,
            threshold=threshold,
            max_iters=max_iters,
            verbose=False,
        )
        c_comp = res_c.pure_compute_time_sec * 1000.0
        c_h2d = res_c.h2d_time_sec * 1000.0
        c_d2h = res_c.d2h_time_sec * 1000.0
        c_tot = (res_c.h2d_time_sec + res_c.pure_compute_time_sec + res_c.d2h_time_sec) * 1000.0
        cuda_comp_list.append(c_comp)
        cuda_h2d_list.append(c_h2d)
        cuda_d2h_list.append(c_d2h)
        cuda_total_list.append(c_tot)
        cuda_suppr_list.append(float(res_c.suppression_db))
        res_cuda = res_c

    torch_compute_ms = float(np.median(torch_comp_list))
    torch_h2d_ms = float(np.median(torch_h2d_list))
    torch_d2h_ms = float(np.median(torch_d2h_list))
    torch_total_gpu_ms = float(np.median(torch_total_list))
    torch_ms_per_iter = torch_compute_ms / max(res_torch.iterations, 1)

    cuda_compute_ms = float(np.median(cuda_comp_list))
    cuda_h2d_ms = float(np.median(cuda_h2d_list))
    cuda_d2h_ms = float(np.median(cuda_d2h_list))
    cuda_total_gpu_ms = float(np.median(cuda_total_list))
    cuda_ms_per_iter = cuda_compute_ms / max(res_cuda.iterations, 1)

    compute_speedup = torch_compute_ms / max(cuda_compute_ms, 1e-6)
    total_gpu_speedup = torch_total_gpu_ms / max(cuda_total_gpu_ms, 1e-6)
    delta_suppr = abs(float(np.median(torch_suppr_list)) - float(np.median(cuda_suppr_list)))

    print(f"      PyTorch Pure Compute (median): {torch_compute_ms:.2f} ms ({torch_ms_per_iter:.3f} ms/iter)")
    print(f"      Native CUDA Pure Compute (median): {cuda_compute_ms:.2f} ms ({cuda_ms_per_iter:.3f} ms/iter)")
    print(f"  [+] Pure GPU Compute Speedup: {compute_speedup:.2f}x (CUDA vs PyTorch)")
    print(f"  [+] Total GPU Speedup:        {total_gpu_speedup:.2f}x (H2D + Compute + D2H)")
    print(f"  [+] Suppression Parity Delta: {delta_suppr:.4f} dB")

    return {
        "filename": base_name,
        "height": H,
        "width": W,
        "mpixels": mpixels,
        "iterations": res_torch.iterations,
        "runs": runs,
        "disk_read_sec": round(disk_read_sec, 3),
        "torch_h2d_ms": round(torch_h2d_ms, 2),
        "torch_compute_ms": round(torch_compute_ms, 2),
        "torch_d2h_ms": round(torch_d2h_ms, 2),
        "torch_total_gpu_ms": round(torch_total_gpu_ms, 2),
        "torch_ms_per_iter": round(torch_ms_per_iter, 3),
        "torch_suppression_db": round(float(np.median(torch_suppr_list)), 2),
        "cuda_h2d_ms": round(cuda_h2d_ms, 2),
        "cuda_compute_ms": round(cuda_compute_ms, 2),
        "cuda_d2h_ms": round(cuda_d2h_ms, 2),
        "cuda_total_gpu_ms": round(cuda_total_gpu_ms, 2),
        "cuda_ms_per_iter": round(cuda_ms_per_iter, 3),
        "cuda_suppression_db": round(float(np.median(cuda_suppr_list)), 2),
        "compute_speedup": round(compute_speedup, 2),
        "total_gpu_speedup": round(total_gpu_speedup, 2),
        "delta_suppression_db": round(delta_suppr, 4),
    }


def print_comparison_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 118)
    print("  HEAD-TO-HEAD BENCHMARK SUMMARY: PyTorch GPU vs. Native C++/CUDA GPU (Medians)")
    print("=" * 118)
    header = (
        f"{'Dataset':<32} | {'Mpix':<5} | {'PyTorch Compute':<16} | {'CUDA Compute':<15} | "
        f"{'Compute':<8} | {'Total GPU':<9} | {'Suppr Parity':<12}"
    )
    print(header)
    print("-" * 118)

    for r in results:
        row = (
            f"{r['filename'][:31]:<32} | "
            f"{r['mpixels']:<5.1f} | "
            f"{r['torch_compute_ms']:>8.1f} ms ({r['torch_ms_per_iter']:>4.2f}ms/it) | "
            f"{r['cuda_compute_ms']:>7.1f} ms ({r['cuda_ms_per_iter']:>4.2f}ms/it) | "
            f"{r['compute_speedup']:>6.2f}x  | "
            f"{r['total_gpu_speedup']:>7.2f}x  | "
            f"Δ {r['delta_suppression_db']:>4.2f} dB"
        )
        print(row)
    print("-" * 118)

    avg_comp_speedup = np.mean([r["compute_speedup"] for r in results])
    avg_total_speedup = np.mean([r["total_gpu_speedup"] for r in results])
    avg_torch_ms = np.mean([r["torch_ms_per_iter"] for r in results])
    avg_cuda_ms = np.mean([r["cuda_ms_per_iter"] for r in results])
    print(f"  Average Pure Compute Speedup:    {avg_comp_speedup:.2f}x")
    print(f"  Average Total GPU Speedup (e2e): {avg_total_speedup:.2f}x")
    print(f"  Average PyTorch Latency:         {avg_torch_ms:.3f} ms / iteration")
    print(f"  Average Native CUDA Latency:     {avg_cuda_ms:.3f} ms / iteration")
    print("=" * 118 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Head-to-Head PyTorch vs CUDA Benchmark.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/benchmarks_comparison")
    parser.add_argument("--pattern", type=str, default="*.nitf")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--psf_size", type=int, default=65)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions per file to measure medians (default: 3).")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True, help="Perform a discarded warm-up run before measuring (default: True).")

    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"[!] No files matching '{args.pattern}' in {args.input_dir}")
        sys.exit(1)

    if args.limit:
        files = files[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Benchmarking {len(files)} files head-to-head (runs={args.runs}, warmup={args.warmup})...")

    results = []
    for f in files:
        try:
            res = benchmark_file(
                file_path=f,
                max_iters=args.max_iters,
                gain=args.gain,
                threshold=args.threshold,
                psf_size=args.psf_size,
                runs=args.runs,
                warmup=args.warmup,
            )
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed {os.path.basename(f)}: {e}")
            import traceback
            traceback.print_exc()

    print_comparison_table(results)

    # Save JSON and CSV
    json_path = os.path.join(args.output_dir, "pytorch_vs_cuda_benchmark.json")
    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)

    csv_path = os.path.join(args.output_dir, "pytorch_vs_cuda_benchmark.csv")
    if results:
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"[+] Saved comparison JSON: {json_path}")
    print(f"[+] Saved comparison CSV:  {csv_path}")


if __name__ == "__main__":
    main()
