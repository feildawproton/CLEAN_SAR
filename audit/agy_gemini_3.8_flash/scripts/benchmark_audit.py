import time
import numpy as np
import torch
from clean_sar.sicd_handler import SICDHandler
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean

file_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
handler = SICDHandler(file_path)

# Chip 512x512
r0, c0 = 1900, 3900
chip, _ = handler.read_chip(r0, c0, r0 + 512, c0 + 512)
config = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(r0, c0))

print(f"Benchmarking 512x512 chip on GPU...")
print(f"Device: {torch.cuda.get_device_name(0)}")

# PyTorch
t0 = time.perf_counter()
res_torch = run_hogbom_clean(chip, config=config, backend="pytorch", gain=0.1, threshold=0.02, max_iters=500, psf_size=65)
torch_time = time.perf_counter() - t0

# CUDA
t0 = time.perf_counter()
res_cuda = run_hogbom_clean(chip, config=config, backend="cuda", gain=0.1, threshold=0.02, max_iters=500, psf_size=65)
cuda_time = time.perf_counter() - t0

print(f"\n--- PYTORCH GPU ---")
print(f"Iterations: {res_torch.iterations}")
print(f"Total time: {torch_time:.3f} s")
print(f"Pure compute time: {res_torch.pure_compute_time_sec*1000:.2f} ms ({res_torch.pure_compute_time_sec*1000/res_torch.iterations:.2f} ms/iter)")
print(f"Suppression: {res_torch.suppression_db:.2f} dB")

print(f"\n--- NATIVE CUDA (NVRTC) ---")
print(f"Iterations: {res_cuda.iterations}")
print(f"Total time: {cuda_time:.3f} s")
print(f"Pure compute time: {res_cuda.pure_compute_time_sec*1000:.2f} ms ({res_cuda.pure_compute_time_sec*1000/res_cuda.iterations:.2f} ms/iter)")
print(f"Suppression: {res_cuda.suppression_db:.2f} dB")

speedup = res_torch.pure_compute_time_sec / max(res_cuda.pure_compute_time_sec, 1e-6)
print(f"\nPure compute speedup (CUDA vs PyTorch): {speedup:.2f}x")
print(f"Absolute output max difference: {np.max(np.abs(res_torch.clean_image - res_cuda.clean_image)):.6e}")

