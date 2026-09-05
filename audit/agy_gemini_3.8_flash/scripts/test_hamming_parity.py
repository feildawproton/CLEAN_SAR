import numpy as np
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.algorithm import run_hogbom_clean

config = CleanPhysicsConfig(
    row_ss=0.5, col_ss=0.5,
    row_bw=1.5, col_bw=1.5,
    row_wid=0.886/1.5, col_wid=0.886/1.5,
    scp_slant_range=600000.0,
    scp_row=64.0, scp_col=64.0,
    row_wgt="HAMMING", col_wgt="HAMMING",
)

psf_gen = PSFGenerator(config)
psf65 = psf_gen.compute_psf(64, 64, psf_size=65)

dirty = np.zeros((128, 128), dtype=np.complex64)
dirty[64 - 32 : 64 + 33, 64 - 32 : 64 + 33] = 10.0 * psf65

res_torch = run_hogbom_clean(dirty, config=config, backend="pytorch", gain=0.1, threshold=0.01, max_iters=200)
res_cuda = run_hogbom_clean(dirty, config=config, backend="cuda", gain=0.1, threshold=0.01, max_iters=200)

print(f"HAMMING Window Results:")
print(f"  PyTorch iterations: {res_torch.iterations} | Final residual max: {np.max(np.abs(res_torch.residual_image)):.4f}")
print(f"  CUDA iterations:    {res_cuda.iterations} | Final residual max: {np.max(np.abs(res_cuda.residual_image)):.4f}")
print(f"  PyTorch suppression: {res_torch.suppression_db:.2f} dB | CUDA suppression: {res_cuda.suppression_db:.2f} dB")
