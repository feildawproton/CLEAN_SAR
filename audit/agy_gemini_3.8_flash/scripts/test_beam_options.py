import numpy as np
from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean

# Synthetic dirty image with a single point target
dirty = np.zeros((128, 128), dtype=np.complex64)
dirty[64, 64] = 10.0 + 5.0j

config = CleanPhysicsConfig(
    row_ss=0.5, col_ss=0.5,
    row_bw=1.5, col_bw=1.5,
    row_wid=0.886/1.5, col_wid=0.886/1.5,
    scp_slant_range=600000.0,
    scp_row=64.0, scp_col=64.0,
)

res_torch_main = run_hogbom_clean(dirty, config=config, backend="pytorch", beam_type="mainlobe", max_iters=5)
res_cuda_main = run_hogbom_clean(dirty, config=config, backend="cuda", beam_type="mainlobe", max_iters=5)
res_cuda_gauss = run_hogbom_clean(dirty, config=config, backend="cuda", beam_type="gaussian", max_iters=5)

diff_cuda_main_vs_gauss = np.max(np.abs(res_cuda_main.clean_image - res_cuda_gauss.clean_image))
diff_torch_vs_cuda_main = np.max(np.abs(res_torch_main.clean_image - res_cuda_main.clean_image))

print(f"Diff between CUDA 'mainlobe' and CUDA 'gaussian': {diff_cuda_main_vs_gauss:.6e}")
print(f"Diff between PyTorch 'mainlobe' and CUDA 'mainlobe': {diff_torch_vs_cuda_main:.6e}")
