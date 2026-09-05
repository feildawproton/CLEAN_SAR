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
    row_wgt="UNIFORM", col_wgt="UNIFORM",
)

# Synthesize an off-grid point target at (64.35, 64.25)
# Using continuous sinc evaluation
dr_idx = np.arange(128) - 64.35
dc_idx = np.arange(128) - 64.25
DR, DC = np.meshgrid(dr_idx, dc_idx, indexing="ij")
U = DR * config.row_ss
V = DC * config.col_ss
dirty = 10.0 * np.sinc(config.row_bw * U) * np.sinc(config.col_bw * V)
dirty = dirty.astype(np.complex64)

res = run_hogbom_clean(dirty, config=config, backend="pytorch", gain=0.1, threshold=0.01, max_iters=200)

print(f"Off-grid target deconvolution:")
print(f"  Iterations: {res.iterations}")
print(f"  Suppression dB: {res.suppression_db:.2f} dB")
print(f"  Non-zero components: {res.num_components}")
print(f"  Initial peak: {res.initial_peak:.4f}, Final residual: {res.final_peak:.4f}")

# Check top 5 component locations and amplitudes
coords = res.history_coords[:10]
print("  First 10 component coordinates:")
for i, c in enumerate(coords):
    print(f"    Iter {i+1}: peak at {c}, comp amp = {res.components_map[c]:.4f}")

