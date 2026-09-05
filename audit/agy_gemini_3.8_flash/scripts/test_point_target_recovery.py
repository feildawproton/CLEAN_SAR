import numpy as np
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.algorithm import run_hogbom_clean

# 1. Setup physics config
row_ss = 0.5
col_ss = 0.5
row_bw = 1.5
col_bw = 1.5
row_wid = 0.886 / row_bw
col_wid = 0.886 / col_bw

config = CleanPhysicsConfig(
    row_ss=row_ss, col_ss=col_ss,
    row_bw=row_bw, col_bw=col_bw,
    row_wid=row_wid, col_wid=col_wid,
    scp_slant_range=600000.0,
    scp_row=64.0, scp_col=64.0,
    row_wgt="UNIFORM", col_wgt="UNIFORM",
)

psf_gen = PSFGenerator(config)
psf65 = psf_gen.compute_psf(64, 64, psf_size=65)

# 2. Create synthetic image with a single point target at (64, 64) with amplitude 10.0 + 0j
H, W = 128, 128
dirty = np.zeros((H, W), dtype=np.complex64)
# embed PSF centered at (64, 64)
kh = 32
target_amp = 10.0 + 0.0j
dirty[64 - kh : 64 + kh + 1, 64 - kh : 64 + kh + 1] = target_amp * psf65

# Measure PSLR of dirty image along range cut through (64, 64)
cut_dirty = np.abs(dirty[:, 64])
# Peak is at 64
peak_dirty = cut_dirty[64]
# Find first sidelobe (around dr = +- 2 or 3)
sidelobes_dirty = []
for i in range(1, H - 1):
    if abs(i - 64) > 1 and cut_dirty[i] > cut_dirty[i-1] and cut_dirty[i] > cut_dirty[i+1]:
        sidelobes_dirty.append(cut_dirty[i])
max_sl_dirty = max(sidelobes_dirty) if sidelobes_dirty else 0.0
pslr_dirty_db = 20.0 * np.log10(max_sl_dirty / peak_dirty)

print(f"=== DIRTY IMAGE METRICS ===")
print(f"Initial Peak: {peak_dirty:.4f} (Expected: {abs(target_amp):.4f})")
print(f"First Sidelobe Peak: {max_sl_dirty:.4f}")
print(f"Dirty PSLR: {pslr_dirty_db:.2f} dB")

# 3. Run CLEAN with PyTorch
res_torch = run_hogbom_clean(
    dirty, config=config, backend="pytorch",
    gain=0.1, threshold=0.01, max_iters=500, psf_size=65
)

# 4. Run CLEAN with CUDA
res_cuda = run_hogbom_clean(
    dirty, config=config, backend="cuda",
    gain=0.1, threshold=0.01, max_iters=500, psf_size=65
)

for name, res in [("PyTorch", res_torch), ("CUDA", res_cuda)]:
    print(f"\n=== {name.upper()} CLEAN RESULTS ===")
    print(f"Iterations: {res.iterations}")
    print(f"Reported 'suppression_db': {res.suppression_db:.2f} dB")
    
    # Recovered component amplitude
    comp_amp = np.sum(res.components_map[63:66, 63:66])
    print(f"Extracted component sum at target: {comp_amp.real:.4f} + {comp_amp.imag:.4f}j (target: {target_amp})")
    
    # Clean image metrics
    clean_cut = np.abs(res.clean_image[:, 64])
    peak_clean = clean_cut[64]
    
    sidelobes_clean = []
    for i in range(1, H - 1):
        if abs(i - 64) > 1 and clean_cut[i] > clean_cut[i-1] and clean_cut[i] > clean_cut[i+1]:
            sidelobes_clean.append(clean_cut[i])
    max_sl_clean = max(sidelobes_clean) if sidelobes_clean else 0.0
    pslr_clean_db = 20.0 * np.log10(max_sl_clean / peak_clean) if max_sl_clean > 0 else -999.0
    true_pslr_improvement = pslr_dirty_db - pslr_clean_db
    
    print(f"Clean Peak: {peak_clean:.4f}")
    print(f"Clean Sidelobe Peak: {max_sl_clean:.6f}")
    print(f"Clean PSLR: {pslr_clean_db:.2f} dB")
    print(f"TRUE PSLR Improvement: {true_pslr_improvement:.2f} dB")
    print(f"Max residual: {np.max(np.abs(res.residual_image)):.6f}")

