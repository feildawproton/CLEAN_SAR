import numpy as np
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator
from clean_sar.sicd_handler import SICDHandler

file_path = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
handler = SICDHandler(file_path)

# Chip bounds from demo: (1932, 3931, 2188, 4187)
r_min, c_min, r_max, c_max = 1932, 3931, 2188, 4187
config_chip = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(r_min, c_min))
psf_gen_chip = PSFGenerator(config_chip)

# Compare PSF at (0, 0) vs (255, 255) in chip
psf_00 = psf_gen_chip.compute_psf(0, 0, psf_size=65)
psf_mid = psf_gen_chip.compute_psf(128, 128, psf_size=65)
psf_end = psf_gen_chip.compute_psf(255, 255, psf_size=65)

# Non-rotated PSF at SCP
psf_scp = psf_gen_chip.compute_psf(handler.scp_pixel[0] - r_min, handler.scp_pixel[1] - c_min, psf_size=65)

diff_intra_chip = np.max(np.abs(psf_00 - psf_end))
diff_vs_scp = np.max(np.abs(psf_mid - psf_scp))

print(f"Max abs diff between corners of 256x256 chip: {diff_intra_chip:.6e}")
print(f"Max abs diff between chip center and SCP:      {diff_vs_scp:.6e}")

# Check full scene corner vs SCP
config_full = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(0, 0))
psf_gen_full = PSFGenerator(config_full)
psf_corner = psf_gen_full.compute_psf(0, 0, psf_size=65)
psf_center = psf_gen_full.compute_psf(handler.scp_pixel[0], handler.scp_pixel[1], psf_size=65)
diff_full_scene = np.max(np.abs(psf_corner - psf_center))
print(f"Max abs diff across full scene (5086x8631):     {diff_full_scene:.6e}")

