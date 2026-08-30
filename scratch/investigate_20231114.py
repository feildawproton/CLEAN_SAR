import numpy as np
import matplotlib.pyplot as plt
import sarkit.sicd as ss
from clean_sar import SICDHandler

f_raw = "/home/feildaw/data/2023-11-14-03-38-20_UMBRA-04_SICD.nitf"
f_diff = "/home/feildaw/diffpfa/workspace/output/2023-11-14-03-38-20_UMBRA-04_SICDU_X_X.nitf"

print("================ METADATA COMPARISON ================")
h_raw = SICDHandler(f_raw)
h_diff = SICDHandler(f_diff)

print("RAW METADATA:")
print(f"  Dimensions:   {h_raw.num_rows} x {h_raw.num_cols}")
print(f"  Row SS:       {h_raw.row_ss:.4f} m | Col SS: {h_raw.col_ss:.4f} m")
print(f"  Row BW:       {h_raw.row_bw:.4f} cyc/m | Col BW: {h_raw.col_bw:.4f} cyc/m")
print(f"  Row Wid:      {h_raw.row_wid:.4f} m | Col Wid: {h_raw.col_wid:.4f} m")
print(f"  Row Wgt:      {h_raw.row_wgt_name} | Col Wgt: {h_raw.col_wgt_name}")
print(f"  Is PFA:       {h_raw.is_pfa}")
if h_raw.is_pfa:
    print(f"  PFA Krg:      [{h_raw.pfa_meta['Krg1']:.4f}, {h_raw.pfa_meta['Krg2']:.4f}]")
    print(f"  PFA Kaz:      [{h_raw.pfa_meta['Kaz1']:.4f}, {h_raw.pfa_meta['Kaz2']:.4f}]")

print("\nDIFFPFA METADATA:")
print(f"  Dimensions:   {h_diff.num_rows} x {h_diff.num_cols}")
print(f"  Row SS:       {h_diff.row_ss:.4f} m | Col SS: {h_diff.col_ss:.4f} m")
print(f"  Row BW:       {h_diff.row_bw:.4f} cyc/m | Col BW: {h_diff.col_bw:.4f} cyc/m")
print(f"  Row Wid:      {h_diff.row_wid:.4f} m | Col Wid: {h_diff.col_wid:.4f} m")
print(f"  Row Wgt:      {h_diff.row_wgt_name} | Col Wgt: {h_diff.col_wgt_name}")
print(f"  Is PFA:       {h_diff.is_pfa}")
if h_diff.is_pfa:
    print(f"  PFA Krg:      [{h_diff.pfa_meta['Krg1']:.4f}, {h_diff.pfa_meta['Krg2']:.4f}]")
    print(f"  PFA Kaz:      [{h_diff.pfa_meta['Kaz1']:.4f}, {h_diff.pfa_meta['Kaz2']:.4f}]")

# Extract chips around the prominent scatterer
chip_raw, _ = h_raw.read_chip(1932, 3931, 2188, 4187)
chip_diff, _ = h_diff.read_chip(2756, 2860, 3012, 3116)

# Find exact peak locations in chip
r_pk_raw, c_pk_raw = np.unravel_index(np.argmax(np.abs(chip_raw)), chip_raw.shape)
r_pk_diff, c_pk_diff = np.unravel_index(np.argmax(np.abs(chip_diff)), chip_diff.shape)

print(f"\nRAW Peak in chip: ({r_pk_raw}, {c_pk_raw}) Mag: {np.abs(chip_raw[r_pk_raw, c_pk_raw]):.4e}")
print(f"DIFFPFA Peak in chip: ({r_pk_diff}, {c_pk_diff}) Mag: {np.abs(chip_diff[r_pk_diff, c_pk_diff]):.4e}")

# 2D FFT / Spectrum analysis of the chip
spec_raw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(chip_raw)))
spec_diff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(chip_diff)))

# Plot 1D cuts through peak and 2D spectra
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1D Cuts (Row = Range / Vertical, Col = Azimuth / Horizontal)
# RAW 1D Cuts
raw_row_cut = np.abs(chip_raw[:, c_pk_raw])
raw_col_cut = np.abs(chip_raw[r_pk_raw, :])
raw_row_db = 20 * np.log10(np.maximum(raw_row_cut / np.max(raw_row_cut), 1e-4))
raw_col_db = 20 * np.log10(np.maximum(raw_col_cut / np.max(raw_col_cut), 1e-4))

axes[0, 0].plot(np.arange(-128, 128) * h_raw.row_ss, np.roll(raw_row_db, 128 - r_pk_raw), label="Row (Range / Vertical)")
axes[0, 0].plot(np.arange(-128, 128) * h_raw.col_ss, np.roll(raw_col_db, 128 - c_pk_raw), label="Col (Azimuth / Horizontal)", linestyle="--")
axes[0, 0].set_title("UMBRA RAW: 1D IPR Cuts (dB)")
axes[0, 0].set_xlabel("Offset (meters)")
axes[0, 0].set_ylabel("Power (dB)")
axes[0, 0].set_ylim([-50, 5])
axes[0, 0].grid(True)
axes[0, 0].legend()

# DIFFPFA 1D Cuts
diff_row_cut = np.abs(chip_diff[:, c_pk_diff])
diff_col_cut = np.abs(chip_diff[r_pk_diff, :])
diff_row_db = 20 * np.log10(np.maximum(diff_row_cut / np.max(diff_row_cut), 1e-4))
diff_col_db = 20 * np.log10(np.maximum(diff_col_cut / np.max(diff_col_cut), 1e-4))

axes[0, 1].plot(np.arange(-128, 128) * h_diff.row_ss, np.roll(diff_row_db, 128 - r_pk_diff), label="Row (Range / Vertical)", color="red")
axes[0, 1].plot(np.arange(-128, 128) * h_diff.col_ss, np.roll(diff_col_db, 128 - c_pk_diff), label="Col (Azimuth / Horizontal)", linestyle="--", color="blue")
axes[0, 1].set_title("DiffPFA: 1D IPR Cuts (dB)")
axes[0, 1].set_xlabel("Offset (meters)")
axes[0, 1].set_ylabel("Power (dB)")
axes[0, 1].set_ylim([-50, 5])
axes[0, 1].grid(True)
axes[0, 1].legend()

# Overlay Row (Vertical) Cuts
axes[0, 2].plot(np.arange(-128, 128) * h_raw.row_ss, np.roll(raw_row_db, 128 - r_pk_raw), label="RAW Row IPR", color="black")
axes[0, 2].plot(np.arange(-128, 128) * h_diff.row_ss, np.roll(diff_row_db, 128 - r_pk_diff), label="DiffPFA Row IPR", color="red", linestyle="--")
axes[0, 2].set_title("Row (Vertical) IPR Comparison")
axes[0, 2].set_xlabel("Offset (meters)")
axes[0, 2].set_ylabel("Power (dB)")
axes[0, 2].set_ylim([-50, 5])
axes[0, 2].grid(True)
axes[0, 2].legend()

# 2D Spectrum Comparison
raw_spec_db = 20 * np.log10(np.maximum(np.abs(spec_raw) / np.max(np.abs(spec_raw)), 1e-4))
diff_spec_db = 20 * np.log10(np.maximum(np.abs(spec_diff) / np.max(np.abs(spec_diff)), 1e-4))

axes[1, 0].imshow(raw_spec_db, cmap="viridis", vmin=-40, vmax=0)
axes[1, 0].set_title("UMBRA RAW: 2D Spectrum (|K-Space| dB)")
axes[1, 0].set_xlabel("K_col (Azimuth)")
axes[1, 0].set_ylabel("K_row (Range)")

axes[1, 1].imshow(diff_spec_db, cmap="viridis", vmin=-40, vmax=0)
axes[1, 1].set_title("DiffPFA: 2D Spectrum (|K-Space| dB)")
axes[1, 1].set_xlabel("K_col (Azimuth)")
axes[1, 1].set_ylabel("K_row (Range)")

# 2D Image Chips
from clean_sar.utils import db_scale
axes[1, 2].imshow(db_scale(chip_diff, dyn_range_db=50), cmap="gray", vmin=-50, vmax=0)
axes[1, 2].set_title("DiffPFA: Spatial Domain Chip (dB)")
axes[1, 2].set_xlabel("Column")
axes[1, 2].set_ylabel("Row")

plt.tight_layout()
plt.savefig("/home/feildaw/CLEAN_SAR/scratch/ipr_investigation_20231114.png", dpi=150)
print("[+] Saved diagnostic plot to /home/feildaw/CLEAN_SAR/scratch/ipr_investigation_20231114.png")
