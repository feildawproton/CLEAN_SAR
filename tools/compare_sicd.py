#!/usr/bin/env python3
"""
compare_sicd.py: Visualization and comparison tool for dirty vs. clean SAR SICDs.

Generates a 6-panel comparison plot (Dirty, Clean, Residual, Restored Model, 1D Cuts, and PSF).
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional

# Add parent directory to path if run as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clean_sar.sicd_handler import SICDHandler
from clean_sar.utils import db_scale


def plot_comparison(
    dirty_image: np.ndarray,
    clean_image: np.ndarray,
    residual_image: Optional[np.ndarray] = None,
    restored_model: Optional[np.ndarray] = None,
    output_png: str = "clean_comparison.png",
    dyn_range_db: float = 50.0,
    title_suffix: str = "",
    ref_val: Optional[float] = None,
) -> None:
    """
    Renders and saves a 6-panel comparison plot between dirty, clean, and residual SAR data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib is required for plotting. Please install via: pip install matplotlib")
        return

    # Determine reference power for 0 dB normalization
    ref = ref_val if (ref_val is not None and ref_val > 0) else np.max(np.abs(dirty_image))

    dirty_db = db_scale(dirty_image, dyn_range_db, ref_val=ref)
    clean_db = db_scale(clean_image, dyn_range_db, ref_val=ref)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    t_str = f" ({title_suffix})" if title_suffix else ""
    fig.suptitle(f"Complex Hogbom CLEAN SAR Deconvolution{t_str}", fontsize=16, fontweight="bold")

    # 1. Dirty Image
    im0 = axes[0, 0].imshow(dirty_db, cmap="gray", vmin=-dyn_range_db, vmax=0.0)
    axes[0, 0].set_title("Dirty Image (Input SICD)")
    axes[0, 0].set_xlabel("Col (Azimuth)")
    axes[0, 0].set_ylabel("Row (Range)")
    fig.colorbar(im0, ax=axes[0, 0], label="Power (dB)")

    # 2. Clean Image
    im1 = axes[0, 1].imshow(clean_db, cmap="gray", vmin=-dyn_range_db, vmax=0.0)
    axes[0, 1].set_title("Clean Image (Restored Model + Residual)")
    axes[0, 1].set_xlabel("Col (Azimuth)")
    axes[0, 1].set_ylabel("Row (Range)")
    fig.colorbar(im1, ax=axes[0, 1], label="Power (dB)")

    # 3. Residual Image
    if residual_image is not None:
        res_db = db_scale(residual_image, dyn_range_db, ref_val=ref)
        im2 = axes[0, 2].imshow(res_db, cmap="gray", vmin=-dyn_range_db, vmax=0.0)
        axes[0, 2].set_title("Residual Map")
        fig.colorbar(im2, ax=axes[0, 2], label="Power (dB)")
    else:
        axes[0, 2].axis("off")

    # 4. Restored Point Scatterers / Model
    if restored_model is not None:
        model_db = db_scale(restored_model, dyn_range_db, ref_val=ref)
        im3 = axes[1, 0].imshow(model_db, cmap="gray", vmin=-dyn_range_db, vmax=0.0)
        axes[1, 0].set_title("Restored Model (Point Components * Clean Beam)")
        fig.colorbar(im3, ax=axes[1, 0], label="Power (dB)")
    else:
        axes[1, 0].axis("off")

    # 5. 1D Range & Azimuth Cuts
    peak_idx = np.unravel_index(np.argmax(np.abs(dirty_image)), dirty_image.shape)
    r_peak, c_peak = peak_idx[0], peak_idx[1]

    d_row_cut = dirty_db[:, c_peak]
    c_row_cut = clean_db[:, c_peak]
    d_col_cut = dirty_db[r_peak, :]
    c_col_cut = clean_db[r_peak, :]

    axes[1, 1].plot(d_row_cut, label="Dirty (Range Cut)", color="crimson", alpha=0.7)
    axes[1, 1].plot(c_row_cut, label="Clean (Range Cut)", color="navy", linewidth=1.5)
    axes[1, 1].set_title(f"1D Range Cut across Peak (Col {c_peak})")
    axes[1, 1].set_xlabel("Row Index (pixels)")
    axes[1, 1].set_ylabel("Power (dB)")
    axes[1, 1].set_ylim([-dyn_range_db, 5.0])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper right")

    axes[1, 2].plot(d_col_cut, label="Dirty (Azimuth Cut)", color="crimson", alpha=0.7)
    axes[1, 2].plot(c_col_cut, label="Clean (Azimuth Cut)", color="navy", linewidth=1.5)
    axes[1, 2].set_title(f"1D Azimuth Cut across Peak (Row {r_peak})")
    axes[1, 2].set_xlabel("Col Index (pixels)")
    axes[1, 2].set_ylabel("Power (dB)")
    axes[1, 2].set_ylim([-dyn_range_db, 5.0])
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"[+] Saved comparison figure to: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Compare dirty and clean SICD NITFs with a 6-panel dB plot.")
    parser.add_argument("-d", "--dirty", required=True, help="Path to dirty / input NITF SICD.")
    parser.add_argument("-c", "--clean", required=True, help="Path to clean / deconvolved NITF SICD.")
    parser.add_argument("-o", "--output", default="comparison.png", help="Path to save comparison PNG.")
    parser.add_argument("--chip", default=None, help="Optional chip bounding box 'start_row,start_col,stop_row,stop_col'.")
    parser.add_argument("--dyn-range", type=float, default=50.0, help="Dynamic range in dB (default: 50.0).")
    parser.add_argument("--ref-val", type=float, default=None, help="Reference power magnitude for 0 dB normalization.")
    args = parser.parse_args()

    h_dirty = SICDHandler(args.dirty)
    h_clean = SICDHandler(args.clean)

    if args.chip:
        r0, c0, r1, c1 = [int(x.strip()) for x in args.chip.split(",")]
        dirty_img, _ = h_dirty.read_chip(r0, c0, r1, c1)
        clean_img, _ = h_clean.read_chip(r0, c0, r1, c1)
    else:
        dirty_img = h_dirty.read_full_image()
        clean_img = h_clean.read_full_image()

    plot_comparison(
        dirty_image=dirty_img,
        clean_image=clean_img,
        output_png=args.output,
        dyn_range_db=args.dyn_range,
        title_suffix=os.path.basename(args.clean),
        ref_val=args.ref_val,
    )


if __name__ == "__main__":
    main()
