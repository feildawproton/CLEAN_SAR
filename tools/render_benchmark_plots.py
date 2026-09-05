import os
import sys
import argparse
from glob import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clean_sar.sicd_handler import SICDHandler
from clean_sar.utils import db_scale


def render_full_and_zoom_comparison(
    dirty_path: str,
    clean_path: str,
    output_png: str,
    zoom_size: int = 256,
    dyn_range_db: float = 50.0,
):
    """
    Renders a multi-panel visual comparison showing full scene overview (fast decimated preview),
    zoomed-in peak ROI at 100% native full resolution, and 1D profile cuts through the dominant point target.
    """
    handler_dirty = SICDHandler(dirty_path)
    handler_clean = SICDHandler(clean_path)

    dirty_img = handler_dirty.read_full_image()
    clean_img = handler_clean.read_full_image()

    ref = np.max(np.abs(dirty_img))
    H, W = dirty_img.shape

    # Find dominant peak coordinate in full scene
    peak_idx = np.unravel_index(np.argmax(np.abs(dirty_img)), dirty_img.shape)
    r_peak, c_peak = int(peak_idx[0]), int(peak_idx[1])

    kh = zoom_size // 2
    r_min = max(0, r_peak - kh)
    r_max = min(H, r_peak + kh)
    c_min = max(0, c_peak - kh)
    c_max = min(W, c_peak + kh)

    # 100% native resolution ROI zoom
    dirty_zoom = db_scale(dirty_img[r_min:r_max, c_min:c_max], dyn_range_db, ref_val=ref)
    clean_zoom = db_scale(clean_img[r_min:r_max, c_min:c_max], dyn_range_db, ref_val=ref)

    # Decimate full scenes for fast, crisp thumbnail rendering
    step = max(1, max(H, W) // 1200)
    dirty_sub = db_scale(dirty_img[::step, ::step], dyn_range_db, ref_val=ref)
    clean_sub = db_scale(clean_img[::step, ::step], dyn_range_db, ref_val=ref)
    diff_sub = db_scale((dirty_img - clean_img)[::step, ::step], dyn_range_db, ref_val=ref)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    base_title = os.path.basename(dirty_path)
    fig.suptitle(f"Full-Scene CLEAN Deconvolution: {base_title}\nPeak at [{r_peak}, {c_peak}] | Dimensions: {H}x{W}", 
                 fontsize=15, fontweight="bold")

    # 1. Full Scene Dirty Overview
    im0 = axes[0, 0].imshow(dirty_sub, cmap="gray", vmin=-dyn_range_db, vmax=0.0, aspect="auto")
    axes[0, 0].set_title(f"Full Scene Dirty ({H}x{W}) [Decimated 1:{step}]")
    axes[0, 0].scatter([c_peak // step], [r_peak // step], color="red", marker="+", s=100, label="Dominant Peak")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.colorbar(im0, ax=axes[0, 0], label="dB")

    # 2. Full Scene Clean Overview
    im1 = axes[0, 1].imshow(clean_sub, cmap="gray", vmin=-dyn_range_db, vmax=0.0, aspect="auto")
    axes[0, 1].set_title(f"Full Scene Clean (Restored) [Decimated 1:{step}]")
    fig.colorbar(im1, ax=axes[0, 1], label="dB")

    # 3. Full Scene Difference
    im2 = axes[0, 2].imshow(diff_sub, cmap="gray", vmin=-dyn_range_db, vmax=0.0, aspect="auto")
    axes[0, 2].set_title("Extracted Sidelobes & Point Model")
    fig.colorbar(im2, ax=axes[0, 2], label="dB")

    # 4. Zoomed Dirty ROI (Native Resolution)
    im3 = axes[1, 0].imshow(dirty_zoom, cmap="inferno", vmin=-dyn_range_db, vmax=0.0)
    axes[1, 0].set_title(f"Zoomed Dirty Peak ({zoom_size}x{zoom_size} @ 100% Native Res)")
    fig.colorbar(im3, ax=axes[1, 0], label="dB")

    # 5. Zoomed Clean ROI (Native Resolution)
    im4 = axes[1, 1].imshow(clean_zoom, cmap="inferno", vmin=-dyn_range_db, vmax=0.0)
    axes[1, 1].set_title(f"Zoomed Clean Peak ({zoom_size}x{zoom_size} @ 100% Native Res)")
    fig.colorbar(im4, ax=axes[1, 1], label="dB")

    # 6. 1D Profile Cuts across Peak
    rel_r_peak = r_peak - r_min
    rel_c_peak = c_peak - c_min

    d_row_cut = dirty_zoom[:, rel_c_peak]
    c_row_cut = clean_zoom[:, rel_c_peak]
    d_col_cut = dirty_zoom[rel_r_peak, :]
    c_col_cut = clean_zoom[rel_r_peak, :]

    axes[1, 2].plot(d_row_cut, label="Dirty Range Cut", color="crimson", alpha=0.8, linestyle="--")
    axes[1, 2].plot(c_row_cut, label="Clean Range Cut", color="navy", linewidth=1.5)
    axes[1, 2].plot(d_col_cut, label="Dirty Azimuth Cut", color="orange", alpha=0.8, linestyle=":")
    axes[1, 2].plot(c_col_cut, label="Clean Azimuth Cut", color="forestgreen", linewidth=1.5)
    axes[1, 2].set_title("1D Profile Cuts across Peak")
    axes[1, 2].set_xlabel("Relative Pixels")
    axes[1, 2].set_ylabel("Power (dB)")
    axes[1, 2].set_ylim([-dyn_range_db, 5.0])
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison plot: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Render comparison plots for benchmarked datasets.")
    parser.add_argument("--dirty_dir", required=True, help="Directory with original dirty NITF files.")
    parser.add_argument("--clean_dir", required=True, help="Directory with clean NITF files.")
    parser.add_argument("--output_dir", default="output/plots", help="Directory to save PNG comparison plots.")
    args = parser.parse_args()

    clean_files = sorted(glob(os.path.join(args.clean_dir, "*_clean.nitf")))
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found {len(clean_files)} clean SICD files to render...")
    for clean_path in clean_files:
        base = os.path.basename(clean_path).replace("_clean.nitf", ".nitf")
        dirty_path = os.path.join(args.dirty_dir, base)
        if not os.path.exists(dirty_path):
            print(f"[!] Dirty file not found: {dirty_path}")
            continue

        out_png = os.path.join(args.output_dir, f"{os.path.splitext(base)[0]}_comparison.png")
        render_full_and_zoom_comparison(dirty_path, clean_path, out_png)


if __name__ == "__main__":
    main()
