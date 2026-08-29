import numpy as np
import torch
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt


def taylor_window_1d(length: int, nbar: int = 4, sll: float = -30.0) -> np.ndarray:
    """
    Computes a 1D Taylor weighting window.

    Parameters
    ----------
    length : int
        Number of output window samples.
    nbar : int
        Number of nearly constant-level sidelobes adjacent to mainlobe.
    sll : float
        Peak sidelobe level in dB (negative float, e.g. -30.0).

    Returns
    -------
    np.ndarray
        1D Taylor window of shape (length,) normalized to peak 1.0.
    """
    if length <= 1:
        return np.ones(length, dtype=np.float64)

    # Convert SLL dB to voltage ratio
    eta = 10.0 ** (-sll / 20.0)
    a = np.arccosh(eta) / np.pi
    sigma2 = (nbar ** 2) / (a ** 2 + (nbar - 0.5) ** 2)

    # Compute Fm coefficients
    m = np.arange(1, nbar)
    fm = np.zeros(nbar - 1, dtype=np.float64)

    for i in range(len(m)):
        mi = m[i]
        num = 1.0
        den = 1.0
        for n in range(1, nbar):
            num *= 1.0 - (mi ** 2) / (sigma2 * (a ** 2 + (n - 0.5) ** 2))
            if n != mi:
                den *= 1.0 - (mi ** 2) / (n ** 2)
        fm[i] = ((-1.0) ** (mi + 1) * num) / (2.0 * den)

    # Generate spatial window
    indices = np.linspace(-0.5, 0.5, length, endpoint=True)
    w = np.ones(length, dtype=np.float64)

    for i in range(len(m)):
        w += 2.0 * fm[i] * np.cos(2.0 * np.pi * m[i] * indices)

    # Normalize peak to 1.0
    max_val = np.max(w)
    if max_val > 0:
        w /= max_val
    return w


def get_1d_window(name: Optional[str], length: int, **kwargs) -> np.ndarray:
    """
    Returns a 1D window by name (UNIFORM, HAMMING, HANN, TAYLOR, KAISER).
    """
    if name is None or name.upper() in ["UNIFORM", "RECT", "RECTANGULAR", "NONE"]:
        return np.ones(length, dtype=np.float64)

    name_upper = name.upper()
    if name_upper == "HAMMING":
        return np.hamming(length)
    elif name_upper in ["HANN", "HANNING"]:
        return np.hanning(length)
    elif name_upper == "TAYLOR":
        nbar = kwargs.get("nbar", 4)
        sll = kwargs.get("sll", -30.0)
        return taylor_window_1d(length, nbar=nbar, sll=sll)
    elif name_upper == "KAISER":
        beta = kwargs.get("beta", 6.0)
        return np.kaiser(length, beta)
    elif name_upper == "BLACKMAN":
        return np.blackman(length)
    else:
        # Default fallback to uniform
        return np.ones(length, dtype=np.float64)


def get_2d_window(
    row_name: Optional[str],
    col_name: Optional[str],
    num_rows: int,
    num_cols: int,
    **kwargs
) -> np.ndarray:
    """
    Computes a separable 2D window W(row, col) = W_row(row) * W_col(col).
    """
    w_row = get_1d_window(row_name, num_rows, **kwargs)
    w_col = get_1d_window(col_name, num_cols, **kwargs)
    return np.outer(w_row, w_col)


def db_scale(
    image: Union[np.ndarray, torch.Tensor],
    dyn_range_db: float = 50.0,
    ref_val: Optional[float] = None
) -> np.ndarray:
    """
    Converts complex or magnitude image to normalized dB scale [ -dyn_range_db, 0 ].

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Complex or magnitude image.
    dyn_range_db : float
        Dynamic range in dB (e.g. 50.0).
    ref_val : float, optional
        Reference peak magnitude for 0 dB normalization. If None, max(|image|) is used.

    Returns
    -------
    np.ndarray
        dB scaled image clipped to [-dyn_range_db, 0].
    """
    if isinstance(image, torch.Tensor):
        mag = torch.abs(image).detach().cpu().numpy()
    else:
        mag = np.abs(image)

    if ref_val is None:
        ref_val = np.max(mag)

    if ref_val <= 0:
        return np.full_like(mag, -dyn_range_db, dtype=np.float64)

    # 20 * log10(mag / ref_val)
    eps = 1e-12
    db = 20.0 * np.log10(np.maximum(mag / ref_val, eps))
    return np.clip(db, -dyn_range_db, 0.0)


def plot_clean_comparison(
    dirty_image: np.ndarray,
    clean_image: np.ndarray,
    residual_image: np.ndarray,
    components_map: np.ndarray,
    output_path: str,
    title: str = "Complex SAR Hogbom CLEAN Deconvolution",
    dyn_range_db: float = 50.0,
    psf_dirty: Optional[np.ndarray] = None,
    psf_clean: Optional[np.ndarray] = None,
):
    """
    Generates and saves a multi-panel comparison figure showing:
    1. Dirty SAR Image (dB)
    2. Restored Clean SAR Image (dB)
    3. Clean Point Components (dB or magnitude)
    4. Residual Image (dB)
    5. (Optional) Dirty and Clean PSF profiles/insets.
    """
    ref_val = np.max(np.abs(dirty_image))

    dirty_db = db_scale(dirty_image, dyn_range_db=dyn_range_db, ref_val=ref_val)
    clean_db = db_scale(clean_image, dyn_range_db=dyn_range_db, ref_val=ref_val)
    resid_db = db_scale(residual_image, dyn_range_db=dyn_range_db, ref_val=ref_val)

    comp_mag = np.abs(components_map)
    comp_db = db_scale(comp_mag, dyn_range_db=dyn_range_db, ref_val=ref_val)

    has_psf = psf_dirty is not None and psf_clean is not None
    num_cols = 3 if has_psf else 2
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))

    # Top-Left: Dirty Image
    im0 = axes[0, 0].imshow(dirty_db, cmap="gray", vmin=-dyn_range_db, vmax=0)
    axes[0, 0].set_title(f"Original Dirty Image ({dyn_range_db} dB)")
    axes[0, 0].set_xlabel("Column (Azimuth)")
    axes[0, 0].set_ylabel("Row (Range)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="dB")

    # Top-Center/Right: Restored Clean Image
    im1 = axes[0, 1].imshow(clean_db, cmap="gray", vmin=-dyn_range_db, vmax=0)
    axes[0, 1].set_title(f"Restored Clean Image ({dyn_range_db} dB)")
    axes[0, 1].set_xlabel("Column (Azimuth)")
    axes[0, 1].set_ylabel("Row (Range)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="dB")

    # Bottom-Left: Clean Point Components
    im2 = axes[1, 0].imshow(comp_db, cmap="inferno", vmin=-dyn_range_db, vmax=0)
    axes[1, 0].set_title(f"Clean Point Components ({np.count_nonzero(comp_mag)} pts)")
    axes[1, 0].set_xlabel("Column (Azimuth)")
    axes[1, 0].set_ylabel("Row (Range)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="dB")

    # Bottom-Center/Right: Residual Image
    im3 = axes[1, 1].imshow(resid_db, cmap="gray", vmin=-dyn_range_db, vmax=0)
    axes[1, 1].set_title(f"Residual Image (Max: {np.max(np.abs(residual_image))/ref_val:.3f}x)")
    axes[1, 1].set_xlabel("Column (Azimuth)")
    axes[1, 1].set_ylabel("Row (Range)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="dB")

    if has_psf:
        psf_d_db = db_scale(psf_dirty, dyn_range_db=dyn_range_db)
        psf_c_db = db_scale(psf_clean, dyn_range_db=dyn_range_db)

        im4 = axes[0, 2].imshow(psf_d_db, cmap="viridis", vmin=-dyn_range_db, vmax=0)
        axes[0, 2].set_title("Dirty PSF (Beam)")
        fig.colorbar(im4, ax=axes[0, 2], fraction=0.046, pad=0.04, label="dB")

        im5 = axes[1, 2].imshow(psf_c_db, cmap="viridis", vmin=-dyn_range_db, vmax=0)
        axes[1, 2].set_title("Clean Restoring Beam")
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04, label="dB")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
