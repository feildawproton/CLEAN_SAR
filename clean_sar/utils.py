import numpy as np
from typing import Optional, Union


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
    image: np.ndarray,
    dyn_range_db: float = 50.0,
    eps: float = 1e-12,
    ref_val: Optional[float] = None
) -> np.ndarray:
    """
    Converts complex or intensity image to normalized decibels [ -dyn_range_db, 0 ].
    """
    mag = np.abs(image)
    peak = ref_val if (ref_val is not None and ref_val > 0) else np.max(mag)
    if peak <= 0:
        peak = 1.0
    mag_norm = np.maximum(mag / peak, eps)
    img_db = 20.0 * np.log10(mag_norm)
    return np.clip(img_db, -dyn_range_db, 0.0)
