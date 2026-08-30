import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Literal
import numpy as np
import torch


@dataclass
class CleanResult:
    """Dataclass holding output products from Complex Hogbom CLEAN."""
    clean_image: np.ndarray
    components_map: np.ndarray
    residual_image: np.ndarray
    restored_model: np.ndarray
    iterations: int
    execution_time_sec: float
    history_peaks: List[float] = field(default_factory=list)
    history_coords: List[Tuple[int, int]] = field(default_factory=list)


def _to_native_complex64(arr: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Converts numpy array or torch tensor to complex64 on device in native byte order."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=torch.complex64)
    native_arr = np.require(arr, dtype=np.complex64, requirements=["C", "A"])
    return torch.as_tensor(native_arr, dtype=torch.complex64, device=device)


def run_hogbom_clean(
    dirty_image: Union[np.ndarray, torch.Tensor],
    psf_generator,
    method: Literal["kspace", "analytic"] = "kspace",
    beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    chip_origin: Optional[Tuple[int, int]] = None,
    clean_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    guard_margin: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> CleanResult:
    """
    Executes Complex Hogbom CLEAN deconvolution with exact spatially-varying PSF computation
    for each global peak position on every iteration.

    Parameters
    ----------
    dirty_image : np.ndarray or torch.Tensor
        Complex 2D dirty SAR image (H, W).
    psf_generator : PSFGenerator
        Instance of PSFGenerator used to compute the exact local dirty PSF and clean beam.
    method : str
        'kspace' (Option A) or 'analytic' (Option B).
    beam_type : str
        'gaussian' (matched 3dB width) or 'mainlobe'.
    psf_size : int
        PSF kernel dimension (must be odd, e.g. 65).
    gain : float
        Loop damping factor (gamma), typically 0.05 - 0.2 (default 0.1).
    threshold : float
        Stopping threshold (fraction of initial peak if < 1.0, or absolute magnitude).
    max_iters : int
        Maximum number of CLEAN iterations.
    chip_origin : tuple of (start_row, start_col), optional
        Global coordinates of the top-left corner of dirty_image.
    clean_mask : np.ndarray or torch.Tensor, optional
        Binary mask (H, W) constraining search region for point components.
    guard_margin : int, optional
        Border margin in pixels excluded from peak selection (default: 0).
    device : str or torch.device, optional
        Target device ('cuda' or 'cpu'). If None, GPU is used if available.
    verbose : bool
        If True, prints progress periodically.

    Returns
    -------
    CleanResult
        Object containing clean image, components map, residual, and iteration history.
    """
    t_start = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Convert image to torch tensor on device in native byte order
    img_t = _to_native_complex64(dirty_image, device)
    H, W = img_t.shape

    if clean_mask is not None:
        if isinstance(clean_mask, np.ndarray):
            mask_native = np.require(clean_mask, dtype=bool, requirements=["C", "A"])
            mask_t = torch.as_tensor(mask_native, dtype=torch.bool, device=device)
        else:
            mask_t = clean_mask.to(device=device, dtype=torch.bool)
    else:
        mask_t = None

    if guard_margin > 0 and guard_margin < min(H // 2, W // 2):
        guard_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        guard_mask[guard_margin : H - guard_margin, guard_margin : W - guard_margin] = True
        mask_t = (mask_t & guard_mask) if mask_t is not None else guard_mask

    H, W = img_t.shape
    if psf_size % 2 == 0:
        psf_size += 1
    kh = psf_size // 2
    kw = psf_size // 2

    residual = img_t.clone()
    components = torch.zeros_like(residual)
    restored_model = torch.zeros_like(residual)

    init_mag = torch.max(torch.abs(residual)).item()
    stop_thresh = (threshold * init_mag) if threshold < 1.0 else threshold

    history_peaks: List[float] = []
    history_coords: List[Tuple[int, int]] = []

    if verbose:
        origin_str = f"chip origin {chip_origin}" if chip_origin else "global coordinates"
        print(f"[CLEAN] Initial peak: {init_mag:.4e}, Stopping threshold: {stop_thresh:.4e}, Max iters: {max_iters} ({origin_str})")
        print(f"        Method: {method.upper()} | Beam: {beam_type.upper()} | PSF size: {psf_size}x{psf_size}")

    it = 0
    while it < max_iters:
        mag = torch.abs(residual)
        if mask_t is not None:
            mag = mag * mask_t

        max_val, flat_idx = torch.max(mag.view(-1), 0)
        curr_peak = max_val.item()
        history_peaks.append(curr_peak)

        if curr_peak <= stop_thresh or curr_peak == 0.0:
            if verbose:
                print(f"[CLEAN] Converged at iteration {it}: peak {curr_peak:.4e} <= {stop_thresh:.4e}")
            break

        r0 = (flat_idx // W).item()
        c0 = (flat_idx % W).item()
        history_coords.append((r0, c0))

        # Peak complex amplitude and subtracted component
        amp = residual[r0, c0]
        comp = gain * amp
        components[r0, c0] += comp

        # Retrieve exact dirty PSF and clean beam for this exact global peak position
        dirty_psf, clean_beam = psf_generator.get_psfs_torch(
            row=r0,
            col=c0,
            psf_size=psf_size,
            method=method,
            beam_type=beam_type,
            chip_origin=chip_origin,
            device=device,
        )

        # Window slice bounds
        r_min = max(0, r0 - kh)
        r_max = min(H, r0 + kh + 1)
        c_min = max(0, c0 - kw)
        c_max = min(W, c0 + kw + 1)

        pr_min = kh - (r0 - r_min)
        pr_max = kh + (r_max - r0)
        pc_min = kw - (c0 - c_min)
        pc_max = kw + (c_max - c0)

        # Subtract exact local dirty PSF from residual
        residual[r_min:r_max, c_min:c_max] -= comp * dirty_psf[pr_min:pr_max, pc_min:pc_max]

        # Add exact local clean restoring beam to restored model
        restored_model[r_min:r_max, c_min:c_max] += comp * clean_beam[pr_min:pr_max, pc_min:pc_max]

        it += 1
        if verbose and it % 250 == 0:
            print(f"  Iter {it:5d}: Current max residual = {curr_peak:.4e} (Peak at [{r0}, {c0}])")

    clean_image = restored_model + residual
    t_elapsed = time.time() - t_start

    return CleanResult(
        clean_image=clean_image.detach().cpu().numpy(),
        components_map=components.detach().cpu().numpy(),
        residual_image=residual.detach().cpu().numpy(),
        restored_model=restored_model.detach().cpu().numpy(),
        iterations=it,
        execution_time_sec=t_elapsed,
        history_peaks=history_peaks,
        history_coords=history_coords,
    )
