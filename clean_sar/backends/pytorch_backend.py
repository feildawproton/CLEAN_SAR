import time
from typing import Optional, Tuple, List, Union, Literal
import numpy as np
import torch

from ..config import CleanPhysicsConfig
from ..psf import PSFGenerator


def run_hogbom_pytorch(
    dirty_image: Union[np.ndarray, torch.Tensor],
    config: Optional[CleanPhysicsConfig] = None,
    psf_generator: Optional[PSFGenerator] = None,
    beam_type: Literal["gaussian", "mainlobe"] = "gaussian",
    psf_size: int = 65,
    gain: float = 0.1,
    threshold: float = 0.02,
    max_iters: int = 2500,
    clean_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    guard_margin: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
):
    """
    Executes Complex Hogbom CLEAN deconvolution using the PyTorch GPU/CPU backend.
    """
    from ..algorithm import CleanResult

    t_pipeline_start = time.perf_counter()

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    # Initialize PSF generator from config if not explicitly provided
    if psf_generator is None:
        if config is None:
            raise ValueError("Either 'config' (CleanPhysicsConfig) or 'psf_generator' must be provided.")
        psf_gen = PSFGenerator(config)
    else:
        psf_gen = psf_generator

    # 1. Host to Device Transfer (H2D)
    t0_h2d = time.perf_counter()
    if isinstance(dirty_image, torch.Tensor):
        img_t = dirty_image.to(device=target_device, dtype=torch.complex64)
    else:
        native_arr = np.require(dirty_image, dtype=np.complex64, requirements=["C", "A"])
        img_t = torch.as_tensor(native_arr, dtype=torch.complex64, device=target_device)
    if target_device.type == "cuda":
        torch.cuda.synchronize()
    t_h2d = time.perf_counter() - t0_h2d

    H, W = img_t.shape

    if clean_mask is not None:
        if isinstance(clean_mask, np.ndarray):
            mask_native = np.require(clean_mask, dtype=bool, requirements=["C", "A"])
            mask_t = torch.as_tensor(mask_native, dtype=torch.bool, device=target_device)
        else:
            mask_t = clean_mask.to(device=target_device, dtype=torch.bool)
    else:
        mask_t = None

    if guard_margin > 0 and guard_margin < min(H // 2, W // 2):
        guard_mask = torch.zeros((H, W), dtype=torch.bool, device=target_device)
        guard_mask[guard_margin : H - guard_margin, guard_margin : W - guard_margin] = True
        mask_t = (mask_t & guard_mask) if mask_t is not None else guard_mask

    if psf_size % 2 == 0:
        psf_size += 1
    kh = psf_size // 2
    kw = psf_size // 2

    # 2. Pure GPU Compute Loop
    t0_compute = time.perf_counter()
    residual = img_t.clone()
    components = torch.zeros_like(residual)
    restored_model = torch.zeros_like(residual)

    init_mag = torch.max(torch.abs(residual)).item()
    stop_thresh = (threshold * init_mag) if threshold < 1.0 else threshold

    history_peaks: List[float] = []
    history_coords: List[Tuple[int, int]] = []

    if verbose:
        chip_str = f"origin ({psf_gen.config.chip_start_row}, {psf_gen.config.chip_start_col})"
        print(f"[CLEAN] Initial peak: {init_mag:.4e}, Stopping threshold: {stop_thresh:.4e}, Max iters: {max_iters} ({chip_str})")
        print(f"        Backend: PYTORCH | Beam: {beam_type.upper()} | PSF size: {psf_size}x{psf_size}")

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
        dirty_psf, clean_beam = psf_gen.get_psfs_torch(
            row=r0,
            col=c0,
            psf_size=psf_size,
            beam_type=beam_type,
            device=target_device,
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

    clean_image_t = restored_model + residual
    if target_device.type == "cuda":
        torch.cuda.synchronize()
    t_compute = time.perf_counter() - t0_compute

    # 3. Device to Host Transfer (D2H)
    t0_d2h = time.perf_counter()
    clean_out = clean_image_t.detach().cpu().numpy()
    components_out = components.detach().cpu().numpy()
    residual_out = residual.detach().cpu().numpy()
    restored_model_out = restored_model.detach().cpu().numpy()
    t_d2h = time.perf_counter() - t0_d2h

    t_total = time.perf_counter() - t_pipeline_start

    res = CleanResult(
        clean_image=clean_out,
        components_map=components_out,
        residual_image=residual_out,
        restored_model=restored_model_out,
        iterations=it,
        execution_time_sec=t_total,
        history_peaks=history_peaks,
        history_coords=history_coords,
    )
    res.h2d_time_sec = t_h2d
    res.pure_compute_time_sec = t_compute
    res.d2h_time_sec = t_d2h
    return res
