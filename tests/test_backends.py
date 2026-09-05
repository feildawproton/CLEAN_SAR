import glob
import numpy as np
import pytest

from clean_sar.backends import resolve_backend, is_cuda_native_available
from clean_sar.algorithm import run_hogbom_clean
from clean_sar.processor import CLEANProcessor
from clean_sar.config import CleanPhysicsConfig
from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator


@pytest.fixture(scope="module")
def chip_and_config(test_sicd_path):
    h = SICDHandler(test_sicd_path)
    chip, _ = h.read_chip(2000, 4000, 2128, 4128)
    cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
    return chip, cfg


def _with_weighting(cfg, wgt):
    d = dict(cfg.__dict__)
    d.update(row_wgt=wgt, col_wgt=wgt)
    return CleanPhysicsConfig(**d)


def _run(chip, cfg, backend, **kw):
    kwargs = dict(beam_type="gaussian", psf_size=65, gain=0.1,
                  threshold=0.02, max_iters=200, verbose=False)
    kwargs.update(kw)
    return run_hogbom_clean(dirty_image=chip, config=cfg, backend=backend, **kwargs)


def test_resolve_backend_logic():
    if is_cuda_native_available():
        assert resolve_backend("auto") == "cuda"
    else:
        assert resolve_backend("auto") == "pytorch"
    assert resolve_backend("pytorch") == "pytorch"
    assert resolve_backend("cuda") == "cuda"

    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend("opencl")


# --------------------------------------------------------------------------
# 1. Array-level parity across EVERY supported weighting
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_backend_parity_arrays(chip_and_config, wgt):
    if not is_cuda_native_available():
        pytest.skip("CUDA native driver not available")
    chip, base = chip_and_config
    cfg = _with_weighting(base, wgt)

    rt = _run(chip, cfg, "pytorch")
    rc = _run(chip, cfg, "cuda")

    assert rt.iterations == rc.iterations, (
        f"[{wgt}] iteration count differs: pytorch={rt.iterations} cuda={rc.iterations}"
    )

    # Peak selection must match step for step
    n = min(len(rt.history_coords), len(rc.history_coords))
    first_div = next((i for i in range(n) if rt.history_coords[i] != rc.history_coords[i]), None)
    assert first_div is None, (
        f"[{wgt}] peak selection diverges at iteration {first_div}: "
        f"pytorch={rt.history_coords[first_div]} cuda={rc.history_coords[first_div]}"
    )

    # Check all output arrays to high precision
    for field in ("clean_image", "residual_image", "components_map", "restored_model"):
        a, b = getattr(rt, field), getattr(rc, field)
        peak = float(np.max(np.abs(a))) or 1.0
        err = float(np.max(np.abs(a - b))) / peak
        assert err < 1e-5, f"[{wgt}] {field}: max|torch-cuda|/peak = {err:.3e}"


# --------------------------------------------------------------------------
# 2. PSF centre value must be 1.0 for every weighting
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_psf_peak_is_unity(chip_and_config, wgt):
    _, base = chip_and_config
    gen = PSFGenerator(_with_weighting(base, wgt))
    psf = gen.compute_psf(64, 64, psf_size=65)
    assert np.isclose(np.abs(psf[32, 32]), 1.0, atol=1e-6), (
        f"[{wgt}] PSF centre = {psf[32, 32]!r}, expected 1.0"
    )


# --------------------------------------------------------------------------
# 3. Options must be honoured or refused -- never silently substituted
# --------------------------------------------------------------------------
def test_cuda_rejects_mainlobe_beam(chip_and_config):
    if not is_cuda_native_available():
        pytest.skip("CUDA native driver not available")
    chip, cfg = chip_and_config
    with pytest.raises(NotImplementedError, match="beam_type='gaussian'"):
        _run(chip, cfg, "cuda", beam_type="mainlobe")


def test_cuda_rejects_clean_mask(chip_and_config):
    if not is_cuda_native_available():
        pytest.skip("CUDA native driver not available")
    chip, cfg = chip_and_config
    mask = np.zeros(chip.shape, dtype=bool)
    mask[:32, :32] = True
    with pytest.raises(NotImplementedError, match="clean_mask"):
        _run(chip, cfg, "cuda", clean_mask=mask)


def test_cuda_requires_physics_config(chip_and_config):
    if not is_cuda_native_available():
        pytest.skip("CUDA native driver not available")
    chip, _ = chip_and_config
    with pytest.raises(ValueError, match="CleanPhysicsConfig 'config' must be provided"):
        run_hogbom_clean(dirty_image=chip, config=None, backend="cuda",
                         psf_size=65, gain=0.1, threshold=0.02,
                         max_iters=50, verbose=False)


def test_processor_with_cuda_backend(tmp_path, test_sicd_path):
    if not is_cuda_native_available():
        pytest.skip("CUDA native driver not available")
    out_file = str(tmp_path / "test_cuda_proc.nitf")

    proc = CLEANProcessor(
        input_path=test_sicd_path,
        output_path=out_file,
        chip_bounds=(100, 100, 228, 228),
        backend="cuda",
    )

    res = proc.run(
        gain=0.1,
        threshold=0.05,
        max_iters=100,
        verbose=False,
    )

    assert res.iterations > 0
    assert res.suppression_db > 0.0
