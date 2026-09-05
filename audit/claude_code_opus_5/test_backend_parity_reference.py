"""
audit/claude_code_opus_5/test_backend_parity_reference.py

Drop-in replacement for tests/test_backends.py::test_cuda_backend_execution_parity.
Lift into tests/ roughly verbatim.

WHY THE EXISTING TEST CANNOT WORK
---------------------------------
The current assertion is:

    assert abs(res_torch.suppression_db - res_cuda.suppression_db) < 1.0

`suppression_db` is computed host-side from `history_peaks`, i.e. from a
different data path than the arrays the backends actually return. The device
could return uninitialised memory and this value would be unchanged (that is
exactly the F8 scenario, where `delta_suppression_db` still reads 0.0000).

It is also run only at the default UNIFORM weighting, which is the one
configuration where the two backends genuinely agree.

PRINCIPLE
---------
A parity test must compare the PAYLOAD, not a summary derived from a different
code path. Anything computed on the host before, alongside, or independently of
the device result cannot witness a device fault.

EXPECTED RESULT ON THE CURRENT TREE
-----------------------------------
    UNIFORM  PASS       TAYLOR  PASS
    HAMMING  FAIL (F1)  HANN    FAIL (F1)
    mainlobe FAIL (F3)  mask    FAIL (F3)

Those failures are the point: they are the bugs. The file should go green once
F1 and F3 are fixed (or, for F3, once the CUDA path raises NotImplementedError
-- see the xfail-style note in test_cuda_rejects_unsupported_options).
"""
import glob

import numpy as np
import pytest

from clean_sar.algorithm import run_hogbom_clean
from clean_sar.backends import is_cuda_native_available
from clean_sar.config import CleanPhysicsConfig
from clean_sar.sicd_handler import SICDHandler

# ---------------------------------------------------------------------------
# F7 WORKAROUND -- REMOVE ONCE cuda_backend USES cuDevicePrimaryCtxRetain.
#
# _init_cuda_driver() calls cuCtxCreate_v2 when no CUDA context is current yet,
# creating a second, non-primary context; each CUDA run then cuCtxSetCurrent's
# to it and never restores. If the NVRTC probe runs BEFORE torch initialises
# CUDA, every pytorch-backend call after the first cuda-backend call dies with
#   RuntimeError: CUDA driver error: invalid resource handle
#
# Any module that probes for CUDA at import time and then alternates backends
# hits this -- which is precisely what a parity test must do. Touching
# torch.cuda first makes _init_cuda_driver adopt torch's primary context
# instead of creating its own, so the two coexist.
#
# Without these two lines this file reports 3 spurious failures that look like
# PSF divergence but are really the context bug.
# ---------------------------------------------------------------------------
import torch  # noqa: E402

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

pytestmark = pytest.mark.skipif(
    not is_cuda_native_available(),
    reason="native CUDA backend unavailable; parity is untestable",
)

ARRAYS = ("clean_image", "residual_image", "components_map", "restored_model")


def _sicd():
    for pat in ("/home/feildaw/data/*.nitf",
                "/home/feildaw/diffpfa/workspace/output/*.nitf"):
        files = sorted(glob.glob(pat))
        if files:
            return files[0]
    pytest.skip("no test SICD available")


@pytest.fixture(scope="module")
def chip_and_config():
    fp = _sicd()
    h = SICDHandler(fp)
    chip, _ = h.read_chip(2000, 4000, 2128, 4128)
    cfg = CleanPhysicsConfig.from_sicd_handler(h, chip_start=(2000, 4000))
    return chip, cfg


def _run(chip, cfg, backend, **kw):
    kwargs = dict(beam_type="gaussian", psf_size=65, gain=0.1,
                  threshold=0.02, max_iters=300, verbose=False)
    kwargs.update(kw)
    return run_hogbom_clean(dirty_image=chip, config=cfg, backend=backend, **kwargs)


def _with_weighting(cfg, wgt):
    d = dict(cfg.__dict__)
    d.update(row_wgt=wgt, col_wgt=wgt)
    return CleanPhysicsConfig(**d)


# --------------------------------------------------------------------------
# 1. Array-level parity across EVERY supported weighting, not just the default
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_backend_parity_arrays(chip_and_config, wgt):
    chip, base = chip_and_config
    cfg = _with_weighting(base, wgt)

    rt = _run(chip, cfg, "pytorch")
    rc = _run(chip, cfg, "cuda")

    assert rt.iterations == rc.iterations, (
        f"[{wgt}] iteration count differs: pytorch={rt.iterations} cuda={rc.iterations}"
    )

    # Peak selection must match step for step; a single divergence means the
    # backends are solving different problems from that point on.
    n = min(len(rt.history_coords), len(rc.history_coords))
    first_div = next((i for i in range(n)
                      if rt.history_coords[i] != rc.history_coords[i]), None)
    assert first_div is None, (
        f"[{wgt}] peak selection diverges at iteration {first_div}: "
        f"pytorch={rt.history_coords[first_div]} cuda={rc.history_coords[first_div]}"
    )

    # The payload itself.
    for field in ARRAYS:
        a, b = getattr(rt, field), getattr(rc, field)
        peak = float(np.max(np.abs(a))) or 1.0
        err = float(np.max(np.abs(a - b))) / peak
        assert err < 1e-5, f"[{wgt}] {field}: max|torch-cuda|/peak = {err:.3e}"


# --------------------------------------------------------------------------
# 2. PSF centre value must be 1.0 for every weighting (the root cause of F1)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wgt", ["UNIFORM", "TAYLOR", "HAMMING", "HANN"])
def test_psf_peak_is_unity(chip_and_config, wgt):
    """
    Hogbom subtracts gain*amp*PSF but books gain*amp as flux. Those agree only
    if PSF(0,0) == 1, so this is a photometric requirement, not a convention.
    """
    from clean_sar.psf import PSFGenerator

    _, base = chip_and_config
    gen = PSFGenerator(_with_weighting(base, wgt))
    psf = gen.compute_psf(64, 64, psf_size=65)
    assert np.isclose(np.abs(psf[32, 32]), 1.0, atol=1e-6), (
        f"[{wgt}] PSF centre = {psf[32, 32]!r}, expected 1.0"
    )


# --------------------------------------------------------------------------
# 3. Options must be honoured or refused -- never silently substituted (F3)
# --------------------------------------------------------------------------
def test_cuda_rejects_or_honours_mainlobe_beam(chip_and_config):
    chip, cfg = chip_and_config
    try:
        rg = _run(chip, cfg, "cuda", beam_type="gaussian")
        rm = _run(chip, cfg, "cuda", beam_type="mainlobe")
    except NotImplementedError:
        return  # refusing is an acceptable outcome

    d = float(np.max(np.abs(rg.restored_model - rm.restored_model)))
    peak = float(np.max(np.abs(rg.restored_model))) or 1.0
    assert d / peak > 1e-6, (
        "beam_type='mainlobe' produced a bit-identical result to 'gaussian': "
        "the option was silently ignored rather than honoured or refused"
    )


def test_cuda_rejects_or_honours_clean_mask(chip_and_config):
    chip, cfg = chip_and_config
    mask = np.zeros(chip.shape, dtype=bool)
    mask[:32, :32] = True
    try:
        r = _run(chip, cfg, "cuda", clean_mask=mask, max_iters=200)
    except NotImplementedError:
        return

    outside = int(np.count_nonzero(r.components_map)) - \
        int(np.count_nonzero(r.components_map[:32, :32]))
    assert outside == 0, (
        f"clean_mask was ignored: {outside} components placed outside the mask"
    )


def test_cuda_requires_physics_config(chip_and_config):
    """config=None must raise, not silently substitute placeholder physics (F11)."""
    chip, _ = chip_and_config
    with pytest.raises((ValueError, NotImplementedError)):
        run_hogbom_clean(dirty_image=chip, config=None, backend="cuda",
                         psf_size=65, gain=0.1, threshold=0.02,
                         max_iters=50, verbose=False)
