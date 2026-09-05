"""
audit/claude_code_opus_5/test_clean_recovery_reference.py

Drop-in replacement for tests/test_clean_algorithm.py::
test_exact_spatially_varying_clean_synthetic. Lift into tests/ roughly verbatim.

WHY THE EXISTING TEST CANNOT WORK
---------------------------------
It is named "synthetic" but reads a real SICD chip, and its only substantive
assertion is

    assert np.max(np.abs(res.residual_image)) < np.max(np.abs(chip))

i.e. the residual peak is below the input peak. EVERY subtractive procedure
satisfies that, including all of the deliberately broken variants measured in
the audit (delta PSF, bandwidth x2, sample spacing x10). It confirms the loop
ran, not that it deconvolved.

WHAT THIS FILE COVERS, AND WHAT IT DOES NOT
-------------------------------------------
The forward model here is the library's OWN analytic PSF, so these tests
validate the ALGORITHM -- peak selection, complex subtraction, flux
bookkeeping, restoration -- given a correct PSF. They deliberately do NOT
validate the PSF physics; that is test_psf_physics_reference.py's job, which
checks the PSF against independently computed ground truth.

Neither file alone is sufficient. Together they close the loop:
    PSF correct (vs. physics)  +  algorithm correct (vs. known truth)

audit/claude_code_opus_5/a16_recovery_bite.py proves these tests fail when the deconvolution
kernel is made to disagree with the true IPR.
"""
import numpy as np
import pytest

from clean_sar.algorithm import run_hogbom_clean
from clean_sar.backends import is_cuda_native_available
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator

# --- F7 workaround: remove once cuda_backend uses cuDevicePrimaryCtxRetain ---
import torch  # noqa: E402

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
# ---------------------------------------------------------------------------

N = 128
PSF_SIZE = 65
KH = PSF_SIZE // 2

# (row, col, complex amplitude); all at least KH from every edge
TARGETS = [
    (40, 45, 1.00 + 0.00j),
    (64, 64, 0.60 - 0.30j),
    (90, 85, 0.35 + 0.20j),
]

BACKENDS = ["pytorch"] + (["cuda"] if is_cuda_native_available() else [])


def _config(wgt="UNIFORM"):
    """Representative spaceborne X-band geometry, SCP at the scene centre."""
    return CleanPhysicsConfig(
        row_ss=0.4974, col_ss=0.6407,
        row_bw=1.5768, col_bw=1.1976,
        row_wid=0.886 / 1.5768, col_wid=0.886 / 1.1976,
        scp_slant_range=646_692.0,
        scp_row=N / 2.0, scp_col=N / 2.0,
        row_wgt=wgt, col_wgt=wgt,
    )


@pytest.fixture(scope="module")
def scene():
    """
    Noise-free scene: each target's peak pixel carries exactly its true complex
    amplitude, built by superposing the library's analytic PSF.
    """
    cfg = _config()
    gen = PSFGenerator(cfg)
    dirty = np.zeros((N, N), dtype=np.complex64)
    for (r, c, a) in TARGETS:
        dirty[r - KH:r + KH + 1, c - KH:c + KH + 1] += a * gen.compute_psf(
            r, c, psf_size=PSF_SIZE
        )
    return dirty, cfg


def _clean(dirty, cfg, backend):
    return run_hogbom_clean(
        dirty_image=dirty, config=cfg, backend=backend, beam_type="gaussian",
        psf_size=PSF_SIZE, gain=0.1, threshold=0.001, max_iters=6000,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 1. Are the true complex amplitudes recovered?
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_recovers_true_complex_amplitudes(scene, backend):
    dirty, cfg = scene
    comp = _clean(dirty, cfg, backend).components_map

    for (r, c, a) in TARGETS:
        got = comp[r - 2:r + 3, c - 2:c + 3].sum()   # allow 1-2 px of leakage
        err = abs(got - a) / abs(a)
        assert err < 0.02, (
            f"[{backend}] target ({r},{c}): recovered {got:.4f}, true {a:.4f}, "
            f"relative error {err:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. Does the flux land on the true targets, and only there?
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_flux_lands_on_true_targets(scene, backend):
    dirty, cfg = scene
    comp = _clean(dirty, cfg, backend).components_map

    total = np.abs(comp).sum()
    on_target = sum(
        abs(comp[r - 2:r + 3, c - 2:c + 3].sum()) for (r, c, _) in TARGETS
    )
    frac = on_target / total
    assert frac > 0.98, (
        f"[{backend}] only {100 * frac:.1f}% of CLEAN flux landed on true "
        f"target sites; the rest was deposited on spurious pixels"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_spurious_components(scene, backend):
    dirty, cfg = scene
    comp = _clean(dirty, cfg, backend).components_map

    n = int(np.count_nonzero(comp))
    assert n == len(TARGETS), (
        f"[{backend}] {n} pixels received components for {len(TARGETS)} true "
        f"targets; a correct run places one component pixel per scatterer"
    )


# ---------------------------------------------------------------------------
# 3. Is the residual actually driven down, not merely reduced?
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_is_driven_to_threshold(scene, backend):
    """
    The weak original assertion was `max|residual| < max|dirty|`, which any
    subtraction satisfies. On a noise-free synthetic scene a correct CLEAN
    should remove essentially all the energy.
    """
    dirty, cfg = scene
    res = _clean(dirty, cfg, backend)

    peak_ratio = np.max(np.abs(res.residual_image)) / np.max(np.abs(dirty))
    assert peak_ratio < 2e-3, (
        f"[{backend}] residual peak is {peak_ratio:.3e} of the dirty peak; "
        f"expected < 2e-3 on a noise-free synthetic scene"
    )

    energy_ratio = np.linalg.norm(res.residual_image) / np.linalg.norm(dirty)
    assert energy_ratio < 0.05, (
        f"[{backend}] residual retains {100 * energy_ratio:.1f}% of the input "
        f"energy"
    )


# ---------------------------------------------------------------------------
# 4. Backends must agree on the synthetic scene too
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not is_cuda_native_available(), reason="no native CUDA")
def test_backends_agree_on_synthetic_scene(scene):
    """
    Differences are normalised by the SCENE peak, not by each array's own peak.
    After convergence the residual is ~1e-3 of the scene, so dividing by the
    residual's own maximum inflates ordinary float32 rounding into an apparent
    1e-4 'disagreement'. The scene peak is the scale that matters physically.
    """
    dirty, cfg = scene
    scale = float(np.max(np.abs(dirty)))
    rt = _clean(dirty, cfg, "pytorch")
    rc = _clean(dirty, cfg, "cuda")
    for field in ("clean_image", "residual_image", "components_map", "restored_model"):
        a, b = getattr(rt, field), getattr(rc, field)
        err = float(np.max(np.abs(a - b))) / scale
        assert err < 1e-5, f"{field}: max|torch-cuda|/scene_peak = {err:.3e}"
