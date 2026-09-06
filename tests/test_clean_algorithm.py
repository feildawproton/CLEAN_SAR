"""
tests/test_clean_algorithm.py: Validates CLEAN algorithm recovery of true complex amplitudes,
flux localization, and convergence against synthetic ground-truth targets across backends.
"""

import numpy as np
import pytest

from clean_sar.algorithm import run_hogbom_clean
from clean_sar.backends import is_cuda_native_available
from clean_sar.config import CleanPhysicsConfig
from clean_sar.psf import PSFGenerator, calculate_psf_size

N = 128

# (row, col, complex amplitude)
TARGETS = [
    (40, 45, 1.00 + 0.00j),
    (64, 64, 0.60 - 0.30j),
    (90, 85, 0.35 + 0.20j),
]

BACKENDS = ["c"] + (["cuda"] if is_cuda_native_available() else [])


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
    psf_sz = calculate_psf_size(cfg, (N, N))
    kh = psf_sz // 2

    dirty = np.zeros((N, N), dtype=np.complex64)
    for (r, c, a) in TARGETS:
        p = gen.compute_psf(r, c, psf_size=psf_sz)
        r0, r1 = max(0, r - kh), min(N, r + kh + 1)
        c0, c1 = max(0, c - kh), min(N, c + kh + 1)
        pr0 = r0 - (r - kh)
        pr1 = pr0 + (r1 - r0)
        pc0 = c0 - (c - kh)
        pc1 = pc0 + (c1 - c0)
        dirty[r0:r1, c0:c1] += a * p[pr0:pr1, pc0:pc1]
    return dirty, cfg


def _clean(dirty, cfg, backend):
    return run_hogbom_clean(
        dirty_image=dirty, config=cfg, backend=backend,
        gain=0.1, threshold=0.001, max_iters=6000,
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
    dirty, cfg = scene
    res = _clean(dirty, cfg, backend)

    peak_ratio = np.max(np.abs(res.residual_image)) / np.max(np.abs(dirty))
    assert peak_ratio < 2e-3, (
        f"[{backend}] residual peak is {peak_ratio:.3e} of the dirty peak; "
        f"expected < 2e-3 on a noise-free synthetic scene"
    )

    energy_ratio = np.linalg.norm(res.residual_image) / np.linalg.norm(dirty)
    assert energy_ratio < 0.05, (
        f"[{backend}] residual retains {100 * energy_ratio:.1f}% of the input energy"
    )


# ---------------------------------------------------------------------------
# 4. Backends must agree on the synthetic scene too
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not is_cuda_native_available(), reason="no native CUDA")
def test_backends_agree_on_synthetic_scene(scene):
    dirty, cfg = scene
    scale = float(np.max(np.abs(dirty)))
    r_c = _clean(dirty, cfg, "c")
    rc = _clean(dirty, cfg, "cuda")
    for field in ("clean_image", "residual_image", "components_map", "restored_model"):
        a, b = getattr(r_c, field), getattr(rc, field)
        err = float(np.max(np.abs(a - b))) / scale
        assert err < 1e-5, f"{field}: max|c-cuda|/scene_peak = {err:.3e}"
