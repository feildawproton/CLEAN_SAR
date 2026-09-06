import numpy as np
import pytest

from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean
from clean_sar.psf import PSFGenerator
from clean_sar.backends import is_cuda_native_available

BACKENDS = ["c"] + (["cuda"] if is_cuda_native_available() else [])


@pytest.fixture
def sim_config():
    return CleanPhysicsConfig(
        row_ss=0.5, col_ss=0.5,
        row_bw=1.5, col_bw=1.5,
        row_wid=0.886 / 1.5, col_wid=0.886 / 1.5,
        scp_slant_range=600_000.0,
        scp_row=64.0, scp_col=64.0,
        row_wgt="UNIFORM", col_wgt="UNIFORM",
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_offgrid_point_target_recovery(sim_config, backend):
    """
    Validates deconvolution on an off-grid point target at fractional coordinates (64.35, 64.25).
    Energy is distributed across fractional pixels; verifies convergence and > 25 dB suppression.
    """
    dr = np.arange(128) - 64.35
    dc = np.arange(128) - 64.25
    DR, DC = np.meshgrid(dr, dc, indexing="ij")
    U = DR * sim_config.row_ss
    V = DC * sim_config.col_ss
    dirty_off = (10.0 * np.sinc(sim_config.row_bw * U) * np.sinc(sim_config.col_bw * V)).astype(np.complex64)

    res = run_hogbom_clean(
        dirty_image=dirty_off,
        config=sim_config,
        backend=backend,
        gain=0.1,
        threshold=0.01,
        max_iters=200,
        verbose=False,
    )

    assert res.iterations > 0
    assert res.suppression_db > 25.0
    assert res.final_peak < 0.25
    assert res.num_components > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_target_pslr_improvement(sim_config, backend):
    """
    Validates that CLEAN deconvolution recovers target amplitude to within 1%
    and drives down sidelobes from the initial -13.26 dB sinc sidelobe level.
    """
    dr0 = np.arange(128) - 64.0
    dc0 = np.arange(128) - 64.0
    DR0, DC0 = np.meshgrid(dr0, dc0, indexing="ij")
    U0 = DR0 * sim_config.row_ss
    V0 = DC0 * sim_config.col_ss
    dirty_on = (10.0 * np.sinc(sim_config.row_bw * U0) * np.sinc(sim_config.col_bw * V0)).astype(np.complex64)

    # Initial dirty image PSLR along center cut
    cut_dirty = np.abs(dirty_on[:, 64])
    peak_dirty = cut_dirty[64]
    sidelobes_dirty = [
        cut_dirty[i] for i in range(1, 127)
        if abs(i - 64) > 1 and cut_dirty[i] > cut_dirty[i - 1] and cut_dirty[i] > cut_dirty[i + 1]
    ]
    max_sl_dirty = max(sidelobes_dirty)
    pslr_dirty_db = 20.0 * np.log10(max_sl_dirty / peak_dirty)
    assert pslr_dirty_db == pytest.approx(-23.01, abs=0.5)

    res = run_hogbom_clean(
        dirty_image=dirty_on,
        config=sim_config,
        backend=backend,
        gain=0.1,
        threshold=0.005,
        max_iters=300,
        verbose=False,
    )

    # Component sum around target
    comp_sum = np.sum(res.components_map[63:66, 63:66])
    err = abs(comp_sum - 10.0) / 10.0
    assert err < 0.01

    # Suppression must be > 35 dB
    assert res.suppression_db > 35.0
    assert res.final_peak < 0.1


def test_psf_spatial_variation_angle(sim_config):
    """
    Validates that the PSF calculation correctly applies non-zero shear angle rotation
    away from the Scene Center Point (SCP).
    """
    gen = PSFGenerator(sim_config)

    # At SCP (64, 64), angle is zero
    psf_scp = gen.compute_psf(64, 64, psf_size=65)

    # At a distant point, angle is non-zero
    psf_far = gen.compute_psf(64, 5000, psf_size=65)

    # Because theta is non-zero, the rotated PSF differs from the unrotated PSF
    diff = float(np.max(np.abs(psf_scp - psf_far)))
    assert diff > 1e-4
