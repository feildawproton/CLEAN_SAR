import numpy as np
import pytest

from clean_sar.config import CleanPhysicsConfig
from clean_sar.algorithm import run_hogbom_clean
from clean_sar.quality import (
    find_bright_targets,
    ipr_quality,
    verdict,
)


def test_find_bright_targets():
    """Validates locating bright, isolated targets in an image against background."""
    img = np.zeros((128, 128), dtype=np.complex64) + 0.01
    # Inject three targets of varying brightness
    img[50, 50] = 100.0 + 0j
    img[80, 80] = 50.0 + 0j
    img[50, 80] = 25.0 + 0j

    targets = find_bright_targets(img, n=3, edge=20, sep=10)
    assert len(targets) == 3
    assert targets[0] == (50, 50)
    assert targets[1] == (80, 80)
    assert targets[2] == (50, 80)


def test_ipr_quality_metrics():
    """Validates that real deconvolution yields negative ISLR change and passes quality check."""
    config = CleanPhysicsConfig(
        row_ss=0.5, col_ss=0.5,
        row_bw=1.5, col_bw=1.5,
        row_wid=0.886 / 1.5, col_wid=0.886 / 1.5,
        scp_slant_range=600_000.0,
        scp_row=64.0, scp_col=64.0,
        row_wgt="UNIFORM", col_wgt="UNIFORM",
    )

    dr = np.arange(128) - 64.0
    dc = np.arange(128) - 64.0
    DR, DC = np.meshgrid(dr, dc, indexing="ij")
    U = DR * config.row_ss
    V = DC * config.col_ss
    dirty = (10.0 * np.sinc(config.row_bw * U) * np.sinc(config.col_bw * V)).astype(np.complex64)

    res = run_hogbom_clean(
        dirty_image=dirty,
        config=config,
        backend="c",
        gain=0.1,
        threshold=0.01,
        max_iters=300,
        verbose=False,
    )

    metrics = ipr_quality(
        dirty=dirty,
        clean=res.clean_image,
        row_wid=config.row_wid,
        col_wid=config.col_wid,
        row_ss=config.row_ss,
        col_ss=config.col_ss,
        target=(64, 64),
    )

    assert "islr_change_db" in metrics
    assert "mainlobe_preservation" in metrics
    # Sidelobe energy should decrease substantially
    assert metrics["islr_change_db"] < -10.0
    # Mainlobe should be preserved close to 1.0
    assert 0.85 <= metrics["mainlobe_preservation"] <= 1.15
    # Verdict should be OK
    v = verdict(metrics)
    assert v == "OK"
