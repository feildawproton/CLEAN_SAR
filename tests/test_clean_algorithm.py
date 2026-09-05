import pytest
import numpy as np
import torch
from clean_sar.algorithm import run_hogbom_clean, CleanResult
from clean_sar.config import CleanPhysicsConfig
from clean_sar.sicd_handler import SICDHandler
import glob


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_exact_spatially_varying_clean_synthetic():
    file_path = get_test_sicd()
    assert file_path is not None, "No test SICD file found."

    handler = SICDHandler(file_path)

    # Read a 128x128 chip
    chip, _ = handler.read_chip(100, 100, 228, 228)
    config = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(100, 100))

    res = run_hogbom_clean(
        dirty_image=chip,
        config=config,
        beam_type="gaussian",
        psf_size=33,
        gain=0.1,
        threshold=0.05,
        max_iters=200,
    )

    assert isinstance(res, CleanResult)
    assert res.iterations > 0
    assert res.clean_image.shape == chip.shape
    assert res.residual_image.shape == chip.shape
    assert np.max(np.abs(res.residual_image)) < np.max(np.abs(chip))
