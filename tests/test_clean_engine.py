import pytest
import numpy as np
import torch
from clean_sar.engine import run_hogbom_clean, CleanResult
from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator
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
    psf_gen = PSFGenerator(handler)

    # Read a 128x128 chip
    chip, _ = handler.read_chip(100, 100, 228, 228)

    res = run_hogbom_clean(
        dirty_image=chip,
        psf_generator=psf_gen,
        method="kspace",
        beam_type="gaussian",
        psf_size=33,
        gain=0.1,
        threshold=0.05,
        max_iters=200,
        chip_origin=(100, 100),
    )

    assert isinstance(res, CleanResult)
    assert res.iterations > 0
    assert res.clean_image.shape == chip.shape
    assert res.residual_image.shape == chip.shape
    assert np.max(np.abs(res.residual_image)) < np.max(np.abs(chip))


def test_exact_spatially_varying_clean_analytic():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    chip, _ = handler.read_chip(200, 200, 328, 328)

    res = run_hogbom_clean(
        dirty_image=chip,
        psf_generator=psf_gen,
        method="analytic",
        beam_type="gaussian",
        psf_size=33,
        gain=0.1,
        threshold=0.05,
        max_iters=200,
        chip_origin=(200, 200),
    )

    assert isinstance(res, CleanResult)
    assert res.iterations > 0
    assert res.clean_image.shape == chip.shape
