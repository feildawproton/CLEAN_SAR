import pytest
import os
import glob
import numpy as np
from clean_sar import CLEANProcessor, SICDHandler


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_clean_processor_chip(tmp_path):
    input_file = get_test_sicd()
    assert input_file is not None, "No test SICD file found."

    out_file = str(tmp_path / "test_chip_clean.nitf")
    chip_bounds = (100, 100, 228, 228)

    proc = CLEANProcessor(
        input_path=input_file,
        output_path=out_file,
        chip_bounds=chip_bounds,
    )

    res = proc.run(
        gain=0.1,
        threshold=0.05,
        max_iters=100,
        psf_size=33,
        verbose=False,
    )

    assert os.path.exists(out_file)
    assert res.iterations > 0
    assert res.clean_image.shape == (128, 128)

    # Verify written SICD can be reopened
    h_out = SICDHandler(out_file)
    assert h_out.num_rows == 128
    assert h_out.num_cols == 128
