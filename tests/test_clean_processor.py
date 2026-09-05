import pytest
import os
import glob
import numpy as np
from clean_sar import CLEANProcessor, SICDHandler


def test_clean_processor_chip(tmp_path, test_sicd_path):
    out_file = str(tmp_path / "test_chip_clean.nitf")
    chip_bounds = (100, 100, 228, 228)

    proc = CLEANProcessor(
        input_path=test_sicd_path,
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
