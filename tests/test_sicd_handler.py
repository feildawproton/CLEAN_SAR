import os
import glob
import tempfile
import pytest
import numpy as np
from clean_sar.sicd_handler import SICDHandler


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_sicd_handler_init():
    file_path = get_test_sicd()
    assert file_path is not None, "No test SICD file found."
    handler = SICDHandler(file_path)

    assert handler.num_rows > 0
    assert handler.num_cols > 0
    assert handler.row_ss > 0
    assert handler.col_ss > 0
    assert handler.row_bw > 0
    assert handler.col_bw > 0
    assert len(handler.scp_pixel) == 2


def test_sicd_handler_chip_read():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)

    r0, c0 = 100, 150
    h, w = 64, 64
    chip, chip_xml = handler.read_chip(r0, c0, r0 + h, c0 + w)

    assert chip.shape == (h, w)
    assert np.iscomplexobj(chip)
    assert chip_xml is not None


def test_sicd_handler_coordinates():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)

    # Test chip to global
    g_r, g_c = handler.chip_to_global_rowcol(10, 20, 100, 200)
    assert g_r == 110
    assert g_c == 220

    # Test metric conversion at SCP (should be close to (0, 0))
    scp_r, scp_c = handler.scp_pixel
    xrow, ycol = handler.global_to_metric(scp_r, scp_c)
    assert np.isclose(xrow, 0.0, atol=1e-3)
    assert np.isclose(ycol, 0.0, atol=1e-3)

    # Round trip
    r_back, c_back = handler.metric_to_global(xrow, ycol)
    assert np.isclose(r_back, scp_r, atol=1e-3)
    assert np.isclose(c_back, scp_c, atol=1e-3)


def test_sicd_handler_write_nitf():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)

    chip, chip_xml = handler.read_chip(100, 100, 164, 164)
    with tempfile.NamedTemporaryFile(suffix=".nitf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        handler.write_nitf(tmp_path, chip, custom_xmltree=chip_xml)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        # Verify reading written file
        handler_reopen = SICDHandler(tmp_path)
        assert handler_reopen.num_rows == 64
        assert handler_reopen.num_cols == 64
        img_read = handler_reopen.read_full_image()
        assert img_read.shape == (64, 64)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_sicd_handler_write_nitf_no_mutation():
    """Verify Finding 3: write_nitf does not mutate handler's self.xmltree in-place."""
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    orig_rows = handler.num_rows
    orig_cols = handler.num_cols

    chip = np.zeros((32, 32), dtype=np.complex64)
    with tempfile.NamedTemporaryFile(suffix=".nitf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write without custom_xmltree (should deepcopy internally)
        handler.write_nitf(tmp_path, chip, custom_xmltree=None)
        # Original handler's XML NumRows/NumCols must be unchanged
        xh = handler.xh
        assert int(xh.load("./{*}ImageData/{*}NumRows")) == orig_rows
        assert int(xh.load("./{*}ImageData/{*}NumCols")) == orig_cols
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

