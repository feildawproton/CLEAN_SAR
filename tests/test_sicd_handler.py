import os
import glob
import tempfile
import pytest
import numpy as np
from clean_sar.sicd_handler import SICDHandler


def test_sicd_handler_init(test_sicd_path):
    handler = SICDHandler(test_sicd_path)

    assert handler.num_rows > 0
    assert handler.num_cols > 0
    assert handler.row_ss > 0
    assert handler.col_ss > 0
    assert handler.row_bw > 0
    assert handler.col_bw > 0
    assert len(handler.scp_pixel) == 2


def test_sicd_handler_chip_read(test_sicd_path):
    handler = SICDHandler(test_sicd_path)

    r0, c0 = 100, 150
    h, w = 64, 64
    chip, chip_xml = handler.read_chip(r0, c0, r0 + h, c0 + w)

    assert chip.shape == (h, w)
    assert np.iscomplexobj(chip)
    assert chip_xml is not None


def test_sicd_handler_coordinates(test_sicd_path):
    handler = SICDHandler(test_sicd_path)

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


def test_sicd_handler_write_nitf(test_sicd_path):
    handler = SICDHandler(test_sicd_path)

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


def test_sicd_handler_write_nitf_no_mutation(test_sicd_path):
    """Verify Finding 3: write_nitf does not mutate handler's self.xmltree in-place."""
    handler = SICDHandler(test_sicd_path)
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


def test_sicd_handler_impresp_validation(test_sicd_path):
    """Verify C12: _validate_impresp issues UserWarning on broadening factor mismatch > 5%."""
    import warnings
    handler = SICDHandler(test_sicd_path)

    # Artificially set ImpRespWid = 1.0 / ImpRespBW (k=1.0 for UNIFORM, ~13% mismatch)
    handler.row_wid = 1.0 / handler.row_bw
    handler.row_wgt_name = "UNIFORM"
    with pytest.warns(UserWarning, match="Rayleigh resolution"):
        handler._validate_impresp()

    # When k=0.886 for UNIFORM, no warning should be raised
    handler.row_wid = 0.886 / handler.row_bw
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        handler._validate_impresp()
        assert len([w for w in record if "Rayleigh" in str(w.message)]) == 0

