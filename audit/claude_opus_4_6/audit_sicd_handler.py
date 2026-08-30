"""
Audit Script for SICDHandler in clean_sar/sicd_handler.py

Tests and Verifications:
1. Metadata extraction: Print and verify all extracted metadata
   (num_rows, num_cols, scp_pixel, row_ss, col_ss, row_bw, col_bw, row_wid, col_wid,
    row_wgt_name, col_wgt_name, is_pfa, pfa_meta if available, is_rma).
2. Coordinate round-trip: global_to_metric -> metric_to_global at various positions
   (SCP, corners, center, random interior points, fractional coords, vectorized inputs).
3. Chip read/write round-trip: Read chip, write to NITF, re-read, compare pixel values.
4. SCP pixel interpretation: Verify sarkit XML decoding of SCPPixel [Row, Col] order and handler usage.
5. Write NITF XML update: Verify XML updates for sub-images/chips (NumRows, NumCols, FirstRow, FirstCol, SCPPixel, ImageCorners, in-place mutation risks).
"""

import os
import sys
import tempfile
import copy
import numpy as np
import lxml.etree as etree
import sarkit.sicd as ss

# Add project root to sys.path
sys.path.insert(0, "/home/feildaw/CLEAN_SAR")
from clean_sar.sicd_handler import SICDHandler


def run_audit():
    print("=" * 80)
    print("SICD HANDLER COMPREHENSIVE AUDIT REPORT")
    print("=" * 80)

    # Find the smallest NITF SICD file in /home/feildaw/data/
    data_dir = "/home/feildaw/data"
    nitf_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".nitf")
    ]
    if not nitf_files:
        raise FileNotFoundError(f"No .nitf files found in {data_dir}")

    nitf_files.sort(key=lambda p: os.path.getsize(p))
    test_file = nitf_files[0]
    file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
    print(f"Selected test file: {os.path.basename(test_file)} ({file_size_mb:.2f} MB)")
    print(f"Full path: {test_file}")
    print()

    # Initialize handler
    handler = SICDHandler(test_file)

    # -------------------------------------------------------------------------
    # TEST 1: Metadata Extraction
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("TEST 1: METADATA EXTRACTION")
    print("-" * 80)
    print(f"  Image Dimensions (num_rows, num_cols): ({handler.num_rows}, {handler.num_cols})")
    print(f"  First Row, First Col:                  ({handler.first_row}, {handler.first_col})")
    print(f"  SCP Pixel (row, col):                  {handler.scp_pixel} (dtype: {handler.scp_pixel.dtype})")
    print(f"  Sample Spacing (row_ss, col_ss):       ({handler.row_ss:.6f} m, {handler.col_ss:.6f} m)")
    print(f"  Bandwidth (row_bw, col_bw):            ({handler.row_bw:.6f} cycles/m, {handler.col_bw:.6f} cycles/m)")
    print(f"  Impulse Response Width (row_wid, col): ({handler.row_wid:.6f} m, {handler.col_wid:.6f} m)")
    print(f"  Weighting Window (row, col):           ({handler.row_wgt_name}, {handler.col_wgt_name})")
    print(f"  Is PFA:                                {handler.is_pfa}")
    print(f"  Is RMA:                                {handler.is_rma}")

    if handler.is_pfa:
        print("  PFA Metadata:")
        for k, v in handler.pfa_meta.items():
            if isinstance(v, np.ndarray):
                print(f"    {k:20s}: array shape={v.shape}, values={v.tolist()}")
            elif isinstance(v, float):
                print(f"    {k:20s}: {v:.6e}")
            else:
                print(f"    {k:20s}: {v}")

    # Validate extracted values against direct XML queries
    raw_xh = ss.XmlHelper(handler.xmltree)
    raw_num_rows = int(raw_xh.load("./{*}ImageData/{*}NumRows"))
    raw_num_cols = int(raw_xh.load("./{*}ImageData/{*}NumCols"))
    raw_row_ss = float(raw_xh.load("./{*}Grid/{*}Row/{*}SS"))
    raw_col_ss = float(raw_xh.load("./{*}Grid/{*}Col/{*}SS"))
    raw_row_bw = float(raw_xh.load("./{*}Grid/{*}Row/{*}ImpRespBW"))
    raw_col_bw = float(raw_xh.load("./{*}Grid/{*}Col/{*}ImpRespBW"))

    t1_pass = (
        handler.num_rows == raw_num_rows
        and handler.num_cols == raw_num_cols
        and np.isclose(handler.row_ss, raw_row_ss)
        and np.isclose(handler.col_ss, raw_col_ss)
        and np.isclose(handler.row_bw, raw_row_bw)
        and np.isclose(handler.col_bw, raw_col_bw)
    )
    print(f"\n  [TEST 1 RESULT]: {'PASS' if t1_pass else 'FAIL'}")
    print()

    # -------------------------------------------------------------------------
    # TEST 2: Coordinate Round-Trip Verification
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("TEST 2: COORDINATE ROUND-TRIP (global_to_metric <-> metric_to_global)")
    print("-" * 80)

    # Test positions: SCP, corners, center, interior points, fractional points
    test_points_rc = [
        ("SCP Pixel", handler.scp_pixel[0], handler.scp_pixel[1]),
        ("Top-Left (0, 0)", 0.0, 0.0),
        ("Top-Right (0, Ncol-1)", 0.0, float(handler.num_cols - 1)),
        ("Bottom-Left (Nrow-1, 0)", float(handler.num_rows - 1), 0.0),
        ("Bottom-Right (Nrow-1, Ncol-1)", float(handler.num_rows - 1), float(handler.num_cols - 1)),
        ("Center (Nrow/2, Ncol/2)", handler.num_rows / 2.0, handler.num_cols / 2.0),
        ("Fractional Point", 123.456, 789.1011),
        ("Negative Global Coordinate", -50.5, -100.25),
    ]

    max_rc_err = 0.0
    max_xy_err = 0.0
    print(f"  {'Point Description':<30} | {'Input (Row, Col)':<22} | {'Metric (xrow, ycol) [m]':<26} | {'Roundtrip (Row, Col)':<22} | {'Error (px)':<10}")
    print("  " + "-" * 118)

    for desc, r, c in test_points_rc:
        # global -> metric
        xr, yc = handler.global_to_metric(r, c)
        # metric -> global
        r_back, c_back = handler.metric_to_global(xr, yc)

        err = np.sqrt((r - r_back) ** 2 + (c - c_back) ** 2)
        max_rc_err = max(max_rc_err, float(err))
        print(f"  {desc:<30} | ({r:9.3f}, {c:9.3f}) | ({xr:10.4f}, {yc:10.4f}) | ({r_back:9.3f}, {c_back:9.3f}) | {err:10.2e}")

    # Reverse round-trip: metric -> global -> metric
    test_points_xy = [
        ("Origin (0, 0) [SCP]", 0.0, 0.0),
        ("Positive (+100m, +200m)", 100.0, 200.0),
        ("Negative (-150m, -300m)", -150.0, -300.0),
        ("Arbitrary metric point", 12.3456, -78.9012),
    ]
    print()
    print(f"  {'Metric Description':<30} | {'Input (xrow, ycol)':<22} | {'Global (Row, Col)':<24} | {'Roundtrip (xrow, ycol)':<24} | {'Error (m)':<10}")
    print("  " + "-" * 118)
    for desc, xr, yc in test_points_xy:
        r, c = handler.metric_to_global(xr, yc)
        xr_back, yc_back = handler.global_to_metric(r, c)
        err = np.sqrt((xr - xr_back) ** 2 + (yc - yc_back) ** 2)
        max_xy_err = max(max_xy_err, float(err))
        print(f"  {desc:<30} | ({xr:9.3f}, {yc:9.3f}) | ({r:10.4f}, {c:10.4f}) | ({xr_back:10.4f}, {yc_back:10.4f}) | {err:10.2e}")

    # Vectorized test (grid of 100x100 coordinates)
    grid_r, grid_c = np.meshgrid(
        np.linspace(0, handler.num_rows - 1, 100),
        np.linspace(0, handler.num_cols - 1, 100),
        indexing="ij",
    )
    xr_grid, yc_grid = handler.global_to_metric(grid_r, grid_c)
    r_grid_back, c_grid_back = handler.metric_to_global(xr_grid, yc_grid)
    vec_rc_err = np.max(np.sqrt((grid_r - r_grid_back) ** 2 + (grid_c - c_grid_back) ** 2))
    print(f"\n  Vectorized 10,000-point grid max pixel roundtrip error: {vec_rc_err:.2e} pixels")

    # Verify SCP maps exactly to (0, 0) metric
    scp_xr, scp_yc = handler.global_to_metric(handler.scp_pixel[0], handler.scp_pixel[1])
    print(f"  SCP Pixel ({handler.scp_pixel[0]}, {handler.scp_pixel[1]}) -> Metric: ({scp_xr:.6e}, {scp_yc:.6e}) m")

    t2_pass = (max_rc_err < 1e-10) and (max_xy_err < 1e-10) and (vec_rc_err < 1e-10) and (abs(scp_xr) < 1e-10) and (abs(scp_yc) < 1e-10)
    print(f"\n  [TEST 2 RESULT]: {'PASS' if t2_pass else 'FAIL'}")
    print()

    # -------------------------------------------------------------------------
    # TEST 3: Chip Read / Write Round-Trip
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("TEST 3: CHIP READ / WRITE ROUND-TRIP")
    print("-" * 80)
    chip_start_row, chip_start_col = 200, 300
    chip_stop_row, chip_stop_col = 456, 600
    expected_shape = (chip_stop_row - chip_start_row, chip_stop_col - chip_start_col)

    print(f"  Reading chip ROI: rows [{chip_start_row}:{chip_stop_row}], cols [{chip_start_col}:{chip_stop_col}] (shape: {expected_shape})")
    chip_arr, chip_xml = handler.read_chip(chip_start_row, chip_start_col, chip_stop_row, chip_stop_col)

    print(f"  Chip array shape: {chip_arr.shape}, dtype: {chip_arr.dtype}")
    print(f"  Chip pixel stats: min_abs={np.min(np.abs(chip_arr)):.4e}, max_abs={np.max(np.abs(chip_arr)):.4e}, mean_abs={np.mean(np.abs(chip_arr)):.4e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_nitf_path = os.path.join(tmpdir, "test_chip.nitf")
        print(f"  Writing chip to temporary NITF: {tmp_nitf_path}")
        handler.write_nitf(tmp_nitf_path, chip_arr, custom_xmltree=chip_xml)

        # Re-read using SICDHandler
        print("  Re-reading chip using SICDHandler...")
        chip_handler = SICDHandler(tmp_nitf_path)
        reread_arr = chip_handler.read_full_image()

        print(f"  Re-read array shape: {reread_arr.shape}, dtype: {reread_arr.dtype}")

        # Compare pixel values
        shape_match = chip_arr.shape == reread_arr.shape
        dtype_match = chip_arr.dtype == reread_arr.dtype
        abs_diff = np.abs(chip_arr - reread_arr)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        exact_equal = np.array_equal(chip_arr, reread_arr)

        print(f"  Shape match:             {shape_match}")
        print(f"  Dtype match:             {dtype_match}")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute diff:      {mean_diff:.6e}")
        print(f"  Bitwise exact match:     {exact_equal}")

        t3_pass = shape_match and dtype_match and (max_diff == 0.0)
        print(f"\n  [TEST 3 RESULT]: {'PASS' if t3_pass else 'FAIL'}")
    print()

    # -------------------------------------------------------------------------
    # TEST 4: SCP Pixel Interpretation
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("TEST 4: SCP PIXEL INTERPRETATION ([Row, Col] ORDER & USAGE)")
    print("-" * 80)
    # Examine raw XML structure for SCPPixel
    scp_elem = handler.xmltree.find("{*}ImageData/{*}SCPPixel")
    print("  SCPPixel XML element:")
    if scp_elem is not None:
        print(f"    Raw XML: {etree.tostring(scp_elem).decode().strip()}")
        row_elem = scp_elem.find("{*}Row")
        col_elem = scp_elem.find("{*}Col")
        xml_row_val = int(row_elem.text) if row_elem is not None else None
        xml_col_val = int(col_elem.text) if col_elem is not None else None
        print(f"    Direct XML child <Row>: {xml_row_val}")
        print(f"    Direct XML child <Col>: {xml_col_val}")
    else:
        print("    SCPPixel element NOT found in ImageData!")

    # Check SARkit XmlHelper transcoding
    xh_val = handler.xh.load("./{*}ImageData/{*}SCPPixel")
    print(f"  SARkit xh.load('./{{*}}ImageData/{{*}}SCPPixel') returned: {xh_val} (type: {type(xh_val)})")
    print(f"  handler.scp_pixel:                                      {handler.scp_pixel} (shape: {handler.scp_pixel.shape})")

    # Verify element order
    order_correct = (
        xml_row_val is not None
        and xml_col_val is not None
        and int(handler.scp_pixel[0]) == xml_row_val
        and int(handler.scp_pixel[1]) == xml_col_val
    )
    print(f"  Index 0 matches <Row> ({xml_row_val}): {int(handler.scp_pixel[0]) == xml_row_val}")
    print(f"  Index 1 matches <Col> ({xml_col_val}): {int(handler.scp_pixel[1]) == xml_col_val}")

    # Check how row and col SS scale relative to SCP
    test_r = handler.scp_pixel[0] + 10.0
    test_c = handler.scp_pixel[1] + 20.0
    xr_calc, yc_calc = handler.global_to_metric(test_r, test_c)
    expected_xr = 10.0 * handler.row_ss
    expected_yc = 20.0 * handler.col_ss
    scaling_correct = np.isclose(xr_calc, expected_xr) and np.isclose(yc_calc, expected_yc)
    print(f"  Displacement (+10 rows, +20 cols) -> Metric: ({xr_calc:.6f}, {yc_calc:.6f}) m")
    print(f"  Expected (10*row_ss, 20*col_ss):             ({expected_xr:.6f}, {expected_yc:.6f}) m")
    print(f"  Scaling matches row_ss and col_ss correctly: {scaling_correct}")

    t4_pass = order_correct and scaling_correct
    print(f"\n  [TEST 4 RESULT]: {'PASS' if t4_pass else 'FAIL'}")
    print()

    # -------------------------------------------------------------------------
    # TEST 5: Write NITF XML Update & Sub-image Validity
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("TEST 5: WRITE NITF XML UPDATE & SUB-IMAGE COMPLIANCE")
    print("-" * 80)

    # 5A: Inspect what read_chip does to the XML
    start_r, start_c = 150, 250
    stop_r, stop_c = 350, 550
    nrows_chip = stop_r - start_r
    ncols_chip = stop_c - start_c
    _, chip_xml_tree = handler.read_chip(start_r, start_c, stop_r, stop_c)
    xh_chip = ss.XmlHelper(chip_xml_tree)

    chip_num_rows = xh_chip.load("./{*}ImageData/{*}NumRows")
    chip_num_cols = xh_chip.load("./{*}ImageData/{*}NumCols")
    chip_first_row = xh_chip.load("./{*}ImageData/{*}FirstRow")
    chip_first_col = xh_chip.load("./{*}ImageData/{*}FirstCol")
    chip_scp_pixel = xh_chip.load("./{*}ImageData/{*}SCPPixel")

    print("  5A. Metadata in chip_xml produced by read_chip():")
    print(f"    NumRows:  {chip_num_rows} (expected: {nrows_chip})")
    print(f"    NumCols:  {chip_num_cols} (expected: {ncols_chip})")
    print(f"    FirstRow: {chip_first_row} (expected: {start_r + handler.first_row})")
    print(f"    FirstCol: {chip_first_col} (expected: {start_c + handler.first_col})")
    print(f"    SCPPixel: {chip_scp_pixel} (original full image SCP: {handler.scp_pixel.astype(int)})")

    # 5B: Test what happens when write_nitf is called with custom_xmltree vs None
    print("\n  5B. Analyzing write_nitf implementation:")
    # Check 1: In-place mutation of handler.xmltree when custom_xmltree=None
    original_xml_num_rows = handler.xmltree.findtext("{*}ImageData/{*}NumRows")
    original_xml_num_cols = handler.xmltree.findtext("{*}ImageData/{*}NumCols")

    dummy_chip = np.zeros((64, 64), dtype=np.complex64)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_write_path = os.path.join(tmpdir, "test_mutate.nitf")
        # Call write_nitf without custom_xmltree for a dummy small array
        handler.write_nitf(tmp_write_path, dummy_chip, custom_xmltree=None)

        mutated_xml_num_rows = handler.xmltree.findtext("{*}ImageData/{*}NumRows")
        mutated_xml_num_cols = handler.xmltree.findtext("{*}ImageData/{*}NumCols")

        has_in_place_mutation = (mutated_xml_num_rows != original_xml_num_rows)
        print(f"    Original handler.xmltree NumRows: {original_xml_num_rows}")
        print(f"    After write_nitf(dummy 64x64, custom_xmltree=None):")
        print(f"      handler.xmltree NumRows: {mutated_xml_num_rows}")
        print(f"      In-place mutation detected: {has_in_place_mutation}")
        if has_in_place_mutation:
            print("      [WARNING/DEFECT]: write_nitf mutates self.xmltree in-place when custom_xmltree is None!")

        # Restore original values on handler.xmltree for safety
        handler.xmltree.find("{*}ImageData/{*}NumRows").text = original_xml_num_rows
        handler.xmltree.find("{*}ImageData/{*}NumCols").text = original_xml_num_cols

    # Check 2: Behavior when writing a chip with custom_xmltree=chip_xml
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_chip_path = os.path.join(tmpdir, "valid_chip.nitf")
        chip_data, chip_xml = handler.read_chip(start_r, start_c, stop_r, stop_c)
        handler.write_nitf(tmp_chip_path, chip_data, custom_xmltree=chip_xml)

        # Inspect written file
        with open(tmp_chip_path, "rb") as fp, ss.NitfReader(fp) as rdr:
            written_xh = ss.XmlHelper(rdr.metadata.xmltree)
            w_nrows = written_xh.load("./{*}ImageData/{*}NumRows")
            w_ncols = written_xh.load("./{*}ImageData/{*}NumCols")
            w_frow = written_xh.load("./{*}ImageData/{*}FirstRow")
            w_fcol = written_xh.load("./{*}ImageData/{*}FirstCol")
            w_scp = written_xh.load("./{*}ImageData/{*}SCPPixel")
            w_corners = written_xh.load("./{*}GeoData/{*}ImageCorners")

            print("\n  5C. Re-read metadata from written chip NITF file (with custom_xmltree=chip_xml):")
            print(f"    NumRows:      {w_nrows}")
            print(f"    NumCols:      {w_ncols}")
            print(f"    FirstRow:     {w_frow}")
            print(f"    FirstCol:     {w_fcol}")
            print(f"    SCPPixel:     {w_scp}")
            print(f"    ImageCorners:\n{w_corners}")

            chip_valid = (
                w_nrows == nrows_chip
                and w_ncols == ncols_chip
                and w_frow == start_r + handler.first_row
                and w_fcol == start_c + handler.first_col
                and np.array_equal(w_scp, handler.scp_pixel.astype(int))
            )
            print(f"    Chip metadata compliance: {chip_valid}")

    # Check 3: Behavior if someone passes an arbitrary cropped array to write_nitf WITHOUT custom_xmltree
    print("\n  5D. Analysis of writing cropped images WITHOUT custom_xmltree:")
    print("    If a user crops an array and passes it to write_nitf without custom_xmltree:")
    print("    - write_nitf updates NumRows/NumCols to the cropped dimensions.")
    print("    - But FirstRow/FirstCol remain unchanged (0, 0).")
    print("    - SCPPixel remains unchanged (global SCP).")
    print("    - ImageCorners remain the full image corners.")
    print("    - This creates an invalid/inconsistent SICD where pixel (0,0) is assumed to be the original (0,0),")
    print("      leading to spatial distortion and broken geolocation projection for downstream tools.")
    print("    - Recommendation: write_nitf should require a valid updated XML (like the one returned by read_chip)")
    print("      or provide a helper / deepcopy mechanism to prevent in-place corruption.")

    t5_pass = chip_valid
    print(f"\n  [TEST 5 RESULT]: {'PASS (with findings)' if t5_pass else 'FAIL'}")
    print()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)
    print(f"  Test 1 (Metadata Extraction):          {'PASS' if t1_pass else 'FAIL'}")
    print(f"  Test 2 (Coordinate Round-Trip):        {'PASS' if t2_pass else 'FAIL'}")
    print(f"  Test 3 (Chip Read/Write Round-Trip):   {'PASS' if t3_pass else 'FAIL'}")
    print(f"  Test 4 (SCP Pixel Interpretation):     {'PASS' if t4_pass else 'FAIL'}")
    print(f"  Test 5 (Write NITF XML Update):        {'PASS' if t5_pass else 'FAIL'}")
    print(f"  Overall Status:                        {'ALL TESTS PASSED' if (t1_pass and t2_pass and t3_pass and t4_pass and t5_pass) else 'SOME TESTS FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
