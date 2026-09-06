import os
from clean_sar import CLEANProcessor, SICDHandler


def test_clean_processor_chip(tmp_path, test_sicd_path):
    # Prepare a caller-provided chip file
    chip_file = str(tmp_path / "caller_chip.nitf")
    h_orig = SICDHandler(test_sicd_path)
    chip_arr, chip_xml = h_orig.read_chip(100, 100, 228, 228)
    h_orig.write_nitf(chip_file, chip_arr, custom_xmltree=chip_xml)

    out_file = str(tmp_path / "test_chip_clean.nitf")

    proc = CLEANProcessor(
        input_path=chip_file,
        output_path=out_file,
    )

    res = proc.run(
        gain=0.1,
        threshold=0.05,
        max_iters=100,
        verbose=False,
    )

    assert os.path.exists(out_file)
    assert res.iterations > 0
    assert res.clean_image.shape == (128, 128)

    # Verify written SICD can be reopened
    h_out = SICDHandler(out_file)
    assert h_out.num_rows == 128
    assert h_out.num_cols == 128
