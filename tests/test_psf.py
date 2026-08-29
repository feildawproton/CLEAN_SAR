import glob
import pytest
import numpy as np
from clean_sar.sicd_handler import SICDHandler
from clean_sar.psf import PSFGenerator


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_psf_kspace_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    psf = psf_gen.compute_psf_kspace(scp_r, scp_c, psf_size=65)

    assert psf.shape == (65, 65)
    assert np.iscomplexobj(psf)
    kh = 65 // 2
    # Peak at center should be 1.0 (magnitude) and zero imaginary
    assert np.isclose(np.abs(psf[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].real, 1.0, atol=1e-5)
    assert np.isclose(psf[kh, kh].imag, 0.0, atol=1e-5)


def test_psf_analytic_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    psf = psf_gen.compute_psf_analytic(scp_r, scp_c, psf_size=65)

    assert psf.shape == (65, 65)
    assert np.iscomplexobj(psf)
    kh = 65 // 2
    assert np.isclose(np.abs(psf[kh, kh]), 1.0, atol=1e-5)


def test_clean_beam_generation():
    file_path = get_test_sicd()
    handler = SICDHandler(file_path)
    psf_gen = PSFGenerator(handler)

    scp_r, scp_c = handler.scp_pixel
    beam_gauss = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=65, beam_type="gaussian")
    beam_main = psf_gen.compute_clean_beam(scp_r, scp_c, psf_size=65, beam_type="mainlobe")

    assert beam_gauss.shape == (65, 65)
    assert beam_main.shape == (65, 65)
    kh = 65 // 2
    assert np.isclose(np.abs(beam_gauss[kh, kh]), 1.0, atol=1e-5)
    assert np.isclose(np.abs(beam_main[kh, kh]), 1.0, atol=1e-5)
