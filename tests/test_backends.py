import pytest
import numpy as np
import glob
from clean_sar.backends import resolve_backend, is_cuda_native_available
from clean_sar.algorithm import run_hogbom_clean
from clean_sar.processor import CLEANProcessor
from clean_sar.config import CleanPhysicsConfig
from clean_sar.sicd_handler import SICDHandler


def get_test_sicd():
    files = sorted(glob.glob("/home/feildaw/data/*.nitf"))
    if not files:
        files = sorted(glob.glob("/home/feildaw/diffpfa/workspace/output/*.nitf"))
    return files[0] if files else None


def test_resolve_backend_logic():
    assert is_cuda_native_available() is True
    assert resolve_backend("auto") == "cuda"
    assert resolve_backend("pytorch") == "pytorch"
    assert resolve_backend("cuda") == "cuda"

    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend("opencl")


def test_cuda_backend_execution_parity():
    """Verify native C++/CUDA backend produces equivalent results to PyTorch backend."""
    file_path = get_test_sicd()
    assert file_path is not None
    handler = SICDHandler(file_path)
    chip, _ = handler.read_chip(100, 100, 228, 228)
    config = CleanPhysicsConfig.from_sicd_handler(handler, chip_start=(100, 100))

    # Run PyTorch Backend
    res_torch = run_hogbom_clean(
        dirty_image=chip,
        config=config,
        backend="pytorch",
        max_iters=100,
        gain=0.1,
        threshold=0.05,
    )

    # Run Native CUDA Backend
    res_cuda = run_hogbom_clean(
        dirty_image=chip,
        config=config,
        backend="cuda",
        max_iters=100,
        gain=0.1,
        threshold=0.05,
    )

    assert res_cuda.iterations > 0
    assert res_cuda.clean_image.shape == chip.shape
    assert res_cuda.suppression_db > 0.0

    # Compare iterations and final suppression
    print(f"PyTorch iters: {res_torch.iterations}, CUDA iters: {res_cuda.iterations}")
    print(f"PyTorch suppr: {res_torch.suppression_db:.2f} dB, CUDA suppr: {res_cuda.suppression_db:.2f} dB")
    assert abs(res_torch.suppression_db - res_cuda.suppression_db) < 1.0


def test_processor_with_cuda_backend(tmp_path):
    file_path = get_test_sicd()
    assert file_path is not None
    out_file = str(tmp_path / "test_cuda_proc.nitf")

    proc = CLEANProcessor(
        input_path=file_path,
        output_path=out_file,
        chip_bounds=(100, 100, 228, 228),
        backend="cuda",
    )

    res = proc.run(
        gain=0.1,
        threshold=0.05,
        max_iters=100,
        verbose=False,
    )

    assert res.iterations > 0
    assert res.suppression_db > 0.0
