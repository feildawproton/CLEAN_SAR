import os
import glob
import pytest


def get_test_sicd():
    """
    Locates a test NITF SICD file.
    Checks CLEAN_SAR_TEST_DATA environment variable first,
    then standard local data directories. Returns None if no files are found.
    """
    env_dir = os.environ.get("CLEAN_SAR_TEST_DATA")
    if env_dir and os.path.exists(env_dir):
        files = sorted(glob.glob(os.path.join(env_dir, "*.nitf")))
        if files:
            return files[0]

    for pat in ("/home/feildaw/data/*.nitf", "/home/feildaw/diffpfa/workspace/output/*.nitf"):
        files = sorted(glob.glob(pat))
        if files:
            return files[0]
    return None


@pytest.fixture(scope="session")
def test_sicd_path():
    fp = get_test_sicd()
    if fp is None:
        pytest.skip("No test SICD file found (set CLEAN_SAR_TEST_DATA to directory with .nitf files)")
    return fp
