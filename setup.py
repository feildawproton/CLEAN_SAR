import os
import sys
import shutil
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def compile_cuda_nvcc():
    """Attempts to compile clean_hogbom.cu to clean_hogbom.ptx using nvcc."""
    c_src_dir = os.path.join(os.path.dirname(__file__), "clean_sar", "backends", "c_src")
    cu_file = os.path.join(c_src_dir, "clean_hogbom.cu")
    ptx_file = os.path.join(c_src_dir, "clean_hogbom.ptx")

    if not os.path.isfile(cu_file):
        return False

    nvcc_bin = shutil.which("nvcc")
    if not nvcc_bin:
        for candidate in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.5/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12/bin/nvcc",
        ]:
            if os.path.isfile(candidate):
                nvcc_bin = candidate
                break

    if not nvcc_bin:
        print("\n" + "=" * 72)
        print("[CLEAN_SAR] Notice: nvcc was not found on PATH or in /usr/local/cuda/bin.")
        print("[CLEAN_SAR] Ahead-of-time compilation skipped.")
        print("[CLEAN_SAR] At runtime, CLEAN_SAR will use the libnvrtc fallback compiler.")
        print("=" * 72 + "\n")
        return False

    print(f"\n[CLEAN_SAR] Found nvcc at: {nvcc_bin}")
    print("[CLEAN_SAR] Compiling CUDA kernels ahead-of-time to clean_hogbom.ptx...")
    cmd = [
        nvcc_bin,
        "-ptx",
        "--std=c++14",
        cu_file,
        "-o",
        ptx_file,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.isfile(ptx_file):
            print("[CLEAN_SAR] Ahead-of-time CUDA PTX compilation successful!\n")
            return True
        else:
            print(f"[CLEAN_SAR] nvcc compilation failed:\n{res.stderr}")
            print("[CLEAN_SAR] Will fall back to runtime NVRTC (libnvrtc) compilation.\n")
            return False
    except Exception as e:
        print(f"[CLEAN_SAR] Failed to invoke nvcc: {e}")
        print("[CLEAN_SAR] Will fall back to runtime NVRTC (libnvrtc) compilation.\n")
        return False


def compile_c_backend():
    """Attempts to compile clean_hogbom.c to libclean_c.so using gcc or clang."""
    c_src_dir = os.path.join(os.path.dirname(__file__), "clean_sar", "backends", "c_src")
    c_file = os.path.join(c_src_dir, "clean_hogbom.c")
    so_file = os.path.join(c_src_dir, "libclean_c.so")

    if not os.path.isfile(c_file):
        return False

    compiler = None
    for cand in ["gcc", "clang", "cc"]:
        if shutil.which(cand):
            compiler = cand
            break

    if not compiler:
        print("\n" + "=" * 72)
        print("[CLEAN_SAR] Notice: No C compiler (gcc/clang/cc) found on PATH.")
        print("[CLEAN_SAR] The C fallback backend will use the vectorized engine")
        print("[CLEAN_SAR] until a C compiler is installed.")
        print("=" * 72 + "\n")
        return False

    print(f"[CLEAN_SAR] Compiling native C backend with {compiler}...")
    cmd = [compiler, "-O3", "-fPIC", "-shared", c_file, "-o", so_file, "-lm"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.isfile(so_file):
            print("[CLEAN_SAR] Successfully built libclean_c.so!\n")
            return True
        else:
            print(f"[CLEAN_SAR] C compilation failed:\n{res.stderr}\n")
            return False
    except Exception as e:
        print(f"[CLEAN_SAR] Failed to invoke {compiler}: {e}\n")
        return False


class CustomBuildPy(build_py):
    def run(self):
        compile_cuda_nvcc()
        compile_c_backend()
        super().run()


class CustomDevelop(develop):
    def run(self):
        compile_cuda_nvcc()
        compile_c_backend()
        super().run()


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_py": CustomBuildPy,
            "develop": CustomDevelop,
        },
    )
