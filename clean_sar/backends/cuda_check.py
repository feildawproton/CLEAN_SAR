"""
clean_sar/backends/cuda_check.py

Checked CUDA Driver API and NVRTC wrappers with full ctypes function prototypes (argtypes),
error status checking, primary context retention (cuDevicePrimaryCtxRetain),
and dynamic compute capability detection.
"""
from __future__ import annotations

import ctypes
from ctypes import (
    POINTER, byref, c_char_p, c_int, c_size_t, c_uint, c_void_p,
)
from typing import Iterable, Optional, Sequence, Tuple

__all__ = ["CudaError", "NvrtcError", "CudaDriver", "Nvrtc"]


class CudaError(RuntimeError):
    """A CUDA driver call returned a non-zero status."""

    def __init__(self, fn: str, code: int, name: str, desc: str):
        self.fn, self.code, self.name, self.desc = fn, code, name, desc
        super().__init__(f"{fn} failed: {name} ({code}) -- {desc}")


class NvrtcError(RuntimeError):
    """NVRTC failed; carries the compiler log when one is available."""

    def __init__(self, fn: str, code: int, log: str = ""):
        self.fn, self.code, self.log = fn, code, log
        msg = f"{fn} failed with NVRTC status {code}"
        super().__init__(f"{msg}\n--- NVRTC log ---\n{log}" if log else msg)


# Signatures for driver entry points used in CLEAN_SAR.
_CUDA_SIGS = {
    "cuInit":                    ([c_uint], c_int),
    "cuDriverGetVersion":        ([POINTER(c_int)], c_int),
    "cuDeviceGet":               ([POINTER(c_int), c_int], c_int),
    "cuDeviceGetCount":          ([POINTER(c_int)], c_int),
    "cuDeviceGetAttribute":      ([POINTER(c_int), c_int, c_int], c_int),
    "cuDevicePrimaryCtxRetain":  ([POINTER(c_void_p), c_int], c_int),
    "cuDevicePrimaryCtxRelease_v2": ([c_int], c_int),
    "cuCtxCreate_v2":            ([POINTER(c_void_p), c_uint, c_int], c_int),
    "cuCtxDestroy_v2":           ([c_void_p], c_int),
    "cuCtxGetCurrent":           ([POINTER(c_void_p)], c_int),
    "cuCtxSetCurrent":           ([c_void_p], c_int),
    "cuCtxSynchronize":          ([], c_int),
    "cuModuleLoadData":          ([POINTER(c_void_p), c_void_p], c_int),
    "cuModuleUnload":            ([c_void_p], c_int),
    "cuModuleGetFunction":       ([POINTER(c_void_p), c_void_p, c_char_p], c_int),
    "cuMemAlloc_v2":             ([POINTER(c_void_p), c_size_t], c_int),
    "cuMemFree_v2":              ([c_void_p], c_int),
    "cuMemsetD8_v2":             ([c_void_p, ctypes.c_ubyte, c_size_t], c_int),
    "cuMemcpyHtoD_v2":           ([c_void_p, c_void_p, c_size_t], c_int),
    "cuMemcpyDtoH_v2":           ([c_void_p, c_void_p, c_size_t], c_int),
    "cuMemGetInfo_v2":           ([POINTER(c_size_t), POINTER(c_size_t)], c_int),
    "cuLaunchKernel":            ([c_void_p, c_uint, c_uint, c_uint,
                                   c_uint, c_uint, c_uint, c_uint,
                                   c_void_p, c_void_p, c_void_p], c_int),
    "cuGetErrorName":            ([c_int, POINTER(c_char_p)], c_int),
    "cuGetErrorString":          ([c_int, POINTER(c_char_p)], c_int),
}

_NVRTC_SIGS = {
    "nvrtcCreateProgram":     ([POINTER(c_void_p), c_char_p, c_char_p,
                                c_int, c_void_p, c_void_p], c_int),
    "nvrtcDestroyProgram":    ([POINTER(c_void_p)], c_int),
    "nvrtcCompileProgram":    ([c_void_p, c_int, POINTER(c_char_p)], c_int),
    "nvrtcGetPTXSize":        ([c_void_p, POINTER(c_size_t)], c_int),
    "nvrtcGetPTX":            ([c_void_p, c_char_p], c_int),
    "nvrtcGetProgramLogSize": ([c_void_p, POINTER(c_size_t)], c_int),
    "nvrtcGetProgramLog":     ([c_void_p, c_char_p], c_int),
}

_CUDA_CANDIDATES = ("libcuda.so.1", "libcuda.so")


class CudaDriver:
    """libcuda with verified argtypes, return-code checking, and primary context management."""

    def __init__(self, paths: Sequence[str] = _CUDA_CANDIDATES):
        self.lib = None
        for p in paths:
            try:
                self.lib = ctypes.CDLL(p)
                break
            except OSError:
                continue
        if self.lib is None:
            raise CudaError("CDLL(libcuda)", -1, "LIBRARY_NOT_FOUND",
                            f"none of {list(paths)} could be loaded")
        for name, (argtypes, restype) in _CUDA_SIGS.items():
            fn = getattr(self.lib, name, None)
            if fn is not None:
                fn.argtypes = argtypes
                fn.restype = restype

    def _describe(self, code: int):
        name = c_char_p()
        desc = c_char_p()
        try:
            self.lib.cuGetErrorName(code, byref(name))
            self.lib.cuGetErrorString(code, byref(desc))
        except Exception:
            pass
        dec = lambda p: p.value.decode() if p.value else "?"  # noqa: E731
        return dec(name), dec(desc)

    def call(self, fn_name: str, *args, allow: Iterable[int] = (0,)) -> int:
        fn = getattr(self.lib, fn_name, None)
        if fn is None:
            raise CudaError(fn_name, -1, "SYMBOL_NOT_FOUND",
                            "not present in this libcuda")
        rc = fn(*args)
        if rc not in allow:
            raise CudaError(fn_name, rc, *self._describe(rc))
        return rc

    def __getattr__(self, name: str):
        if not name.startswith("cu"):
            raise AttributeError(name)

        def _checked(*args, allow=(0,)):
            return self.call(name, *args, allow=allow)

        return _checked

    def init(self, flags: int = 0) -> "CudaDriver":
        self.call("cuInit", c_uint(flags))
        return self

    def device(self, ordinal: int = 0) -> c_int:
        d = c_int()
        self.call("cuDeviceGet", byref(d), c_int(ordinal))
        return d

    def compute_capability(self, dev: c_int) -> Tuple[int, int]:
        major = c_int()
        minor = c_int()
        # 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        # 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        self.call("cuDeviceGetAttribute", byref(major), 75, dev)
        self.call("cuDeviceGetAttribute", byref(minor), 76, dev)
        return major.value, minor.value

    def primary_context(self, dev: c_int) -> c_void_p:
        """
        Retain the device PRIMARY context.
        """
        ctx = c_void_p()
        self.call("cuDevicePrimaryCtxRetain", byref(ctx), dev)
        self.call("cuCtxSetCurrent", ctx)
        return ctx

    def current_context(self) -> Optional[int]:
        ctx = c_void_p()
        self.call("cuCtxGetCurrent", byref(ctx))
        return ctx.value

    def mem_info(self):
        free, total = c_size_t(), c_size_t()
        self.call("cuMemGetInfo_v2", byref(free), byref(total))
        return free.value, total.value

    def mem_alloc(self, nbytes: int) -> c_void_p:
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        p = c_void_p()
        self.call("cuMemAlloc_v2", byref(p), c_size_t(nbytes))
        if not p.value:
            raise CudaError("cuMemAlloc_v2", 0, "NULL_POINTER",
                            f"driver reported success but returned NULL for {nbytes} bytes")
        return p

    def mem_free(self, ptr: c_void_p) -> None:
        if ptr and ptr.value:
            self.call("cuMemFree_v2", ptr)

    def memset_d8(self, ptr: c_void_p, value: int, nbytes: int) -> None:
        self.call("cuMemsetD8_v2", ptr, ctypes.c_ubyte(value), c_size_t(nbytes))

    def memcpy_htod(self, dst: c_void_p, src, nbytes: int) -> None:
        self.call("cuMemcpyHtoD_v2", dst, src, c_size_t(nbytes))

    def memcpy_dtoh(self, dst, src: c_void_p, nbytes: int) -> None:
        self.call("cuMemcpyDtoH_v2", dst, src, c_size_t(nbytes))

    def launch(self, fn: c_void_p, grid, block, args, shared: int = 0, stream=None) -> None:
        gx, gy, gz = grid
        bx, by, bz = block
        self.call("cuLaunchKernel", fn,
                  c_uint(gx), c_uint(gy), c_uint(gz),
                  c_uint(bx), c_uint(by), c_uint(bz),
                  c_uint(shared), stream, ctypes.cast(args, c_void_p), None)

    def synchronize(self) -> None:
        self.call("cuCtxSynchronize")


class Nvrtc:
    """NVRTC compiler with argtypes, error checking, and diagnostic log capture."""

    def __init__(self, paths: Sequence[str]):
        self.lib = None
        for p in paths:
            try:
                self.lib = ctypes.CDLL(p)
                break
            except OSError:
                continue
        if self.lib is None:
            raise NvrtcError(f"CDLL(nvrtc) from {list(paths)}", -1)
        for name, (argtypes, restype) in _NVRTC_SIGS.items():
            fn = getattr(self.lib, name, None)
            if fn is not None:
                fn.argtypes = argtypes
                fn.restype = restype

    def _log(self, prog: c_void_p) -> str:
        try:
            n = c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, byref(n))
            if n.value <= 1:
                return ""
            buf = ctypes.create_string_buffer(n.value)
            self.lib.nvrtcGetProgramLog(prog, buf)
            return buf.value.decode(errors="replace").strip()
        except Exception:
            return ""

    def compile_to_ptx(self, source: bytes, name: bytes = b"kernel.cu",
                       options: Sequence[bytes] = ()) -> bytes:
        """Compile CUDA C++ to PTX, raising NvrtcError with compiler diagnostics on failure."""
        prog = c_void_p()
        rc = self.lib.nvrtcCreateProgram(byref(prog), source, name, 0, None, None)
        if rc != 0:
            raise NvrtcError("nvrtcCreateProgram", rc)
        try:
            arr = (c_char_p * len(options))(*options) if options else None
            rc = self.lib.nvrtcCompileProgram(prog, c_int(len(options)), arr)
            if rc != 0:
                raise NvrtcError("nvrtcCompileProgram", rc, self._log(prog))

            size = c_size_t()
            rc = self.lib.nvrtcGetPTXSize(prog, byref(size))
            if rc != 0:
                raise NvrtcError("nvrtcGetPTXSize", rc, self._log(prog))
            buf = ctypes.create_string_buffer(size.value)
            rc = self.lib.nvrtcGetPTX(prog, buf)
            if rc != 0:
                raise NvrtcError("nvrtcGetPTX", rc, self._log(prog))
            return buf.raw
        finally:
            try:
                self.lib.nvrtcDestroyProgram(byref(prog))
            except Exception:
                pass
