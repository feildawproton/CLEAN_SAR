from typing import Literal
from .cuda_backend import is_cuda_lib_available
from .c_backend import is_c_lib_available

AvailableBackend = Literal["auto", "cuda", "c", "cpu"]


def is_cuda_native_available() -> bool:
    """Checks if the native C++/CUDA backend (NVRTC / Driver API) is available."""
    return is_cuda_lib_available()


def resolve_backend(backend: str) -> str:
    """
    Resolves the requested backend to an active implementation.
    
    Parameters
    ----------
    backend : str
        'auto', 'cuda', 'c', or 'cpu'

    Returns
    -------
    str
        Resolved backend name ('cuda' or 'c')
    """
    backend_lower = backend.lower()
    if backend_lower == "auto":
        return "cuda" if is_cuda_native_available() else "c"
    elif backend_lower == "cuda":
        if not is_cuda_native_available():
            raise NotImplementedError(
                "Native C++/CUDA backend could not be initialized via NVRTC / CUDA Driver API. "
                "Please verify NVIDIA drivers and NVRTC runtime libraries, or use backend='c'."
            )
        return "cuda"
    elif backend_lower in ("c", "cpu"):
        return "c"
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported options are: 'auto', 'cuda', 'c'."
        )
