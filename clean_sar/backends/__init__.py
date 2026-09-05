from typing import Literal
from .cuda_backend import is_cuda_lib_available

AvailableBackend = Literal["auto", "pytorch", "cuda"]


def is_cuda_native_available() -> bool:
    """
    Checks if the compiled native C++/CUDA shared library (libcleansar.so) is available.
    """
    return is_cuda_lib_available()


def resolve_backend(backend: str) -> str:
    """
    Resolves the requested backend to an active implementation.
    
    Parameters
    ----------
    backend : str
        'auto', 'pytorch', or 'cuda'

    Returns
    -------
    str
        Resolved backend name ('pytorch' or 'cuda')
    """
    backend_lower = backend.lower()
    if backend_lower == "auto":
        return "cuda" if is_cuda_native_available() else "pytorch"
    elif backend_lower == "pytorch":
        return "pytorch"
    elif backend_lower == "cuda":
        if not is_cuda_native_available():
            raise NotImplementedError(
                "Native C++/CUDA backend (libcleansar.so) is not compiled or not found. "
                "Please use backend='pytorch' or compile the CUDA library in clean_sar/backends/c_src/."
            )
        return "cuda"
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported options are: 'auto', 'pytorch', 'cuda'."
        )
