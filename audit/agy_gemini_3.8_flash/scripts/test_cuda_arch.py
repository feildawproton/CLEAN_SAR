import ctypes
import torch

print("Torch device name:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print(f"Device compute capability: {cap[0]}.{cap[1]} (compute_{cap[0]}{cap[1]})")

from clean_sar.backends.cuda_backend import _init_cuda_driver, _CUDA_KERNELS
ok = _init_cuda_driver()
print("CUDA backend init on this machine:", ok)
print("Loaded kernels:", list(_CUDA_KERNELS.keys()))
