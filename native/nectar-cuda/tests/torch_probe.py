"""A small PyTorch workload: finds the GPU, multiplies two matrices with cuBLAS, runs an
elementwise kernel, and checks both against the CPU. Run it on the real driver, through the
tracer, and through Nectar.

NECTAR_CUDA_PRELOAD names an nvcuda.dll (Nectar's or the tracer's) to load before torch.
Windows then hands that module to the CUDA runtime's own load of nvcuda.dll. This is how
Python is wired up, since the Store's python.exe runs the interpreter through a broker that
nectar-cuda-run cannot follow."""

import ctypes
import os
import time

if os.environ.get("NECTAR_CUDA_PRELOAD"):
    ctypes.WinDLL(os.environ["NECTAR_CUDA_PRELOAD"])

import torch  # noqa: E402

print("torch     ", torch.__version__, "CUDA", torch.version.cuda)
print("available ", torch.cuda.is_available())
print("device    ", torch.cuda.get_device_name(0))

started = time.perf_counter()
a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)
expected = a @ b

ga = a.cuda()
gb = b.cuda()
product = ga @ gb
total = (ga + gb).sum()
torch.cuda.synchronize()

error = (product.cpu() - expected).abs().max().item()
total_error = abs(total.item() - (a + b).sum().item())
print(f"matmul     max error {error:.2e}")
print(f"add + sum  error {total_error:.2e}")
print(f"elapsed    {time.perf_counter() - started:.2f} s")
print("PASS" if error < 1e-2 and total_error < 1.0 else "FAIL")
