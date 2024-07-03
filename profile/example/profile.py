"""
use torch.autograd.profiler to compare performance of kernels in this project and pytorch's
"""

from functools import partial

import torch
from torch.utils.cpp_extension import load


data = torch.randn(8196, 8196).cuda()


customs = load(
    name="customs",
    sources=["bind.cpp", "softmax.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


def count_elapsed_time(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    for _ in range(100):
        func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 100


class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return customs.softmax(input)


def custom_softmax(input):
    CustomSoftmax.apply(input)


torch_softmax_elapsed_time = count_elapsed_time(partial(torch.softmax, dim=1), data)
custom_softmax_elapsed_time = count_elapsed_time(custom_softmax, data)

print(f"torch.softmax: Elapsed Time: {torch_softmax_elapsed_time:.4f} ms")

print(f"customs.softmax: Elapsed Time: {custom_softmax_elapsed_time:.4f} ms")

# profile the native softmax of pytorch
print("=" * 60, "PyTorch Native Softmax", "=" * 60)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.softmax(data, dim=1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# profile our custom softmax kernel
print("=" * 60, "Custom Softmax", "=" * 60)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    customs.softmax(data)
    # custom_softmax(data)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
