# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This is a CMake + CUDA project. Each kernel is a standalone executable.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Binaries are placed under `build/kernels/<name>/<name>_forward`. Run a kernel by selecting its variant by number:

```bash
./build/kernels/softmax/softmax_forward 1          # run kernel variant 1
./build/kernels/softmax/softmax_forward 2 256 200  # kernel 2, blockSize=256, repeatTimes=200
./build/kernels/gemm/gemm_forward 1
```

There is no test framework. Each `.cu` file is self-contained: it includes a CPU reference implementation, runs the GPU kernel, compares results with `checkResults()`, then benchmarks the kernel.

Build flags are set globally in the root `CMakeLists.txt`:
- `-O3 --use_fast_math -Wno-deprecated-gpu-targets`

## Project Architecture

```
include/common.h          # Shared helpers: error checking macros, Package128<T>, warp reduce,
                          # ceilDiv, benchmarkKernel, initArrFloat, checkResults
kernels/<name>/
  CMakeLists.txt          # add_executable(name_forward name_forward.cu)
  <name>_forward.cu       # CPU ref + GPU kernel(s) + main() — fully self-contained
profile_example/          # PyTorch profiler comparison (uses torch.utils.cpp_extension.load)
```

Each kernel file follows a consistent pattern:
1. File header comment describing variants and usage
2. CPU reference implementation
3. One or more `__global__` kernel functions (increasingly optimized)
4. `main()` that parses args, allocates host/device memory, runs the selected kernel, validates against CPU, then benchmarks via `benchmarkKernel()` from `common.h`

## Common Helpers (`include/common.h`)

- **`Package128<T>`** — 128-bit (float4) aligned type for coalesced memory access. Use `load128`/`load128cs`/`store128`/`store128cs` to read/write.
- **`warpReduceMax`/`warpReduceSum`** — warp-level reductions using `__shfl_xor_sync`.
- **`benchmarkKernel()`** — CUDA event-based timing loop. Accepts `dim3` grid/block or scalar grid/block sizes.
- **`checkResults()`** — element-wise comparison with `1e-4` tolerance, handles NaN.
- **`initArrFloat()`** — random initialization in `[-1.0, 1.0]`.
- **`cudaErrorCheck`/`cublasErrorCheck`** — macros that print file:line and exit on error.

## Formatting

Google C++ style via clang-format (`.clang-format`), 2-space indent, 80-column limit. Pre-commit hook (`.pre-commit-config.yaml`) auto-formats `.c`/`.h`/`.cu`/`.cuh` files.

## Adding a New Kernel

1. Create `kernels/<name>/` directory
2. Add `CMakeLists.txt` with `add_executable(<name>_forward <name>_forward.cu)`
3. Create `<name>_forward.cu` following the established pattern (CPU ref → kernels → main with benchmark)
4. Add `add_subdirectory(kernels/<name>)` to root `CMakeLists.txt`
