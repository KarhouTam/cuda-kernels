---
name: cuda-kernel-writer
description: Write CUDA kernels in this educational codebase. Use whenever the user asks to create a new CUDA kernel, implement a GPU operation, or add a new kernel directory. Triggers on requests like "write a kernel for X", "implement Y in CUDA", "add a Z kernel", or any mention of creating new CUDA kernel code in this repo.
---

# CUDA Kernel Writer

This skill guides writing CUDA kernel files in the `cuda-kernels` educational repository. The codebase prioritizes readability and learning over extreme optimization — every kernel should clearly explain what it does and how it works.

## Guiding principles

1. **This is educational code.** Prioritize clear comments, readable variable names, and step-by-step explanations of the algorithm. A student should be able to read the file top-to-bottom and understand what's happening.
2. **Follow the established pattern exactly.** Every kernel file in this repo follows the same structure. Deviating from it creates confusion.
3. **Build up from naive to optimized.** Each kernel variant should introduce one new concept (warp-level parallelism, shared memory, float4 coalescing, etc.) and explain what changed.
4. **Import from `common.h`.** Use the shared helpers — `Package128`, `warpReduceSum`/`warpReduceMax`, `benchmarkKernel`, `checkResults`, `initArrFloat`, `cudaErrorCheck`, `ceilDiv` — instead of reimplementing them.

## File structure

Every kernel file (`kernels/<name>/<name>_forward.cu`) has these sections in order:

### 1. Includes

```c
#include "common.h"
// Add <cooperative_groups.h>, <cub/cub.cuh>, <cassert> only if actually needed
```

### 2. File header comment

A block comment describing the operation, usage, and every kernel variant:

```c
/* <Operation Name> Forward Implementation

Usage: ./<name>_forward <kernel> [blockSize]
e.g. ./<name>_forward 1

<name>_cpu(): CPU reference implementation

<name>_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row/element of the input.

<name>_forward_kernel2(): Optimized implementation on CUDA. Compared to
kernel1, each warp (32 threads) handles one row.

// ... one entry per kernel variant
*/
```

Each kernel variant description says what makes it different from the previous one. Mention the technique used (warp reduce, shared memory tiling, float4 coalescing, Welford's algorithm, etc.) and why it helps.

### 3. Helper defines and device functions (if any)

```c
#define ALPHA 1.0f  // only if the operation has parameters

__device__ __forceinline__ float my_op(float x) {
  // Helper used by both CPU and GPU code
  return ...;
}
```

### 4. CPU reference implementation

Always named `<op>_cpu()` or `<op>_forward_cpu()`. This is the ground truth — every GPU kernel is validated against it.

- Use the same variable naming conventions as the GPU code
- Comment the key algorithm steps (e.g., `// compute mean`, `// softmax normalization`)
- For element-wise ops with M rows and N columns, use the standard M×N loop pattern

```c
void gelu_cpu(float* input, float* output, const int M, const int N) {
  for (int m = 0; m < M; ++m) {
    const float* x = input + m * N;
    float* const y = output + m * N;
    for (int n = 0; n < N; ++n) {
      float xn = x[n];
      float cube = 0.044715f * xn * xn * xn;
      y[n] = 0.5f * xn * (1.0f + tanhf(GELU_SCALING_FACTOR * (xn + cube)));
    }
  }
}
```

### 5. GPU kernels (numbered sequentially)

Each kernel is a `__global__` function named `<op>_forward_kernel<N>()`. Follow this progression:

| Variant | Technique | When to use |
|---------|-----------|-------------|
| kernel1 | Naive — one thread per row | Always include as baseline |
| kernel2 | One warp per row, warp-level reduction | When rows are large enough to benefit |
| kernel3 | Same as kernel2 but using cooperative groups (`cg::reduce`) | For teaching cooperative groups API |
| kernel4 | Shared memory + warp-level processing | When data reuse within a block helps |
| kernel5+ | float4/Package128 coalesced access | When memory bandwidth is the bottleneck |
| kernel6+ | Block-level reduction | When warp-level isn't enough (very wide rows) |
| kernel7+ | Advanced algorithms (Welford, online softmax, etc.) | When numerical stability matters |

Not every kernel needs all variants — pick the ones that make educational sense for the operation. But always start with kernel1 (naive baseline).

#### Kernel comment style

Each kernel gets a short comment at the top explaining its approach:

```c
__global__ void layernorm_forward_kernel2(float* input, float* output,
                                          float* weight, float* bias, float eps,
                                          int B, int T, int C) {
  // one warp one row
  // each thread handles part of a row, then warp-reduces to get the full result
  int warpsPerBlock = blockDim.x / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  ...
```

Inline comments should explain the **why**, not the **what**. Good examples from the codebase:

```c
// one warp (32 threads) process one row
// warp-reduce to calculate the MAX of maxval among all lanes
// use float4 to accelerate memory access
// using formula D(X) = E(X^2) - E(X)^2 to reduce the number of loops
// Welford's online update for numerically stable variance
```

#### Variable naming conventions

| Variable | Meaning |
|----------|---------|
| `idx` / `tid` | Thread index within the grid / block |
| `laneId` | Thread index within a warp (0–31) |
| `warpId` | Which warp within the block |
| `warpsPerBlock` | `blockDim.x / warpSize` |
| `numWarps` | Total warps across all blocks (`gridDim.x * warpsPerBlock`) |
| `row`, `col` | Row and column indices for matrix operations |
| `partialSum` / `partialSum2` | Per-thread accumulator before reduction |
| `laneMax` / `laneSum` / `warpMax` | Values scoped to a lane or warp |
| `inv_std` / `invStd` | Inverse standard deviation (use `rsqrtf`) |

Use `const` and `__restrict__` where appropriate — the existing code uses `const float* const x` for input pointers.

### 6. Dimension macros

Define test dimensions just before `main()`:

```c
#define M 8192        // rows (power of 2 or large round number)
#define N 8192        // columns
#define BLOCK_SIZE 128
#define REPEAT_TIMES 100
```

For multi-dimensional problems (GEMM, attention, layernorm), use `B`, `T`, `C`, `H`, `D`, `K` as appropriate with explicit comments about what each dimension means.

### 7. main() function

Follow this exact sequence. Deviate only when the operation requires different allocation patterns:

```
1. Parse args: kernel number, optional blockSize, optional repeatTimes
2. Allocate host memory (malloc)
3. Initialize with initArrFloat() or zeros()
4. Allocate device memory (cudaMalloc)
5. Copy inputs to device (cudaMemcpy HostToDevice)
6. Run CPU reference
7. Launch selected GPU kernel via switch(kernel)
8. cudaDeviceSynchronize()
9. Copy results back (cudaMemcpy DeviceToHost)
10. If checkResults passes:
    a. Benchmark selected kernel via switch(kernel) using benchmarkKernel()
    b. Print: "<op>_forward kernel: %i | <dimensions> | Times: %f ms | blockSize: %i"
11. Free host memory
12. cudaFree device memory
13. Return EXIT_SUCCESS
```

### 8. CMakeLists.txt

Create `kernels/<name>/CMakeLists.txt` with a single line:

```cmake
add_executable(<name>_forward <name>_forward.cu)
```

### 9. Register in root CMakeLists.txt

Add to the root `CMakeLists.txt`:

```cmake
add_subdirectory(kernels/<name>)
```

## What NOT to do

- Don't strip comments to make the code "cleaner" — comments are the point
- Don't skip kernel variants — start naive, then optimize step by step
- Don't introduce optimization techniques without explaining them
- Don't use single-letter variables except for loop counters (`i`, `k`, `n`, `m`)
- Don't reimplement `warpReduceSum`, `warpReduceMax`, `checkResults`, `ceilDiv`, or `benchmarkKernel` — they're in `common.h`
- Don't change the `main()` structure or the CLI argument convention
- Don't add timing or `#pragma unroll` without commenting why
