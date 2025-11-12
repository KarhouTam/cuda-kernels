#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <cub/cub.cuh>

#include "common.h"

/* Layer Normalization forward implementation

Usage: ./layernorm_forward <kernel> [blockSize]
e.g. ./layernorm_forward 1

layernorm_forward_cpu(): CPU implementation

layernorm_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row of the input.

layernorm_forward_kernel2(): Optimized implementation on CUDA. Compares to
kernel1, each warp (32 threads) handles one row.

layernorm_forward_kernel3(): Similar to kernel2, each warp handles one row, but
uses CUDA's cooperative groups instead.

layernorm_forward_kernel4(): On the base of kernel2, plus using shared memory to
store the intermediate shift values (x - mean).

layernorm_forward_kernel5(): On the base of kernel2, using formula D(X) = E(X^2)
- E(X)^2 to reduce the number of loops.

layernorm_forward_kernel6(): Block-level reduction implementation using shared memory.
Each block handles one row, with efficient memory access patterns and reduced
synchronization overhead. Uses a two-phase reduction approach with shared memory
for intermediate results.

layernorm_forward_kernel7(): Advanced implementation using Welford's online algorithm
for numerically stable mean and variance computation. Uses hierarchical reduction
with warp-level primitives and shared memory for optimal performance.

*/

void layernorm_forward_cpu(float* input, float* output, float* weight,
                           float* bias, float eps, int B, int T, int C) {
  // In normal, the input data has shape [B, T, C], B is batch size, T is sequence
  // length, C is token length
  for (int row = 0; row < B * T; ++row) {
    float* const x = input + row * C;
    float* const y = output + row * C;
    // mean
    float mean = 0.0f;
    for (int i = 0; i < C; ++i) {
      mean += x[i];
    }
    mean /= C;

    float var = 0.f;
    for (int i = 0; i < C; ++i) {
      float xShift = x[i] - mean;
      var += xShift * xShift;
    }
    float inv_std = 1.0f / sqrt(var / C + eps);

    for (int i = 0; i < C; ++i) {
      y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
    }
  }
}

__global__ void layernorm_forward_kernel1(float* input, float* output,
                                          float* weight, float* bias, float eps,
                                          int B, int T, int C) {
  // naive implementation
  // each thread handle one row of input
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < B * T) {
    float* const x = input + idx * C;
    float* const y = output + idx * C;
    float mean = 0.0f;
    for (int i = 0; i < C; ++i) {
      mean += x[i];
    }

    mean /= C;

    float var = 0.f;
    for (int i = 0; i < C; ++i) {
      float xShift = x[i] - mean;
      var += xShift * xShift;
    }
    float inv_std = 1.0f / sqrt(var / C + eps);

    for (int i = 0; i < C; ++i) {
      y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
    }
  }
}

__global__ void layernorm_forward_kernel2(float* input, float* output,
                                          float* weight, float* bias, float eps,
                                          int B, int T, int C) {
  // one warp one row
  // each thread handle one row of input
  int warpsPerBlock = blockDim.x / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int numWarps = gridDim.x * warpsPerBlock;
  for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T;
       row += numWarps)
    if (row < B * T) {
      float* const x = input + row * C;
      float* const y = output + row * C;
      float sum = 0.0f;
      for (int i = laneId; i < C; i += warpSize) {
        sum += x[i];
      }

      sum = warpReduceSum(sum);
      float mean = sum / C;

      float var = 0.f;
      for (int i = laneId; i < C; i += warpSize) {
        float xShift = x[i] - mean;
        var += xShift * xShift;
      }

      var = warpReduceSum(var);
      float inv_std = 1.0f / sqrt(var / C + eps);

      for (int i = laneId; i < C; i += warpSize) {
        y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
      }
    }
}

__global__ void layernorm_forward_kernel3(float* input, float* output,
                                          float* weight, float* bias, float eps,
                                          int B, int T, int C) {
  // compares to kernel2, use cooperative groups (just for practice)
  // performance is very close to kernel 2
  namespace cg = cooperative_groups;
  cg::thread_block thisBlock = cg::this_thread_block();
  cg::thread_block_tile<32> thisWarp = cg::tiled_partition<32>(thisBlock);
  int warpId = thisWarp.meta_group_rank();
  int warpsPerBlock = thisWarp.meta_group_size();
  int laneId = thisWarp.thread_rank();
  int numWarps = gridDim.x * warpsPerBlock;
  for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T;
       row += numWarps) {
    float* const x = input + row * C;
    float* const y = output + row * C;
    float sum = 0.0f;

    for (int i = laneId; i < C; i += thisWarp.num_threads()) {
      sum += x[i];
    }

    sum = cg::reduce(thisWarp, sum, plus<float>);

    float mean = sum / C;
    float var = 0.f;

    for (int i = laneId; i < C; i += thisWarp.num_threads()) {
      float xShift = x[i] - mean;
      var += xShift * xShift;
    }

    var = cg::reduce(thisWarp, var, plus<float>);

    float inv_std = 1.0f / sqrt(var / C + eps);

    for (int i = laneId; i < C; i += thisWarp.num_threads()) {
      y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
    }
  }
}

__global__ void layernorm_forward_kernel4(float* input, float* output,
                                          float* weight, float* bias, float eps,
                                          int B, int T, int C) {
  // one warp one row, plus using smem to store the shift (x - mean) values
  assert((C % warpSize) == 0);
  extern __shared__ float xShifts[];
  int warpsPerBlock = blockDim.x / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  float* const xShiftsWarp = xShifts + warpId * C;
  int row = blockIdx.x * warpsPerBlock + warpId;
  if (row < B * T) {
    float* const x = input + row * C;
    float* const y = output + row * C;
    float partialSum = 0.0f;
    for (int i = laneId; i < C; i += warpSize) {
      xShiftsWarp[i] = x[i];
      partialSum += x[i];
    }

    float sum = warpReduceSum(partialSum);
    float mean = sum / C;

    float var = 0.f;
    for (int i = laneId; i < C; i += warpSize) {
      xShiftsWarp[i] -= mean;
      var += xShiftsWarp[i] * xShiftsWarp[i];
    }

    var = warpReduceSum(var);
    float inv_std = 1.0f / sqrt(var / C + eps);

    for (int i = laneId; i < C; i += warpSize) {
      y[i] = weight[i] * xShiftsWarp[i] * inv_std + bias[i];
    }
  }
}

__global__ void layernorm_forward_kernel5(float* input, float* output,
                                          float* weight, float* bias, float eps,
                                          int B, int T, int C) {
  // using formula D(X) = E(X^2) - E(X)^2 to reduce the number of loops
  int warpsPerBlock = blockDim.x / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int numWarps = gridDim.x * warpsPerBlock;
  for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T;
       row += numWarps)
    if (row < B * T) {
      float* const x = input + row * C;
      float* const y = output + row * C;
      float partialSum = 0.0f;
      float partialSum2 = 0.0f;
      for (int i = laneId; i < C; i += warpSize) {
        float xi = x[i];
        partialSum += xi;
        partialSum2 += xi * xi;
      }

      float mean = warpReduceSum(partialSum) / C;
      float mean2 = warpReduceSum(partialSum2) / C;

      float var = (mean2 - mean * mean);
      float inv_std = rsqrtf(var + eps);

      for (int i = laneId; i < C; i += warpSize) {
        y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
      }
    }
}

__global__ void layernorm_forward_kernel6(float* __restrict__ input,
                                          float* __restrict__ output,
                                          float* __restrict__ weight,
                                          float* __restrict__ bias, float eps,
                                          int B, int T, int C) {
  // block reduce
  int tid = threadIdx.x;
  int warpId = tid / warpSize;
  int laneId = tid % warpSize;
  int warpsPerBlock = ceilDiv(blockDim.x, warpSize);
  int dataPerWarp = ceilDiv(C, warpsPerBlock);
  int dataPerLane = ceilDiv(dataPerWarp, warpSize);
  int start = dataPerWarp * warpId + dataPerLane * laneId;
  int end = min(start + dataPerLane, C);
  extern __shared__ float sharedMem[];
  float* invStdShared = sharedMem;
  float* meanShared = invStdShared + 1;
  float* xSumShared = meanShared + 1;
  float* xSum2Shared = xSumShared + warpsPerBlock;
  float* xShared = xSum2Shared + warpsPerBlock;
  int row = blockIdx.x;
  float* x = input + row * C;
  float* y = output + row * C;
  float laneSum = 0.f;
  float laneSum2 = 0.f;
  for (int i = start; i < end; ++i) {
    float xi = x[i];
    xShared[i] = xi;
    laneSum += xi;
    laneSum2 += xi * xi;
  }
  float warpSum = warpReduceSum(laneSum);
  float warpSum2 = warpReduceSum(laneSum2);
  if (laneId == 0) {
    xSumShared[warpId] = warpSum;
    xSum2Shared[warpId] = warpSum2;
  }
  __syncthreads();
  if (warpId == 0) {
    float sum = laneId < warpsPerBlock ? xSumShared[laneId] : 0.f;
    float sum2 = laneId < warpsPerBlock ? xSum2Shared[laneId] : 0.f;
    float blockSum = warpReduceSum(sum);
    float blockSum2 = warpReduceSum(sum2);
    float mean = blockSum / C;
    float mean2 = blockSum2 / C;
    float var = mean2 - mean * mean;

    if (laneId == 0) {
      *meanShared = mean;
      *invStdShared = rsqrtf(var + eps);
    }
  }
  __syncthreads();

  float mean = *meanShared;
  float invStd = *invStdShared;
  for (int i = start; i < end; ++i) {
    y[i] = weight[i] * invStd * (xShared[i] - mean) + bias[i];
  }
}

__global__ void layernorm_forward_kernel7(float* __restrict__ input,
                                          float* __restrict__ output,
                                          float* __restrict__ weight,
                                          float* __restrict__ bias, float eps,
                                          int B, int T, int C) {
  const int tid = threadIdx.x;
  const int warpId = tid / warpSize;
  const int laneId = tid % warpSize;
  const int warpsPerBlock = blockDim.x / warpSize;
  const int dataPerWarp = ceilDiv(C, warpsPerBlock);
  const int start = dataPerWarp * warpId;
  const int end = min((warpId + 1) * dataPerWarp, C);

  extern __shared__ float shared[];
  // Shared memory layout: [means, m2s, counts, xData]
  float* means = shared;                            // warpsPerBlock elements
  float* m2s = means + warpsPerBlock;               // warpsPerBlock elements
  int* counts = (int*)(m2s + warpsPerBlock);        // warpsPerBlock elements
  float* xData = (float*)(counts + warpsPerBlock);  // C elements

  const int row = blockIdx.x;
  float* const x = input + row * C;
  float* const y = output + row * C;

  // Initialize Welford's algorithm for this thread
  float mean = 0.0f;
  float m2 = 0.0f;
  int count = 0;

  // Load data and compute local mean/variance
  for (int i = start + laneId; i < end; i += warpSize) {
    float xi = x[i];
    xData[i] = xi;  // Store in shared memory for later use
    count++;

    // Welford's online update
    float delta = xi - mean;
    mean += delta / count;
    float delta2 = xi - mean;
    m2 += delta * delta2;
  }

// Warp reduction for Welford's statistics
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int n1 = count;
    int n2 = __shfl_xor_sync(0xffffffff, n1, offset);
    float mean2 = __shfl_xor_sync(0xffffffff, mean, offset);
    float m22 = __shfl_xor_sync(0xffffffff, m2, offset);

    // Combine parallel Welford
    if (n2 > 0) {
      float delta = mean2 - mean;
      mean = mean + (delta * n2) / (n1 + n2);
      m2 = m2 + m22 + (delta * delta * n1 * n2) / (n1 + n2);
      count = n1 + n2;
    }
  }

  // Store warp results to shared memory
  if (laneId == 0) {
    means[warpId] = mean;
    m2s[warpId] = m2;
    counts[warpId] = count;
  }
  __syncthreads();

  // Block reduction using first warp
  if (warpId == 0) {
    if (laneId < warpsPerBlock) {
      mean = means[laneId];
      m2 = m2s[laneId];
      count = counts[laneId];
    } else {
      mean = 0.0f;
      m2 = 0.0f;
      count = 0;
    }

// Combine results from all warps
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      int n1 = count;
      int n2 = __shfl_xor_sync(0xffffffff, n1, offset);
      float mean2 = __shfl_xor_sync(0xffffffff, mean, offset);
      float m22 = __shfl_xor_sync(0xffffffff, m2, offset);

      if (n2 > 0) {
        float delta = mean2 - mean;
        mean = mean + (delta * n2) / (n1 + n2);
        m2 = m2 + m22 + (delta * delta * n1 * n2) / (n1 + n2);
        count = n1 + n2;
      }
    }

    // Store final results
    if (laneId == 0) {
      means[0] = mean;
      m2s[0] = m2 / C;  // Convert M2 to variance
    }
  }
  __syncthreads();

  // Get final mean and variance
  float finalMean = means[0];
  float variance = m2s[0];
  float invStd = rsqrtf(variance + eps);

  // Apply normalization
  for (int i = start + laneId; i < end; i += warpSize) {
    y[i] = weight[i] * (xData[i] - finalMean) * invStd + bias[i];
  }
}

#define B 8
#define T 1024
#define C 768
#define EPS 1e-5
#define BLOCK_SIZE 128
#define REPEAT_TIMES 100

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: layernorm_forward <kernel> [blockSize] "
            "[benchmarkRepeatTimes]\n");
    return EXIT_FAILURE;
  }
  int kernel = atoi(argv[1]);

  int blockSize = BLOCK_SIZE;
  if (argc > 2) {
    blockSize = atoi(argv[2]);
  }
  int repeatTimes = REPEAT_TIMES;
  if (argc > 3) {
    repeatTimes = atoi(argv[3]);
  }

  float* input = (float*)malloc(B * T * C * sizeof(float));
  float* output = (float*)malloc(B * T * C * sizeof(float));
  float* weight = (float*)malloc(C * sizeof(float));
  float* bias = (float*)malloc(C * sizeof(float));
  float* resFromGPU = (float*)malloc(B * T * C * sizeof(float));
  initArrFloat(input, B * T * C);
  initArrFloat(weight, C);
  initArrFloat(bias, C);
  zeros(output, B * T * C);

  float *inputGPU, *outputGPU, *weightGPU, *biasGPU;

  cudaErrorCheck(cudaMalloc(&inputGPU, B * T * C * sizeof(float)));
  cudaErrorCheck(cudaMemcpy(inputGPU, input, B * T * C * sizeof(float),
                            cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMalloc(&weightGPU, C * sizeof(float)));
  cudaErrorCheck(
      cudaMemcpy(weightGPU, weight, C * sizeof(float), cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMalloc(&biasGPU, C * sizeof(float)));
  cudaErrorCheck(
      cudaMemcpy(biasGPU, bias, C * sizeof(float), cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMalloc(&outputGPU, B * T * C * sizeof(float)));
  cudaErrorCheck(cudaMemset(outputGPU, 0, B * T * C * sizeof(float)));

  float elapsedTime = 0.0f;

  layernorm_forward_cpu(input, output, weight, bias, EPS, B, T, C);

  switch (kernel) {
    case 1:
      layernorm_forward_kernel1<<<B * T / blockSize, blockSize>>>(
          inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
      break;
    case 2:
      layernorm_forward_kernel2<<<B * T * 32 / blockSize, blockSize>>>(
          inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
      break;
    case 3:
      layernorm_forward_kernel3<<<B * T * 32 / blockSize, blockSize>>>(
          inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
      break;
    case 4: {
      int smemSize = C * sizeof(float) * (blockSize / 32);
      layernorm_forward_kernel4<<<B * T * 32 / blockSize, blockSize,
                                  smemSize>>>(inputGPU, outputGPU, weightGPU,
                                              biasGPU, EPS, B, T, C);
      break;
    }
    case 5:
      layernorm_forward_kernel5<<<B * T * 32 / blockSize, blockSize>>>(
          inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
      break;
    case 6: {
      const int smemSize =
          sizeof(float) * (2 + 2 * ceilDiv(blockSize, 32) +
                           C);  // 2 for invStdShared, meanShared
      layernorm_forward_kernel6<<<B * T, blockSize, smemSize>>>(
          inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
      break;
    }
    case 7: {
      const int smemSize =
          sizeof(float) * (2 * ceilDiv(blockSize, 32)) +  // means and m2s
          sizeof(int) * ceilDiv(blockSize, 32) +          // counts
          sizeof(float) * C;  // 2 for invStdShared, meanShared
      layernorm_forward_kernel7<<<B * T, blockSize, smemSize>>>(
          inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
      break;
    }
    default:
      printf("Error: Invalid kernel type: %i\n", kernel);
      return EXIT_FAILURE;
  }
  cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, B * T * C * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cudaErrorCheck(cudaDeviceSynchronize());

  if (checkResults(output, resFromGPU, B * T * C)) {
    switch (kernel) {
      case 1:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel1,
                        B * T / blockSize, blockSize, 0, 0, &elapsedTime,
                        inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
        break;
      case 2:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel2,
                        ceilDiv(B * T * 32, blockSize), blockSize, 0, 0,
                        &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU,
                        EPS, B, T, C);
        break;
      case 3:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel3,
                        ceilDiv(B * T * 32, blockSize), blockSize, 0, 0,
                        &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU,
                        EPS, B, T, C);
        break;
      case 4:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel4,
                        ceilDiv(B * T * 32, blockSize), blockSize,
                        C * sizeof(float) * (blockSize / 32), 0, &elapsedTime,
                        inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
        break;
      case 5:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel5,
                        ceilDiv(B * T * 32, blockSize), blockSize, 0, 0,
                        &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU,
                        EPS, B, T, C);
        break;
      case 6:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel6,
                        ceilDiv(B * T, blockSize), blockSize,
                        sizeof(float) * (2 + 2 * ceilDiv(blockSize, 32) + C), 0,
                        &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU,
                        EPS, B, T, C);
        break;
      case 7:
        benchmarkKernel(repeatTimes, layernorm_forward_kernel7,
                        ceilDiv(B * T, blockSize), blockSize,
                        sizeof(float) * (2 * ceilDiv(blockSize, 32)) +
                            sizeof(int) * ceilDiv(blockSize, 32) +
                            sizeof(float) * C,
                        0, &elapsedTime, inputGPU, outputGPU, weightGPU,
                        biasGPU, EPS, B, T, C);
        break;
    }
    printf(
        "layer_norm_forward kernel: %i | matrixSize: %i x %i x %i | "
        "Times: "
        "%f ms | "
        "blockSize: %i\n",
        kernel, B, T, C, elapsedTime, blockSize);
  }
  free(input);
  free(weight);
  free(bias);
  free(output);
  free(resFromGPU);
  cudaErrorCheck(cudaFree(inputGPU));
  cudaErrorCheck(cudaFree(weightGPU));
  cudaErrorCheck(cudaFree(biasGPU));
  cudaErrorCheck(cudaFree(outputGPU));
  return EXIT_SUCCESS;
}