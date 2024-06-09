#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaErrorCheck(err)                                       \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));      \
    exit(EXIT_FAILURE);                                           \
  }

void initArrFloat(float* arr, const int N) {
  // -1.0 ~ 1.0
  // copied from llm.c
  for (int i = 0; i < N; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
  }
}

void initArrInt(int* arr, const int N) {
  for (int i = 0; i < N; i++) {
    arr[i] = rand() & 0x00ff;
  }
}

bool checkResults(float* resCPU, float* resFromGPU, const int N) {
  float tolerance = 1e-4f;
  for (int i = 0; i < N; i++) {
    if (abs(resFromGPU[i] - resCPU[i]) > tolerance) {
      printf("checkResultsError: %f != %f, index: %i\n", resCPU[i],
             resFromGPU[i], i);
      return false;
    }
  }
  return true;
}

template <typename T>
__device__ float warpReduceMax(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

template <typename T>
__device__ float warpReduceSum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename T>
__host__ __device__ inline float ceilDiv(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

template <typename Kernel, typename... Args>
void benchmarkKernel(Kernel kernel, const dim3 gridDim, const dim3 blockDim,
                     float* totalElapsedTime, Args&&... args) {
  cudaEvent_t begin, end;
  cudaErrorCheck(cudaEventCreate(&begin));
  cudaErrorCheck(cudaEventCreate(&end));
  float elapsedTime;
  for (int i = 0; i < 100; ++i) {
    elapsedTime = 0.0f;
    cudaEventRecord(begin);
    kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
    cudaEventRecord(end);
    cudaErrorCheck(cudaEventSynchronize(begin));
    cudaErrorCheck(cudaEventSynchronize(end));
    cudaErrorCheck(cudaEventElapsedTime(&elapsedTime, begin, end));
    *totalElapsedTime += elapsedTime;
  }
  *totalElapsedTime /= 100.0f;
  cudaErrorCheck(cudaEventDestroy(begin));
  cudaErrorCheck(cudaEventDestroy(end));
}

template <typename Kernel, typename... Args>
void benchmarkKernel(Kernel kernel, const int gridSize, const int blockSize,
                     float* totalElapsedTime, Args&&... args) {
  benchmarkKernel(kernel, dim3(gridSize), dim3(blockSize), totalElapsedTime,
                  std::forward<Args>(args)...);
}

#endif
