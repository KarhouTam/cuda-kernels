#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaErrorCheck(err)                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                           \
    }

void initArrFloat(float* arr, const int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

void initArrInt(int* arr, const int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = (rand() & 0xffff) / 1000;
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

__device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename T>
__host__ __device__ inline float ceilDiv(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

template <typename Kernel>
void benchmark_kernel(Kernel kernel, const int gridSize, const int blockSize,
                      float* inputGPU, float* outputGPU, float* resFromGPU,
                      const int M, const int N, float* totalElapsedTime) {
    cudaEvent_t begin, end;
    cudaErrorCheck(cudaEventCreate(&begin));
    cudaErrorCheck(cudaEventCreate(&end));
    float elapsedTime;
    for (int i = 0; i < 100; ++i) {
        elapsedTime = 0.0f;
        cudaEventRecord(begin);
        kernel<<<gridSize, blockSize>>>(inputGPU, outputGPU, M, N);
        cudaEventRecord(end);
        cudaErrorCheck(cudaEventSynchronize(begin));
        cudaErrorCheck(cudaEventSynchronize(end));
        cudaErrorCheck(cudaEventElapsedTime(&elapsedTime, begin, end));
        *totalElapsedTime += elapsedTime;
        if (i == 0)
            cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU,
                                      M * N * sizeof(float),
                                      cudaMemcpyDeviceToHost));
    }
    *totalElapsedTime /= 100.0f;
    cudaErrorCheck(cudaEventDestroy(begin));
    cudaErrorCheck(cudaEventDestroy(end));
}

#endif
