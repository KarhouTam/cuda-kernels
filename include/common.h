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

// inspired by llm.c
// Use Package128 to store 128 bits and urge GPU to load it in one time
template <typename T>
struct Package128 {
    constexpr static int size = 16 / sizeof(T);
    T data[size];
    explicit Package128(int4& seq) { load(seq); }

    T& operator[](int idx) { return data[idx]; }

    const T& operator[](int idx) const { return data[idx]; }

    void constant(T val) {
        for (int i = 0; i < size; i++) {
            data[i] = val;
        }
    }
    void zeros() { constant(0); }
    void ones() { constant(1); }

    void load(T& seq) {
        static_assert(16 == sizeof(T) * size);
        memcpy(data, reinterpret_cast<int4*>(&seq), 128);
    }

    int4& getData() { return *reinterpret_cast<int4*>(data); }
};

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
                     unsigned int smemSize, CUstream_st* stream,
                     float* totalElapsedTime, Args&&... args) {
    cudaEvent_t begin, end;
    cudaErrorCheck(cudaEventCreate(&begin));
    cudaErrorCheck(cudaEventCreate(&end));
    float elapsedTime;
    for (int i = 0; i < 100; ++i) {
        elapsedTime = 0.0f;
        cudaEventRecord(begin);
        kernel<<<gridDim, blockDim, smemSize, stream>>>(
            std::forward<Args>(args)...);
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

template <typename T>
__device__ __forceinline__ T plus(const T& a, const T& b) {
    return a + b;
}
template <typename T>
__device__ __forceinline__ T subtract(const T& a, const T& b) {
    return a - b;
}
template <typename T>
__device__ __forceinline__ T multiply(const T& a, const T& b) {
    return a * b;
}
template <typename T>
__device__ __forceinline__ T divide(const T& a, const T& b) {
    return a / b;
}

template <typename Kernel, typename... Args>
void benchmarkKernel(Kernel kernel, const int gridSize, const int blockSize,
                     unsigned int smemSize, CUstream_st* stream,
                     float* totalElapsedTime, Args&&... args) {
    benchmarkKernel(kernel, dim3(gridSize), dim3(blockSize), smemSize, stream,
                    totalElapsedTime, std::forward<Args>(args)...);
}

#endif
