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
#define cublasErrorCheck(err)                                                   \
    if (err != CUBLAS_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "cublas error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                     \
    }
// inspired by llm.c
// Use Package128 to store 128 bits and urge GPU to load it in one time
template <typename T>
struct __align__(16) Package128 {
    static constexpr const int size = sizeof(float4) / sizeof(T);
    T data[size];
    Package128() = default;
    __device__ explicit Package128(float4 bits) {
        static_assert(16 == sizeof(T) * size);
        memcpy(&data, &bits, 128);
    }

    __device__ T& operator[](int idx) { return data[idx]; }

    __device__ const T& operator[](int idx) const { return data[idx]; }

    __device__ static Package128<T> constant(T val) {
        Package128<T> results;
        for (int i = 0; i < size; i++) {
            results[i] = val;
        }
        return results;
    }
    __device__ static Package128<T> zeros() { return constant((T)0); }
    __device__ static Package128<T> ones() { return constant((T)1); }
    __device__ float4 getBits() {
        float4 bits;
        static_assert(sizeof(float4) == sizeof(T) * size);
        memcpy(&bits, &data, sizeof(bits));
        return bits;
    }
};

template <typename T>
__device__ inline Package128<T> load128(const T* address) {
    return Package128<T>{*reinterpret_cast<const float4*>(address)};
}

template <typename T>
__device__ inline Package128<T> load128cs(const T* address) {
    return Package128<T>{__ldcs(reinterpret_cast<const float4*>(address))};
}

template <typename T>
__device__ void inline store128(T* address, Package128<T> val) {
    *reinterpret_cast<float4*>(address) = val.getBits();
}

template <typename T>
__device__ void inline store128cs(T* address, Package128<T> val) {
    __stcs(reinterpret_cast<float4*>(address), val.getBits());
}

template <typename T>
void constant(T* arr, const int N, T val) {
    for (int i = 0; i < N; ++i) {
        arr[i] = val;
    }
}

template <typename T>
void ones(T* arr, const int N) {
    constant(arr, N, (T)1);
}

template <typename T>
void zeros(T* arr, const int N) {
    constant(arr, N, (T)1);
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
            printf("checkResultsError: %f != %f, index: %i\n", resCPU[i], resFromGPU[i],
                   i);
            return false;
        }
    }
    return true;
}

template <typename T>
__device__ float warpReduceMax(T val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template <typename T>
__device__ float warpReduceSum(T val) {
#pragma unroll
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
void benchmarkKernel(int repeatTimes, Kernel kernel, const dim3 gridDim,
                     const dim3 blockDim, unsigned int smemSize, CUstream_st* stream,
                     float* totalElapsedTime, Args&&... args) {
    cudaEvent_t begin, end;
    cudaErrorCheck(cudaEventCreate(&begin));
    cudaErrorCheck(cudaEventCreate(&end));
    float elapsedTime;
    for (int i = 0; i < repeatTimes; ++i) {
        elapsedTime = 0.0f;
        cudaEventRecord(begin);
        kernel<<<gridDim, blockDim, smemSize, stream>>>(std::forward<Args>(args)...);
        cudaEventRecord(end);
        cudaErrorCheck(cudaEventSynchronize(begin));
        cudaErrorCheck(cudaEventSynchronize(end));
        cudaErrorCheck(cudaEventElapsedTime(&elapsedTime, begin, end));
        *totalElapsedTime += elapsedTime;
    }
    *totalElapsedTime /= (float)repeatTimes;
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
void benchmarkKernel(int repeatTimes, Kernel kernel, const int gridSize,
                     const int blockSize, unsigned int smemSize, CUstream_st* stream,
                     float* totalElapsedTime, Args&&... args) {
    benchmarkKernel(repeatTimes, kernel, dim3(gridSize), dim3(blockSize), smemSize,
                    stream, totalElapsedTime, std::forward<Args>(args)...);
}

#endif
