#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "common.h"

void softmax_cpu(float* input, float* output, const int M, const int N) {
    for (int m = 0; m < M; ++m) {
        float maxval = -INFINITY;
        const float* x = input + m * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        float s = 0.0f;
        for (int n = 0; n < N; ++n) {
            s += exp(x[n] - maxval);
        }
        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / s;
        }
    }
}

void online_softmax_cpu(float* input, float* output, const int M, const int N) {
    for (int m = 0; m < M; ++m) {
        const float* x = input + m * N;
        float maxval = -INFINITY;
        float s = 0.0f;
        for (int n = 0; n < N; ++n) {
            if (maxval < x[n]) {
                s *= exp(maxval - x[n]);
                maxval = x[n];
            }
            s += exp(x[n] - maxval);
        }

        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / s;
        }
    }
}

__global__ void softmax_kernel1(float* input, float* output, const int M,
                                const int N) {
    // naive implementation
    // one thread one row
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;
    if (idx < M) {
        float maxval = -INFINITY;
        const float* x = input + idx * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        float s = 0.0f;
        for (int n = 0; n < N; ++n) {
            s += exp(x[n] - maxval);
        }
        float* const y = output + idx * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / s;
        }
    }
}

__global__ void softmax_kernel2(float* input, float* output, const int M,
                                const int N) {
    // use more threads per row than kernel1
    // one warp (32 threads) process one row
    // use warp reduce functions
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;
    const int numWarps = warpsPerBlock * gridDim.x;
    const int idx = warpsPerBlock * blockIdx.x + warpId;
    for (int m = idx; m < M; m += numWarps) {
        // each lane (thread in a warp) calculate the maxval among
        // data with indices [landId, landId + 32, laneId + 64, ...]
        const float* x = input + m * N;
        float* const y = output + m * N;

        float maxval = -INFINITY;
        for (int i = laneId; i < N; i += warpSize) {
            maxval = fmaxf(maxval, x[i]);
        }
        // warp-reduce to calculate the MAX of maxval among all lanes
        // and the 0-th lane will store the result
        maxval = warpReduceMax(maxval);

        float sum = 0.0f;
        for (int i = laneId; i < N; i += warpSize) {
            sum += expf(x[i] - maxval);
        }

        sum = warpReduceSum(sum);
        for (int i = laneId; i < N; i += warpSize) {
            y[i] = expf(x[i] - maxval) / sum;
        }
    }
}

__global__ void online_softmax_kernel3(float* input, float* output, const int M,
                                       const int N) {
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int warpsPerBlock = blockDim.x / warpSize;
    const int numWarps = warpsPerBlock * gridDim.x;
    const int idx = warpsPerBlock * blockIdx.x + warpId;
    for (int m = idx; m < M; m += numWarps) {
        const float* x = input + m * N;
        float* const y = output + m * N;
        float maxval = -INFINITY, sum = 0.0f, bigger;
        for (int i = laneId; i < N; i += warpSize) {
            bigger = fmaxf(maxval, x[i]);
            sum = sum * expf(maxval - bigger) + expf(x[i] - bigger);
        }

        float offsetMax, offsetSum;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            __syncwarp();
            offsetMax = __shfl_xor_sync(0xFFFFFFFF, maxval, offset);
            offsetSum = __shfl_xor_sync(0xFFFFFFFF, sum, offset);
            if (offsetMax > maxval) {
                sum *= expf(maxval - offsetMax);
                maxval = offsetMax;
            } else {
                offsetSum *= expf(offsetMax - maxval);
            }
            sum += offsetSum;
        }
        for (int i = laneId; i < N; i += warpSize) {
            y[i] = expf(x[i] - maxval) / sum;
        }
    }
}

#define M 8196
#define N 8196
#define BLOCK_SIZE 128

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: softmax_forward <kernel> [blockSize]\n");
        return EXIT_FAILURE;
    }
    int kernel = atoi(argv[1]);

    int blockSize = BLOCK_SIZE;
    if (argc > 2) {
        blockSize = atoi(argv[2]);
    }

    float* input = (float*)malloc(M * N * sizeof(float));
    float* output = (float*)malloc(M * N * sizeof(float));
    float* resFromGPU = (float*)malloc(M * N * sizeof(float));
    initArrFloat(input, M * N);

    float *inputGPU, *outputGPU;
    cudaErrorCheck(cudaMalloc(&inputGPU, M * N * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(inputGPU, input, M * N * sizeof(float),
                              cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&outputGPU, M * N * sizeof(float)));

    float elapsedTime;
    online_softmax_cpu(input, output, M, N);

    switch (kernel) {
        case 1:
            benchmark_kernel(softmax_kernel1, M * N / blockSize, blockSize,
                             inputGPU, outputGPU, resFromGPU, M, N,
                             &elapsedTime);
            break;
        case 2:
            benchmark_kernel(softmax_kernel2, ceilDiv(M * 32, blockSize),
                             blockSize, inputGPU, outputGPU, resFromGPU, M, N,
                             &elapsedTime);
            break;
        case 3:
            benchmark_kernel(online_softmax_kernel3, ceilDiv(M * 32, blockSize),
                             blockSize, inputGPU, outputGPU, resFromGPU, M, N,
                             &elapsedTime);
            break;
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }

    if (checkResults(output, resFromGPU, M * N)) {
        printf(
            "softmax_forward kernel: %i | matrixSize: %i x %i | Times: %f ms | "
            "blockSize: %i\n",
            kernel, M, N, elapsedTime, blockSize);
    }
    free(input);
    free(output);
    free(resFromGPU);
    cudaErrorCheck(cudaFree(inputGPU));
    cudaErrorCheck(cudaFree(outputGPU));
    return EXIT_SUCCESS;
}