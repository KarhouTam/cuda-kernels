#include "common.h"

/* ReLU forward implementation

Usage: ./relu_forward <kernel> [blockSize]
e.g. ./relu_forward 1

relu_forward_cpu(): CPU implementation

relu_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row of the input.

relu_forward_kernel2(): Optimized implementation on CUDA. Compares to
kernel1, each warp (32 threads) handles one row.

*/

void relu_cpu(float* input, float* output, const int M, const int N) {
    for (int m = 0; m < M; ++m) {
        const float* x = input + m * N;
        float* const y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = x[n] > 0.0f ? x[n] : 0.0f;
        }
    }
}

__global__ void relu_forward_kernel1(const float* input, float* output,
                                     const int M, const int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M) {
        const float* x = input + idx * N;
        float* const y = output + idx * N;
        for (int n = 0; n < N; ++n) {
            y[n] = x[n] > 0.0f ? x[n] : 0.0f;
        }
    }
}

__global__ void relu_forward_kernel2(float* input, float* output, int M,
                                     int N) {
    // each warp handles one row of the input
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < M;
         row += numWarps)
        if (row < M) {
            float* const x = input + row * N;
            float* const y = output + row * N;

            for (int i = laneId; i < N; i += warpSize) {
                y[i] = x[i] > 0 ? x[i] : 0.0f;
            }
        }
}

__global__ void relu_forward_kernel3(float* input, float* output, int M,
                                     int N) {
    // each warp handles one row of the input
    // use floar4 to acclerate memory accessing
    // but seems improvement is not significant
    using f128 = Package128<float>;
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < M;
         row += numWarps)
        if (row < M) {
            float* const x = input + row * N;
            float* const y = output + row * N;

            for (int i = laneId * f128::size; i < N; i += warpSize * f128::size) {
                f128 packedX = load128(x + i);
                f128 out;
                #pragma unroll
                for (int k = 0; k < f128::size; ++k) {
                    out[k] = packedX[k] > 0 ? packedX[k] : 0.0f;
                }
                store128(y + i, out);
            }
        }
}

#define M 8196
#define N 8196
#define BLOCK_SIZE 128

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: relu_forward <kernel> [blockSize]\n");
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

    relu_cpu(input, output, M, N);

    switch (kernel) {
        case 1:
            relu_forward_kernel1<<<M * N / blockSize, blockSize>>>(
                inputGPU, outputGPU, M, N);
            break;
        case 2:
            relu_forward_kernel2<<<M * N / blockSize, blockSize>>>(
                inputGPU, outputGPU, M, N);
            break;
        case 3:
            relu_forward_kernel3<<<M * N / blockSize, blockSize>>>(
                inputGPU, outputGPU, M, N);
            break;
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }
    cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDeviceSynchronize());

    if (checkResults(output, resFromGPU, M * N)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(relu_forward_kernel1, M * N / blockSize,
                                blockSize, 0, 0, &elapsedTime, inputGPU,
                                outputGPU, M, N);
                break;
            case 2:
                benchmarkKernel(relu_forward_kernel2, M * N / blockSize,
                                blockSize, 0, 0, &elapsedTime, inputGPU,
                                outputGPU, M, N);
                break;
            case 3:
                benchmarkKernel(relu_forward_kernel3, M * N / blockSize,
                                blockSize, 0, 0, &elapsedTime, inputGPU,
                                outputGPU, M, N);
                break;
        }
        printf(
            "softmax_forward kernel: %i | matrixSize: %i x %i | Times: "
            "%f ms | "
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