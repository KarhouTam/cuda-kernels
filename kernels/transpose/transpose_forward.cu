#include <cassert>

#include "common.h"

/* Matrix Transpose Implementation

Usage: ./transpose_forward <kernel> [blockSize]
e.g. ./transpose_forward 1

transpose_forward_cpu(): CPU implementation of matrix transpose

transpose_forward_kernel1(): Basic CUDA implementation of matrix transpose.
Each thread handles one element of the matrix.
*/

void transpose_forward_cpu(float* input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

__global__ void transpose_forward_kernel1(float* input, float* output, int M, int N) {
    // Basic implementation - each thread handles one element
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N;
        int j = idx % N;
        output[j * M + i] = input[idx];
    }
}

#define M 1024
#define N 1024
#define BLOCK_SIZE 256
#define REPEAT_TIMES 100

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: transpose_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
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

    // Allocate host memory
    float* input = (float*)malloc(M * N * sizeof(float));
    float* output = (float*)malloc(M * N * sizeof(float));
    float* resFromGPU = (float*)malloc(M * N * sizeof(float));

    // Initialize input data
    initArrFloat(input, M * N);
    zeros(output, M * N);
    zeros(resFromGPU, M * N);

    // Allocate device memory
    float *inputGPU, *outputGPU;
    cudaErrorCheck(cudaMalloc(&inputGPU, M * N * sizeof(float)));
    cudaErrorCheck(cudaMalloc(&outputGPU, M * N * sizeof(float)));

    // Copy input to device
    cudaErrorCheck(cudaMemcpy(inputGPU, input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    float elapsedTime = 0.0f;

    // Run CPU version for reference
    transpose_forward_cpu(input, output, M, N);

    // Run selected kernel
    switch (kernel) {
        case 1: {
            int gridSize = (M * N + blockSize - 1) / blockSize;
            transpose_forward_kernel1<<<gridSize, blockSize>>>(inputGPU, outputGPU, M, N);
            break;
        }
        default:
            printf("Error: Invalid kernel type: %d\n", kernel);
            return EXIT_FAILURE;
    }

    // Copy result back to host
    cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDeviceSynchronize());

    // Verify results and benchmark
    if (checkResults(output, resFromGPU, M * N)) {
        switch (kernel) {
            case 1: {
                int gridSize = (M * N + blockSize - 1) / blockSize;
                benchmarkKernel(repeatTimes, transpose_forward_kernel1, gridSize, blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            }
        }
        printf("transpose_forward kernel: %d | matrixSize: %d x %d | Times: %f ms | blockSize: %d\n",
               kernel, M, N, elapsedTime, blockSize);
    }

    // Cleanup
    free(input);
    free(output);
    free(resFromGPU);
    cudaErrorCheck(cudaFree(inputGPU));
    cudaErrorCheck(cudaFree(outputGPU));

    return EXIT_SUCCESS;
}
