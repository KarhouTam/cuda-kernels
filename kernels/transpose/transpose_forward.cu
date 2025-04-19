#include <cassert>

#include "common.h"

/* Matrix Transpose Implementation

Usage: ./transpose_forward <kernel> [blockSize]
e.g. ./transpose_forward 1

transpose_forward_cpu(): CPU implementation of matrix transpose

transpose_forward_kernel1(): Basic CUDA implementation of matrix transpose.
Each thread handles one element of the matrix.

transpose_forward_kernel2(): CUDA implementation using shared memory.
Each thread loads one element into shared memory, synchronizes, and writes it back.
This helps coalesce memory accesses but shared memory bank conflicts may occur.

transpose_forward_kernel3(): Optimized CUDA implementation with swizzling.
Similar to kernel2 but uses swizzled thread indices to reduce shared memory bank conflicts.
This improves performance by better utilizing shared memory bandwidth.
*/

void transpose_forward_cpu(float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            output[col * M + row] = input[row * N + col];
        }
    }
}

__global__ void transpose_forward_kernel1(float* input, float* output, int M, int N) {
    // Basic implementation - each thread handles one element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int n_row = blockIdx.x * blockDim.x + threadIdx.x;
        int n_col = blockIdx.y * blockDim.y + threadIdx.y;
        if (n_col < M && n_row < N) {
            output[n_row * M + n_col] = input[row * N + col];
        }
    }
}

__global__ void transpose_forward_kernel2(float* input, float* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        extern __shared__ float smem[];
        smem[tid] = input[row * N + col];
        __syncthreads();
        int n_row = blockIdx.x * blockDim.x + threadIdx.x;
        int n_col = blockIdx.y * blockDim.y + threadIdx.y;
        if (n_col < M && n_row < N) {
            output[n_row * M + n_col] = smem[tid];
        }
    }
}

__global__ void transpose_forward_kernel3(float* input, float* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        extern __shared__ float smem[];
        int swizzling_tid = threadIdx.y * blockDim.x + (threadIdx.y ^ threadIdx.x);
        smem[swizzling_tid] = input[row * N + col];
        __syncthreads();
        int n_row = blockIdx.x * blockDim.x + threadIdx.x;
        int n_col = blockIdx.y * blockDim.y + threadIdx.y;
        if (n_col < M && n_row < N) {
            output[n_row * M + n_col] = smem[swizzling_tid];
        }
    }
}

#define M 1024
#define N 1024
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define REPEAT_TIMES 100

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: transpose_forward <kernel> [blockDimX blockDimY] [benchmarkRepeatTimes]\n");
        return EXIT_FAILURE;
    }

    int kernel = atoi(argv[1]);
    int blockDimX = BLOCK_DIM_X;
    int blockDimY = BLOCK_DIM_Y;
    if (argc > 3) {
        blockDimX = atoi(argv[2]);
        blockDimY = atoi(argv[3]);
    }
    int repeatTimes = REPEAT_TIMES;
    if (argc > 4) {
        repeatTimes = atoi(argv[4]);
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
    dim3 block(blockDimX, blockDimY);
    dim3 grid(ceilDiv(N, block.x), ceilDiv(M, block.y));
    size_t smemSize = block.y * block.x * sizeof(float);
    // Run selected kernel
    switch (kernel) {
        case 1: {
            transpose_forward_kernel1<<<grid, block, smemSize>>>(inputGPU, outputGPU, M, N);
            break;
        }
        case 2: {
            transpose_forward_kernel2<<<grid, block, smemSize>>>(inputGPU, outputGPU, M, N);
            break;
        }
        case 3: {
            transpose_forward_kernel3<<<grid, block, smemSize>>>(inputGPU, outputGPU, M, N);
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
                benchmarkKernel(repeatTimes, transpose_forward_kernel1, grid, block, smemSize, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            }
            case 2: {
                benchmarkKernel(repeatTimes, transpose_forward_kernel2, grid, block, smemSize, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            }
            case 3: {
                benchmarkKernel(repeatTimes, transpose_forward_kernel3, grid, block, smemSize, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            }
        }
        printf(
            "transpose_forward kernel: %d | matrixSize: %d x %d | Times: %f ms | blockDim: (%d, %d, %d)\n",
            kernel, M, N, elapsedTime, block.x, block.y, block.z);
    }

    // Cleanup
    free(input);
    free(output);
    free(resFromGPU);
    cudaErrorCheck(cudaFree(inputGPU));
    cudaErrorCheck(cudaFree(outputGPU));

    return EXIT_SUCCESS;
}
