#include "common.h"

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
            benchmark_kernel(relu_forward_kernel1, M * N / blockSize, blockSize,
                             inputGPU, outputGPU, resFromGPU, M, N,
                             &elapsedTime);
            break;
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }

    if (checkResults(output, resFromGPU, M * N)) {
        printf(
            "relu_forward kernel: %i | matrixSize: %i x %i | Times: %f ms | "
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