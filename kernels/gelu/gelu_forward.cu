#include "common.h"

/* ReLU forward implementation

Usage: ./gelu_forward <kernel> [blockSize]
e.g. ./gelu_forward 1

gelu_forward_cpu(): CPU implementation

gelu_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row of the input.

gelu_forward_kernel2(): Optimized implementation on CUDA. Compares to
kernel1, each warp (32 threads) handles one row.

gelu_forward_kernel3(): Optimized implementation on CUDA. Compares to
kernel2, using float4.

gelu_forward_kernel4(): Optimized implementation on CUDA. Each thread handles one FLOAT4.
*/
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_cpu(float* input, float* output, const int M, const int N) {
    for (int m = 0; m < M; ++m) {
        const float* x = input + m * N;
        float* const y = output + m * N;
        for (int n = 0; n < N; ++n) {
            float xn = x[n];
            float cube = 0.044715f * xn * xn * xn;
            y[n] = 0.5f * xn * (1.0f + tanhf(GELU_SCALING_FACTOR * (xn + cube)));
        }
    }
}

__global__ void gelu_forward_kernel1(const float* input, float* output, const int M, const int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M) {
        const float* x = input + idx * N;
        float* const y = output + idx * N;
        for (int n = 0; n < N; ++n) {
            float xn = x[n];
            float cube = 0.044715f * xn * xn * xn;
            y[n] = 0.5f * xn * (1.0f + tanhf(GELU_SCALING_FACTOR * (xn + cube)));
        }
    }
}

__global__ void gelu_forward_kernel2(float* input, float* output, int M, int N) {
    // each warp handles one row of the input
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < M; row += numWarps)
        if (row < M) {
            float* const x = input + row * N;
            float* const y = output + row * N;

            for (int i = laneId; i < N; i += warpSize) {
                float xi = x[i];
                float cube = 0.044715f * xi * xi * xi;
                y[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
            }
        }
}

__global__ void gelu_forward_kernel3(float* input, float* output, int M, int N) {
    // each warp handles one row of the input
    // use floar4 to acclerate memory accessing
    // but seems improvement is not significant
    using f128 = Package128<float>;
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < M; row += numWarps)
        if (row < M) {
            float* const x = input + row * N;
            float* const y = output + row * N;

            for (int i = laneId * f128::size; i < N; i += warpSize * f128::size) {
                f128 packedX = load128(x + i);
                f128 out;
#pragma unroll
                for (int k = 0; k < f128::size; ++k) {
                    float xik = packedX[k];
                    float cube = 0.044715f * xik * xik * xik;
                    out[k] = 0.5f * xik * (1.0f + tanhf(GELU_SCALING_FACTOR * (xik + cube)));
                }
                store128(y + i, out);
            }
        }
}

__global__ void gelu_forward_kernel4(float* input, float* output, int M, int N) {
    using f128 = Package128<float>;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    if (idx + f128::size < M * N) {
        f128 packedX = load128cs(input + idx);
        f128 packedY;
        for (int k = 0; k < f128::size; ++k) {
            float xik = packedX[k];
            float cube = 0.044715f * xik * xik * xik;
            packedY[k] = 0.5f * xik * (1.0f + tanhf(GELU_SCALING_FACTOR * (xik + cube)));
        }
        store128(output + idx, packedY);
    } else {
        for (int i = idx; i < M * N; ++i) {
            float xi = input[i];
            float cube = 0.044715f * xi * xi * xi;
            output[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
        }
    }
}

#define M 8192
#define N 8192
#define BLOCK_SIZE 128
#define REPEAT_TIMES 100

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: gelu_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
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

    float* input = (float*)malloc(M * N * sizeof(float));
    float* output = (float*)malloc(M * N * sizeof(float));
    float* resFromGPU = (float*)malloc(M * N * sizeof(float));
    initArrFloat(input, M * N);

    float *inputGPU, *outputGPU;
    cudaErrorCheck(cudaMalloc(&inputGPU, M * N * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(inputGPU, input, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMalloc(&outputGPU, M * N * sizeof(float)));

    float elapsedTime = 0.0f;

    gelu_cpu(input, output, M, N);

    switch (kernel) {
        case 1:
            gelu_forward_kernel1<<<M * N / blockSize, blockSize>>>(inputGPU, outputGPU, M, N);
            break;
        case 2:
            gelu_forward_kernel2<<<M * N / blockSize, blockSize>>>(inputGPU, outputGPU, M, N);
            break;
        case 3:
            gelu_forward_kernel3<<<M * N / blockSize, blockSize>>>(inputGPU, outputGPU, M, N);
            break;
        case 4:
            gelu_forward_kernel4<<<ceilDiv(M * N, (blockSize * Package128<float>::size)), blockSize>>>(
                inputGPU, outputGPU, M, N);
            break;
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }
    cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDeviceSynchronize());

    if (checkResults(output, resFromGPU, M * N)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(repeatTimes, gelu_forward_kernel1, M * N / blockSize, blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            case 2:
                benchmarkKernel(repeatTimes, gelu_forward_kernel2, M * N / blockSize, blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            case 3:
                benchmarkKernel(repeatTimes, gelu_forward_kernel3, M * N / blockSize, blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
                break;
            case 4:
                benchmarkKernel(repeatTimes, gelu_forward_kernel4,
                                ceilDiv(M * N, (blockSize * Package128<float>::size)), blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, M, N);
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