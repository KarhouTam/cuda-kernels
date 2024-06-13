#include "common.h"

void layernorm_forward_cpu(float* input, float* out, float* weight, float* bias,
                           float eps, int B, int C, int K) {
    // In normal, the input data has shape [B, C, K], B is batch size, C is
    // number of channels, K is sequence length
    for (int row = 0; row < B * C; ++row) {
        float* const x = input + row * K;
        float* const y = out + row * K;
        // mean
        float mean = 0.0f;
        for (int i = 0; i < K; ++i) {
            mean += x[i];
        }
        mean /= K;

        float var = 0.f;
        for (int i = 0; i < K; ++i) {
            float xShift = x[i] - mean;
            var += xShift * xShift;
        }
        float inv_std = 1.0f / sqrt(var / K + eps);

        for (int i = 0; i < K; ++i) {
            y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel1(float* input, float* out,
                                          float* weight, float* bias, float eps,
                                          int B, int C, int K) {
    // naive implementation
    // each thread handle one row of input
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C) {
        float* const x = input + idx * K;
        float* const y = out + idx * K;
        float mean = 0.0f;
        for (int i = 0; i < K; ++i) {
            mean += x[i];
        }

        mean /= K;

        float var = 0.f;
        for (int i = 0; i < K; ++i) {
            float xShift = x[i] - mean;
            var += xShift * xShift;
        }
        float inv_std = 1.0f / sqrt(var / K + eps);

        for (int i = 0; i < K; ++i) {
            y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel2(float* input, float* out,
                                          float* weight, float* bias, float eps,
                                          int B, int C, int K) {
    // one warp one row
    // each thread handle one row of input
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * C;
         row += numWarps)
        if (row < B * C) {
            float* const x = input + row * K;
            float* const y = out + row * K;
            float sum = 0.0f;
            for (int i = laneId; i < K; i += warpSize) {
                sum += x[i];
            }

            sum = warpReduceSum(sum);
            float mean = sum / K;

            float var = 0.f;
            for (int i = laneId; i < K; i += warpSize) {
                float xShift = x[i] - mean;
                var += xShift * xShift;
            }

            var = warpReduceSum(var);
            float inv_std = 1.0f / sqrt(var / K + eps);

            for (int i = laneId; i < K; i += warpSize) {
                y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
            }
        }
}

#define B 1024
#define C 3
#define K 2048
#define EPS 1e-5
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

    float* input = (float*)malloc(B * C * K * sizeof(float));
    float* output = (float*)malloc(B * C * K * sizeof(float));
    float* weight = (float*)malloc(K * sizeof(float));
    float* bias = (float*)malloc(K * sizeof(float));
    float* resFromGPU = (float*)malloc(B * C * K * sizeof(float));
    initArrFloat(input, B * C * K);
    initArrFloat(weight, K);
    initArrFloat(bias, K);

    float *inputGPU, *outputGPU, *weightGPU, *biasGPU;

    cudaErrorCheck(cudaMalloc(&inputGPU, B * C * K * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(inputGPU, input, B * C * K * sizeof(float),
                              cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&weightGPU, K * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(weightGPU, weight, K * sizeof(float),
                              cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&biasGPU, K * sizeof(float)));
    cudaErrorCheck(
        cudaMemcpy(biasGPU, bias, K * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&outputGPU, B * C * K * sizeof(float)));

    float elapsedTime;

    layernorm_forward_cpu(input, output, weight, bias, EPS, B, C, K);

    switch (kernel) {
        case 1:
            layernorm_forward_kernel1<<<B * C * K / blockSize, blockSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        case 2:
            layernorm_forward_kernel2<<<B * C * K / blockSize, blockSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }
    cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, B * C * K * sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDeviceSynchronize());

    if (checkResults(output, resFromGPU, B * C * K)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(layernorm_forward_kernel1, B * C / blockSize,
                                blockSize, &elapsedTime, inputGPU, outputGPU,
                                weightGPU, biasGPU, EPS, B, C, K);
                break;
            case 2:
                benchmarkKernel(layernorm_forward_kernel2,
                                ceilDiv(B * C * 32, blockSize), blockSize,
                                &elapsedTime, inputGPU, outputGPU, weightGPU,
                                biasGPU, EPS, B, C, K);
                break;
        }
        printf(
            "layer_norm_forward kernel: %i | matrixSize: %i x %i x %i | Times: "
            "%f ms | "
            "blockSize: %i\n",
            kernel, B, C, K, elapsedTime, blockSize);
    }
    free(input);
    free(weight);
    free(bias);
    free(output);
    free(resFromGPU);
    cudaErrorCheck(cudaFree(inputGPU));
    cudaErrorCheck(cudaFree(weightGPU));
    cudaErrorCheck(cudaFree(biasGPU));
    cudaErrorCheck(cudaFree(outputGPU));
    return EXIT_SUCCESS;
}