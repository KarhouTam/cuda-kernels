#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>

#include "common.h"

/* Layer Normalization forward implementation

Usage: ./layernorm_forward <kernel> [blockSize]
e.g. ./layernorm_forward 1

layernorm_forward_cpu(): CPU implementation

layernorm_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row of the input.

layernorm_forward_kernel2(): Optimized implementation on CUDA. Compares to
kernel1, each warp (32 threads) handles one row.

layernorm_forward_kernel3(): Similar to kernel2, each warp handles one row, but
uses CUDA's cooperative groups instead.

layernorm_forward_kernel4(): On the base of kernel2, plus using shared memory to
store the intermediate shift values (x - mean).

layernorm_forward_kernel5(): On the base of kernel2, using formula D(X) = E(X^2)
- E(X)^2 to reduce the number of loops.

*/

void layernorm_forward_cpu(float *input, float *out, float *weight, float *bias,
                           float eps, int B, int C, int K) {
    // In normal, the input data has shape [B, C, K], B is batch size, C is
    // number of channels, K is sequence length
    for (int row = 0; row < B * C; ++row) {
        float *const x = input + row * K;
        float *const y = out + row * K;
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

__global__ void layernorm_forward_kernel1(float *input, float *out,
                                          float *weight, float *bias, float eps,
                                          int B, int C, int K) {
    // naive implementation
    // each thread handle one row of input
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C) {
        float *const x = input + idx * K;
        float *const y = out + idx * K;
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

__global__ void layernorm_forward_kernel2(float *input, float *out,
                                          float *weight, float *bias, float eps,
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
            float *const x = input + row * K;
            float *const y = out + row * K;
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

__global__ void layernorm_forward_kernel3(float *input, float *out,
                                          float *weight, float *bias, float eps,
                                          int B, int C, int K) {
    // compares to kernel2, use cooperative groups (just for practice)
    // performance is very close to kernel 2
    namespace cg = cooperative_groups;
    cg::thread_block thisBlock = cg::this_thread_block();
    cg::thread_block_tile<32> thisWarp = cg::tiled_partition<32>(thisBlock);
    int warpId = thisWarp.meta_group_rank();
    int warpsPerBlock = thisWarp.meta_group_size();
    int laneId = thisWarp.thread_rank();
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * C;
         row += numWarps) {
        float *const x = input + row * K;
        float *const y = out + row * K;
        float sum = 0.0f;

        for (int i = laneId; i < K; i += thisWarp.num_threads()) {
            sum += x[i];
        }

        sum = cg::reduce(thisWarp, sum, plus<float>);

        float mean = sum / K;
        float var = 0.f;

        for (int i = laneId; i < K; i += thisWarp.num_threads()) {
            float xShift = x[i] - mean;
            var += xShift * xShift;
        }

        var = cg::reduce(thisWarp, var, plus<float>);

        float inv_std = 1.0f / sqrt(var / K + eps);

        for (int i = laneId; i < K; i += thisWarp.num_threads()) {
            y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel4(float *input, float *out,
                                          float *weight, float *bias, float eps,
                                          int B, int C, int K) {
    // one warp one row, plus using smem to store the shift (x - mean) values
    // when K < 2048, the performance is better than kernel 2 and 3
    assert((K % warpSize) == 0);
    extern __shared__ float xShifts[];
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    float *const xShiftsThisWarp = xShifts + warpId * K;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * C;
         row += numWarps)
        if (row < B * C) {
            float *const x = input + row * K;
            float *const y = out + row * K;
            float partialSum = 0.0f;
            for (int i = laneId; i < K; i += warpSize) {
                xShiftsThisWarp[i] = x[i];
                partialSum += x[i];
            }

            float sum = warpReduceSum(partialSum);
            float mean = sum / K;

            float var = 0.f;
            for (int i = laneId; i < K; i += warpSize) {
                xShiftsThisWarp[i] -= mean;
                var += xShiftsThisWarp[i] * xShiftsThisWarp[i];
            }

            var = warpReduceSum(var);
            float inv_std = 1.0f / sqrt(var / K + eps);

            for (int i = laneId; i < K; i += warpSize) {
                y[i] = weight[i] * xShiftsThisWarp[i] * inv_std + bias[i];
            }
        }
}

__global__ void layernorm_forward_kernel5(float *input, float *out,
                                          float *weight, float *bias, float eps,
                                          int B, int C, int K) {
    // using formula D(X) = E(X^2) - E(X)^2 to reduce the number of loops
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * C;
         row += numWarps)
        if (row < B * C) {
            float *const x = input + row * K;
            float *const y = out + row * K;
            float partialSum = 0.0f;
            float partialSum2 = 0.0f;
            for (int i = laneId; i < K; i += warpSize) {
                float xi = x[i];
                partialSum += xi;
                partialSum2 += xi * xi;
            }

            float mean = warpReduceSum(partialSum) / K;
            float mean2 = warpReduceSum(partialSum2) / K;

            float var = (mean2 - mean * mean);
            float inv_std = rsqrtf(var + eps);

            for (int i = laneId; i < K; i += warpSize) {
                y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
            }
        }
}

__global__ void layernorm_forward_kernel6(float *__restrict__ input,
                                          float *__restrict__ out,
                                          float *__restrict__ weight,
                                          float *__restrict__ bias, float eps,
                                          int B, int C, int K) {
    // using formula D(X) = E(X^2) - E(X)^2 to reduce the number of loops
    // while using smem to store input data
    // and use float4 to acclerate memory access
    using f128 = Package128<float>;
    extern __shared__ f128 xShared[];
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    f128 *xSharedWarp = xShared + warpId * (K / f128::size);
    int row = blockIdx.x * warpsPerBlock + warpId;
    if (row < B * C) {
        float *x = input + row * K;
        float *y = out + row * K;
        float warpSum = 0.0f;
        float warpSum2 = 0.0f;
        for (int i = laneId * f128::size; i < K; i += warpSize * f128::size) {
            f128 xi = load128(x + i);
            xSharedWarp[i / f128::size] = xi;
            for (int k = 0; k < f128::size; ++k) {
                warpSum += xi[k];
                warpSum2 += xi[k] * xi[k];
            }
        }

        float mean = warpReduceSum(warpSum) / K;
        float mean2 = warpReduceSum(warpSum2) / K;

        float var = (mean2 - mean * mean);
        float inv_std = rsqrtf(var + eps);

        for (int i = laneId * f128::size; i < K; i += warpSize * f128::size) {
            f128 packedW = load128(weight + i);
            f128 packedB = load128(bias + i);
            f128 packedX = xSharedWarp[i / f128::size];
            f128 out;
#pragma unroll
            for (int k = 0; k < f128::size; ++k) {
                out[k] = packedW[k] * inv_std * ((float)packedX[k] - mean) +
                         packedB[k];
            }
            store128(y + i, out);
        }
    }
}

#define B 4096
#define C 3
#define K 1024
#define EPS 1e-5
#define BLOCK_SIZE 128

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: relu_forward <kernel> [blockSize]\n");
        return EXIT_FAILURE;
    }
    int kernel = atoi(argv[1]);

    int blockSize = BLOCK_SIZE;
    if (argc > 2) {
        blockSize = atoi(argv[2]);
    }

    float *input = (float *)malloc(B * C * K * sizeof(float));
    float *output = (float *)malloc(B * C * K * sizeof(float));
    float *weight = (float *)malloc(K * sizeof(float));
    float *bias = (float *)malloc(K * sizeof(float));
    float *resFromGPU = (float *)malloc(B * C * K * sizeof(float));
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
            layernorm_forward_kernel1<<<B * C / blockSize, blockSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        case 2:
            layernorm_forward_kernel2<<<B * C * 32 / blockSize, blockSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        case 3:
            layernorm_forward_kernel3<<<B * C * 32 / blockSize, blockSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        case 4: {
            int smemSize = K * sizeof(float) * (blockSize / 32);
            layernorm_forward_kernel4<<<B * C * 32 / blockSize, blockSize,
                                        smemSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        }
        case 5:
            layernorm_forward_kernel5<<<B * C * 32 / blockSize, blockSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        case 6: {
            const int smemSize = blockSize / 32 * K * sizeof(float);
            layernorm_forward_kernel6<<<B * C * 32 / blockSize, blockSize,
                                        smemSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, C, K);
            break;
        }
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
                                blockSize, 0, 0, &elapsedTime, inputGPU,
                                outputGPU, weightGPU, biasGPU, EPS, B, C, K);
                break;
            case 2:
                benchmarkKernel(layernorm_forward_kernel2,
                                ceilDiv(B * C * 32, blockSize), blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, weightGPU,
                                biasGPU, EPS, B, C, K);
                break;
            case 3:
                benchmarkKernel(layernorm_forward_kernel3,
                                ceilDiv(B * C * 32, blockSize), blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, weightGPU,
                                biasGPU, EPS, B, C, K);
                break;
            case 4:
                benchmarkKernel(layernorm_forward_kernel4,
                                ceilDiv(B * C * 32, blockSize), blockSize,
                                K * sizeof(float) * (blockSize / 32), 0,
                                &elapsedTime, inputGPU, outputGPU, weightGPU,
                                biasGPU, EPS, B, C, K);
                break;
            case 5:
                benchmarkKernel(layernorm_forward_kernel5,
                                ceilDiv(B * C * 32, blockSize), blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, weightGPU,
                                biasGPU, EPS, B, C, K);
                break;
            case 6:
                benchmarkKernel(layernorm_forward_kernel6,
                                ceilDiv(B * C * 32, blockSize), blockSize,
                                blockSize / 32 * K * sizeof(float), 0,
                                &elapsedTime, inputGPU, outputGPU, weightGPU,
                                biasGPU, EPS, B, C, K);
                break;
        }
        printf(
            "layer_norm_forward kernel: %i | matrixSize: %i x %i x %i | "
            "Times: "
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