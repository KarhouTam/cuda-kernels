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

layernorm_forward_kernel6(): use D(X) = E(X^2) and smem to acclerate memory access.

layernorm_forward_kernel7(): block reduce version.

layernorm_forward_kernel8(): on the base of kernel 6, plus using float4.

*/

void layernorm_forward_cpu(float *input, float *output, float *weight, float *bias, float eps, int B, int T,
                           int C) {
    // In normal, the input data has shape [B, T, C], B is batch size, T is sequence
    // length, C is token length
    for (int row = 0; row < B * T; ++row) {
        float *const x = input + row * C;
        float *const y = output + row * C;
        // mean
        float mean = 0.0f;
        for (int i = 0; i < C; ++i) {
            mean += x[i];
        }
        mean /= C;

        float var = 0.f;
        for (int i = 0; i < C; ++i) {
            float xShift = x[i] - mean;
            var += xShift * xShift;
        }
        float inv_std = 1.0f / sqrt(var / C + eps);

        for (int i = 0; i < C; ++i) {
            y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel1(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    // naive implementation
    // each thread handle one row of input
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * T) {
        float *const x = input + idx * C;
        float *const y = output + idx * C;
        float mean = 0.0f;
        for (int i = 0; i < C; ++i) {
            mean += x[i];
        }

        mean /= C;

        float var = 0.f;
        for (int i = 0; i < C; ++i) {
            float xShift = x[i] - mean;
            var += xShift * xShift;
        }
        float inv_std = 1.0f / sqrt(var / C + eps);

        for (int i = 0; i < C; ++i) {
            y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel2(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    // one warp one row
    // each thread handle one row of input
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T; row += numWarps)
        if (row < B * T) {
            float *const x = input + row * C;
            float *const y = output + row * C;
            float sum = 0.0f;
            for (int i = laneId; i < C; i += warpSize) {
                sum += x[i];
            }

            sum = warpReduceSum(sum);
            float mean = sum / C;

            float var = 0.f;
            for (int i = laneId; i < C; i += warpSize) {
                float xShift = x[i] - mean;
                var += xShift * xShift;
            }

            var = warpReduceSum(var);
            float inv_std = 1.0f / sqrt(var / C + eps);

            for (int i = laneId; i < C; i += warpSize) {
                y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
            }
        }
}

__global__ void layernorm_forward_kernel3(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    // compares to kernel2, use cooperative groups (just for practice)
    // performance is very close to kernel 2
    namespace cg = cooperative_groups;
    cg::thread_block thisBlock = cg::this_thread_block();
    cg::thread_block_tile<32> thisWarp = cg::tiled_partition<32>(thisBlock);
    int warpId = thisWarp.meta_group_rank();
    int warpsPerBlock = thisWarp.meta_group_size();
    int laneId = thisWarp.thread_rank();
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T; row += numWarps) {
        float *const x = input + row * C;
        float *const y = output + row * C;
        float sum = 0.0f;

        for (int i = laneId; i < C; i += thisWarp.num_threads()) {
            sum += x[i];
        }

        sum = cg::reduce(thisWarp, sum, plus<float>);

        float mean = sum / C;
        float var = 0.f;

        for (int i = laneId; i < C; i += thisWarp.num_threads()) {
            float xShift = x[i] - mean;
            var += xShift * xShift;
        }

        var = cg::reduce(thisWarp, var, plus<float>);

        float inv_std = 1.0f / sqrt(var / C + eps);

        for (int i = laneId; i < C; i += thisWarp.num_threads()) {
            y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel4(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    // one warp one row, plus using smem to store the shift (x - mean) values
    assert((C % warpSize) == 0);
    extern __shared__ float xShifts[];
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    float *const xShiftsWarp = xShifts + warpId * C;
    int row = blockIdx.x * warpsPerBlock + warpId;
    if (row < B * T) {
        float *const x = input + row * C;
        float *const y = output + row * C;
        float partialSum = 0.0f;
        for (int i = laneId; i < C; i += warpSize) {
            xShiftsWarp[i] = x[i];
            partialSum += x[i];
        }

        float sum = warpReduceSum(partialSum);
        float mean = sum / C;

        float var = 0.f;
        for (int i = laneId; i < C; i += warpSize) {
            xShiftsWarp[i] -= mean;
            var += xShiftsWarp[i] * xShiftsWarp[i];
        }

        var = warpReduceSum(var);
        float inv_std = 1.0f / sqrt(var / C + eps);

        for (int i = laneId; i < C; i += warpSize) {
            y[i] = weight[i] * xShiftsWarp[i] * inv_std + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel5(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    // using formula D(X) = E(X^2) - E(X)^2 to reduce the number of loops
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T; row += numWarps)
        if (row < B * T) {
            float *const x = input + row * C;
            float *const y = output + row * C;
            float partialSum = 0.0f;
            float partialSum2 = 0.0f;
            for (int i = laneId; i < C; i += warpSize) {
                float xi = x[i];
                partialSum += xi;
                partialSum2 += xi * xi;
            }

            float mean = warpReduceSum(partialSum) / C;
            float mean2 = warpReduceSum(partialSum2) / C;

            float var = (mean2 - mean * mean);
            float inv_std = rsqrtf(var + eps);

            for (int i = laneId; i < C; i += warpSize) {
                y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
            }
        }
}

__global__ void layernorm_forward_kernel6(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    // one warp one row
    // use smem to store the shift (x - mean) values
    // use D(X) = E(X^2) - E(X)^2
    assert((C % warpSize) == 0);
    extern __shared__ float sharedX[];
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    float *const xSharedWarp = sharedX + warpId * C;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < B * T; row += numWarps)
        if (row < B * T) {
            float *const x = input + row * C;
            float *const y = output + row * C;
            float partialSum = 0.0f, partialSum2 = 0.0f;
            for (int i = laneId; i < C; i += warpSize) {
                float xi = x[i];
                xSharedWarp[i] = xi;
                partialSum += xi;
                partialSum2 += xi * xi;
            }

            float mean = warpReduceSum(partialSum) / C;
            float mean2 = warpReduceSum(partialSum2) / C;
            float var = (mean2 - mean * mean);
            float inv_std = rsqrtf(var + eps);

            for (int i = laneId; i < C; i += warpSize) {
                y[i] = weight[i] * (xSharedWarp[i] - mean) * inv_std + bias[i];
            }
        }
}

__global__ void layernorm_forward_kernel7(float *input, float *output, float *weight, float *bias,
                                          float eps, int B, int T, int C) {
    extern __shared__ float shared[];
    if (blockIdx.x < B * T) {
        const int laneId = threadIdx.x % warpSize;
        const int warpId = threadIdx.x / warpSize;
        const int warpsPerBlock = ceilDiv(blockDim.x, warpSize);
        const int dataPerWarp = ceilDiv(C, warpsPerBlock);
        const int start = dataPerWarp * warpId;
        const int end = min((warpId + 1) * dataPerWarp, C);

        float *const smem1 = shared;
        float *const smem2 = shared + warpsPerBlock;

        const float *x = input + blockIdx.x * C;
        float *const y = output + blockIdx.x * C;

        float sumVal = 0.f, sum2Val = 0.f;
        for (int i = start + laneId; i < end; i += warpSize) {
            float xi = x[i];
            sumVal += xi;
            sum2Val += xi * xi;
        }

        sumVal = warpReduceSum(sumVal);
        sum2Val = warpReduceSum(sum2Val);
        if (laneId == 0) {
            smem1[warpId] = sumVal;
            smem2[warpId] = sum2Val;
        }
        __syncthreads();

        if (warpId == 0) {
            if (laneId < warpsPerBlock) {
                sumVal = smem1[laneId];
                sum2Val = smem2[laneId];
            } else {
                sumVal = 0.f;
                sum2Val = 0.f;
            }
            sumVal = warpReduceSum(sumVal);
            sum2Val = warpReduceSum(sum2Val);
            if (laneId == 0) {
                smem1[0] = sumVal / C;
                smem2[0] = sum2Val / C;
            }
        }
        __syncthreads();

        float mean1 = smem1[0];
        float mean2 = smem2[0];

        float var = (mean2 - mean1 * mean1);
        float invStd = rsqrtf(var + eps);

        for (int i = start + laneId; i < end; i += warpSize) {
            float xi = x[i];
            y[i] = weight[i] * (xi - mean1) * invStd + bias[i];
        }
    }
}

__global__ void layernorm_forward_kernel8(float *__restrict__ input, float *__restrict__ output,
                                          float *__restrict__ weight, float *__restrict__ bias, float eps,
                                          int B, int T, int C) {
    // using formula D(X) = E(X^2) - E(X)^2 to reduce the number of loops
    // while using smem to store input data
    // and use float4 to acclerate memory access
    using f128 = Package128<float>;
    extern __shared__ f128 xShared[];
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    f128 *xSharedWarp = xShared + warpId * (C / f128::size);
    int row = blockIdx.x * warpsPerBlock + warpId;
    if (row < B * T) {
        float *x = input + row * C;
        float *y = output + row * C;
        float warpSum = 0.0f;
        float warpSum2 = 0.0f;
        for (int i = laneId * f128::size; i < C; i += warpSize * f128::size) {
            f128 xi = load128(x + i);
            xSharedWarp[i / f128::size] = xi;
            for (int k = 0; k < f128::size; ++k) {
                warpSum += xi[k];
                warpSum2 += xi[k] * xi[k];
            }
        }

        float mean = warpReduceSum(warpSum) / C;
        float mean2 = warpReduceSum(warpSum2) / C;

        float var = (mean2 - mean * mean);
        float inv_std = rsqrtf(var + eps);

        for (int i = laneId * f128::size; i < C; i += warpSize * f128::size) {
            f128 packedW = load128(weight + i);
            f128 packedB = load128(bias + i);
            f128 packedX = xSharedWarp[i / f128::size];
            f128 out;
#pragma unroll
            for (int k = 0; k < f128::size; ++k) {
                out[k] = packedW[k] * inv_std * ((float)packedX[k] - mean) + packedB[k];
            }
            store128(y + i, out);
        }
    }
}

#define B 8
#define T 1024
#define C 768
#define EPS 1e-5
#define BLOCK_SIZE 128
#define REPEAT_TIMES 100

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: layernorm_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
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

    float *input = (float *)malloc(B * T * C * sizeof(float));
    float *output = (float *)malloc(B * T * C * sizeof(float));
    float *weight = (float *)malloc(C * sizeof(float));
    float *bias = (float *)malloc(C * sizeof(float));
    float *resFromGPU = (float *)malloc(B * T * C * sizeof(float));
    initArrFloat(input, B * T * C);
    initArrFloat(weight, C);
    initArrFloat(bias, C);
    zeros(output, B * T * C);

    float *inputGPU, *outputGPU, *weightGPU, *biasGPU;

    cudaErrorCheck(cudaMalloc(&inputGPU, B * T * C * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(inputGPU, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&weightGPU, C * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(weightGPU, weight, C * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&biasGPU, C * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(biasGPU, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&outputGPU, B * T * C * sizeof(float)));
    cudaErrorCheck(cudaMemset(outputGPU, 0, B * T * C * sizeof(float)));

    float elapsedTime = 0.0f;

    layernorm_forward_cpu(input, output, weight, bias, EPS, B, T, C);

    switch (kernel) {
        case 1:
            layernorm_forward_kernel1<<<B * T / blockSize, blockSize>>>(inputGPU, outputGPU, weightGPU,
                                                                        biasGPU, EPS, B, T, C);
            break;
        case 2:
            layernorm_forward_kernel2<<<B * T * 32 / blockSize, blockSize>>>(inputGPU, outputGPU, weightGPU,
                                                                             biasGPU, EPS, B, T, C);
            break;
        case 3:
            layernorm_forward_kernel3<<<B * T * 32 / blockSize, blockSize>>>(inputGPU, outputGPU, weightGPU,
                                                                             biasGPU, EPS, B, T, C);
            break;
        case 4: {
            int smemSize = C * sizeof(float) * (blockSize / 32);
            layernorm_forward_kernel4<<<B * T * 32 / blockSize, blockSize, smemSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
            break;
        }
        case 5:
            layernorm_forward_kernel5<<<B * T * 32 / blockSize, blockSize>>>(inputGPU, outputGPU, weightGPU,
                                                                             biasGPU, EPS, B, T, C);
            break;
        case 6: {
            int smemSize = C * sizeof(float) * (blockSize / 32);
            layernorm_forward_kernel6<<<B * T * 32 / blockSize, blockSize, smemSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
            break;
        }
        case 7: {
            layernorm_forward_kernel7<<<B * T, blockSize, ceilDiv(blockSize, 32) * 2 * sizeof(float)>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
            break;
        }
        case 8: {
            const int smemSize = blockSize / 32 * C * sizeof(float);
            layernorm_forward_kernel8<<<B * T * 32 / blockSize, blockSize, smemSize>>>(
                inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
            break;
        }
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }
    cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDeviceSynchronize());

    if (checkResults(output, resFromGPU, B * T * C)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel1, B * T / blockSize, blockSize, 0, 0,
                                &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU, EPS, B, T, C);
                break;
            case 2:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel2, ceilDiv(B * T * 32, blockSize),
                                blockSize, 0, 0, &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU, EPS,
                                B, T, C);
                break;
            case 3:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel3, ceilDiv(B * T * 32, blockSize),
                                blockSize, 0, 0, &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU, EPS,
                                B, T, C);
                break;
            case 4:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel4, ceilDiv(B * T * 32, blockSize),
                                blockSize, C * sizeof(float) * (blockSize / 32), 0, &elapsedTime, inputGPU,
                                outputGPU, weightGPU, biasGPU, EPS, B, T, C);
                break;
            case 5:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel5, ceilDiv(B * T * 32, blockSize),
                                blockSize, 0, 0, &elapsedTime, inputGPU, outputGPU, weightGPU, biasGPU, EPS,
                                B, T, C);
                break;
            case 6:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel6, ceilDiv(B * T * 32, blockSize),
                                blockSize, blockSize / 32 * C * sizeof(float), 0, &elapsedTime, inputGPU,
                                outputGPU, weightGPU, biasGPU, EPS, B, T, C);
            case 7:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel7, B * T, blockSize,
                                ceilDiv(blockSize, 32) * 2 * sizeof(float), 0, &elapsedTime, inputGPU,
                                outputGPU, weightGPU, biasGPU, EPS, B, T, C);
            case 8:
                benchmarkKernel(repeatTimes, layernorm_forward_kernel8, ceilDiv(B * T * 32, blockSize),
                                blockSize, blockSize / 32 * C * sizeof(float), 0, &elapsedTime, inputGPU,
                                outputGPU, weightGPU, biasGPU, EPS, B, T, C);
                break;
        }
        printf(
            "layer_norm_forward kernel: %i | matrixSize: %i x %i x %i | "
            "Times: "
            "%f ms | "
            "blockSize: %i\n",
            kernel, B, T, C, elapsedTime, blockSize);
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