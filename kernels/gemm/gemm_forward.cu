#include "common.h"

/* GEMM (General Matrix Multiplication) forward implementation

Formula: D = A * B + C, where A (M x K), B (K x N), C (M x N), D (M x N)

Usage: ./gemm_forward <kernel>
e.g. ./gemm_forward 1

gemm_forward_cpu(): CPU implementation

gemm_forward_kernel1(): Naive implementation on CUDA. Each thread handles
one row of the input.

gemm_forward_kernel2(): Used shared memory and matrix tiling.

gemm_forward_kernel3(): On the base of kernel2, further let each thread handles
computation of (stride * stride) elements of D. However, this kernel fucked up
when M and N both larger than 512 (still don't know why).

*/

void gemm_cpu(const float *A, const float *B, const float *C, float *const D, const int M, const int N,
              const int K) {
    // D = A * B + C
    // A: M x K
    // B: K x N
    // C: M x N
    // D: M x N
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float val = 0.f;
            for (int k = 0; k < K; ++k) {
                val += A[m * K + k] * B[k * N + n];
            }
            D[m * N + n] = val + C[m * N + n];
        }
    }
}

__global__ void gemm_kernel1(const float *A, const float *B, const float *C, float *const D, const int M,
                             const int N, const int K) {
    // naive implementation
    // each thread calculates one row of D (M rows in total, one row has N
    // elements)
    const int m = blockDim.x * blockIdx.x + threadIdx.x;
    for (int n = 0; n < N; ++n) {
        float val = 0.f;
        for (int k = 0; k < K; ++k) {
            val += A[m * K + k] * B[k * N + n];
        }
        D[m * N + n] = val + C[m * N + n];
    }
}
template <int blockSize>
__global__ void gemm_kernel2(const float *A, const float *B, const float *C, float *const D, const int M,
                             const int N, const int K) {
    __shared__ float sharedA[blockSize][blockSize];
    __shared__ float sharedB[blockSize][blockSize];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0;
    if (row < M && col < N) {
        val = C[row * N + col];
    }

    for (int k = 0; k < K; k += blockSize) {
        sharedA[threadIdx.y][threadIdx.x] =
            (k + threadIdx.x) < K && row < M ? A[row * K + k + threadIdx.x] : 0.0f;

        sharedB[threadIdx.y][threadIdx.x] =
            (k + threadIdx.y) < K && col < N ? B[(k + threadIdx.y) * N + col] : 0.0f;

        __syncthreads();

        if (row < M && col < N) {
            for (int i = 0; i < blockSize; ++i) {
                val += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        D[row * N + col] = val;
    }
}

template <int blockSize, int stride, int step = blockSize * stride>
__global__ void gemm_kernel3(const float *__restrict__ const A, const float *__restrict__ const B,
                             const float *__restrict__ const C, float *__restrict__ const D, const int M,
                             const int N, const int K) {
    // add __restrict__ for guiding further compile optimization
    // compares to kernel2, kernel3 let each thread handles (stride * stride)
    // elements of D and results in less thread blocks
    // this version should perform better than version 2 with large matrices
    constexpr int padding = 0; 
    __shared__ float sharedA[step][step + padding];
    __shared__ float sharedB[step][step + padding];
    float vals[stride][stride];

    // do the addition first
    for (int r = 0; r < stride; ++r) {
        const int row = blockIdx.y * step + threadIdx.y * stride + r;
        for (int c = 0; c < stride; ++c) {
            const int col = blockIdx.x * step + threadIdx.x * stride + c;
            if (row < M && col < N) {
                D[row * N + col] = C[row * N + col];
            }
        }
    }

    for (int r = 0; r < stride; ++r) {
        for (int c = 0; c < stride; ++c) {
            vals[r][c] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += step) {
        // load (step, step) size of data into smem
        for (int r = 0; r < stride; ++r) {
            const int row = blockIdx.y * step + threadIdx.y * stride + r;
            for (int c = 0; c < stride; ++c) {
                const int col = blockIdx.x * step + threadIdx.x * stride + c;
                const int smemRow = threadIdx.y * stride + r, smemCol = threadIdx.x * stride + c;
                sharedA[smemRow][smemCol] = (k + threadIdx.x * stride + c) < K && row < M
                                                ? A[row * K + (k + threadIdx.x * stride + c)]
                                                : 0.0f;
                sharedB[smemRow][smemCol] = (k + threadIdx.y * stride + r) < K && col < N
                                                ? B[(k + threadIdx.y * stride + r) * N + col]
                                                : 0.0f;
            }
        }
        __syncthreads();

        // calculate the chunk matmul within smem
        for (int r = 0; r < stride; ++r) {
            int row = threadIdx.y * stride + r;
            for (int c = 0; c < stride; ++c) {
                int col = threadIdx.x * stride + c;
                float val = 0.0f;
                for (int i = 0; i < step; ++i) {
                    val += sharedA[row][i] * sharedB[i][col];
                }
                vals[r][c] += val;
            }
        }
        __syncthreads();
    }

    // udpate final vals to the output matrix
    for (int r = 0; r < stride; ++r) {
        const int row = blockIdx.y * step + threadIdx.y * stride + r;
        for (int c = 0; c < stride; ++c) {
            const int col = blockIdx.x * step + threadIdx.x * stride + c;
            if (row < M && col < N) {
                D[row * N + col] += vals[r][c];
            }
        }
    }
}

#define M 1024
#define K 1024
#define N 1024
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D 16
#define STRIDE_KERNEL3 2
#define REPEAT_TIMES 100

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: gemm_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
        return EXIT_FAILURE;
    }
    int kernel = atoi(argv[1]);

    int blockSize = BLOCK_SIZE_1D;
    if (argc > 2) {
        blockSize = atoi(argv[2]);
    }
    int repeatTimes = REPEAT_TIMES;
    if (argc > 3) {
        repeatTimes = atoi(argv[3]);
    }

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));
    float *resFromGPU = (float *)malloc(M * N * sizeof(float));
    initArrFloat(A, M * K);
    initArrFloat(B, K * N);
    initArrFloat(C, M * N);

    float *AGPU, *BGPU, *CGPU, *DGPU;

    cudaErrorCheck(cudaMalloc(&AGPU, M * K * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(AGPU, A, M * K * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&BGPU, K * N * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(BGPU, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&CGPU, M * N * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(CGPU, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&DGPU, M * N * sizeof(float)));

    float elapsedTime;
    gemm_cpu(A, B, C, D, M, N, K);

    switch (kernel) {
        case 1:
            gemm_kernel1<<<ceilDiv(M, blockSize), blockSize>>>(AGPU, BGPU, CGPU, DGPU, M, N, K);
            break;
        case 2: {
            blockSize = BLOCK_SIZE_2D * BLOCK_SIZE_2D;
            dim3 blockDim(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            dim3 gridDim(ceilDiv(N, BLOCK_SIZE_2D), ceilDiv(M, BLOCK_SIZE_2D));
            gemm_kernel2<BLOCK_SIZE_2D><<<gridDim, blockDim>>>(AGPU, BGPU, CGPU, DGPU, M, N, K);
            break;
        }
        case 3: {
            blockSize = BLOCK_SIZE_2D * BLOCK_SIZE_2D;
            dim3 blockDim(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            dim3 gridDim(ceilDiv(N, BLOCK_SIZE_2D * STRIDE_KERNEL3),
                         ceilDiv(M, BLOCK_SIZE_2D * STRIDE_KERNEL3));
            gemm_kernel3<BLOCK_SIZE_2D, STRIDE_KERNEL3>
                <<<gridDim, blockDim>>>(AGPU, BGPU, CGPU, DGPU, M, N, K);
            break;
        }
        default:
            printf("Error: Invalid kernel type: %i\n", kernel);
            return EXIT_FAILURE;
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(resFromGPU, DGPU, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (checkResults(D, resFromGPU, M * N)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(repeatTimes, gemm_kernel1, ceilDiv(M, blockSize), blockSize, 0, 0,
                                &elapsedTime, AGPU, BGPU, CGPU, DGPU, M, N, K);
                break;
            case 2:
                benchmarkKernel(repeatTimes, gemm_kernel2<BLOCK_SIZE_2D>,
                                dim3(ceilDiv(N, BLOCK_SIZE_2D), ceilDiv(M, BLOCK_SIZE_2D)),
                                dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D), 0, 0, &elapsedTime, AGPU, BGPU, CGPU,
                                DGPU, M, N, K);
                break;
            case 3:
                benchmarkKernel(repeatTimes, gemm_kernel3<BLOCK_SIZE_2D, STRIDE_KERNEL3>,
                                dim3(ceilDiv(N, BLOCK_SIZE_2D * STRIDE_KERNEL3),
                                     ceilDiv(M, BLOCK_SIZE_2D * STRIDE_KERNEL3)),
                                dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D), 0, 0, &elapsedTime, AGPU, BGPU, CGPU,
                                DGPU, M, N, K);
                break;
                break;
            default:
                printf("Error: Invalid kernel type: %i\n", kernel);
                return EXIT_FAILURE;
        }
        printf(
            "gemm_forward kernel: %i | A: (%i, %i), B: (%i, %i), C: (%i, "
            "%i) | "
            "Times: %f ms | "
            "blockSize: %i\n",
            kernel, M, K, K, N, M, N, elapsedTime, blockSize);
    }

    free(A);
    free(B);
    free(C);
    free(D);
    free(resFromGPU);
    cudaErrorCheck(cudaFree(AGPU));
    cudaErrorCheck(cudaFree(BGPU));
    cudaErrorCheck(cudaFree(CGPU));
    cudaErrorCheck(cudaFree(DGPU));
    return EXIT_SUCCESS;
}