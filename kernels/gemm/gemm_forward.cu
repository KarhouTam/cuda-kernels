#include "common.h"

void gemm_cpu(const float *A, const float *B, const float *C, float *const D,
              const int M, const int N, const int K) {
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

__global__ void gemm_kernel1(const float *A, const float *B, const float *C,
                             float *const D, const int M, const int N,
                             const int K) {
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
__global__ void gemm_kernel2(const float *A, const float *B, const float *C,
                             float *const D, const int M, const int N,
                             const int K) {
    __shared__ float sharedA[blockSize][blockSize];
    __shared__ float sharedB[blockSize][blockSize];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0;
    if (row < M && col < N) {
        val = C[row * N + col];
    }

    for (int k = 0; k < K; k += blockSize) {
        if (row < M && (k + threadIdx.x) < K) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((k + threadIdx.y) < K && col < N) {
            sharedB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

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
__global__ void gemm_kernel3(const float *A, const float *B, const float *C,
                             float *const D, const int M, const int N,
                             const int K) {
    // compares to kernel2, kernel3 let each thread handles (stride * stride)
    // elements of D and results in less thread blocks
    // but in fact kernel2 and kernel3's performance are closed but kernel3 meets bug when M * N > 512 * 512
    // currently don't know why...
    static_assert(stride > 0);
    __shared__ float sharedA[step][step];
    __shared__ float sharedB[step][step];
    float vals[stride][stride];
    for (int k = 0; k < K; k += step) {
        for (int r = 0; r < stride; ++r) {
            for (int c = 0; c < stride; ++c) {
                int colOffset = blockSize * c + threadIdx.x;
                int rowOffset = blockSize * r + threadIdx.y;
                int row = r * gridDim.y * blockSize + blockIdx.y * blockSize +
                          threadIdx.y;
                int col = c * gridDim.x * blockSize + blockIdx.x * blockSize +
                          threadIdx.x;
                if (row < M && (k + colOffset) < K) {
                    sharedA[rowOffset][colOffset] = A[row * K + k + colOffset];
                } else {
                    sharedA[rowOffset][colOffset] = 0.0f;
                }

                if ((k + rowOffset) < K && col < N) {
                    sharedB[rowOffset][colOffset] =
                        B[(k + rowOffset) * N + col];
                } else {
                    sharedB[rowOffset][colOffset] = 0.0f;
                }
            }
        }
        __syncthreads();
        for (int r = 0; r < stride; ++r) {
            for (int c = 0; c < stride; ++c) {
                int colOffset = blockSize * c + threadIdx.x;
                int rowOffset = blockSize * r + threadIdx.y;
                int row = blockIdx.y * blockSize + r * gridDim.y * blockSize +
                          threadIdx.y;
                int col = blockIdx.x * blockSize + c * gridDim.x * blockSize +
                          threadIdx.x;

                if (row < M && col < N) {
                    for (int i = 0; i < step; ++i) {
                        vals[r][c] +=
                            sharedA[rowOffset][i] * sharedB[i][colOffset];
                    }
                }
                __syncthreads();
            }
        }
    }
    __syncthreads();
    for (int r = 0; r < stride; ++r) {
        for (int c = 0; c < stride; ++c) {
            int row = r * gridDim.y * blockSize + blockIdx.y * blockSize +
                      threadIdx.y;
            int col = c * gridDim.x * blockSize + blockIdx.x * blockSize +
                      threadIdx.x;
            if (row < M && col < N)
                D[row * N + col] = vals[r][c] + C[row * N + col];
        }
    }
}

constexpr unsigned int M = 512;
constexpr unsigned int K = 512;
constexpr unsigned int N = 256;
constexpr unsigned int BLOCK_SIZE_1D = 256;
constexpr unsigned int BLOCK_SIZE_2D = 16;
constexpr unsigned int STRIDE_KERNEL3 = 2;

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: gemm_forward <kernel> [blockSize]\n");
        return EXIT_FAILURE;
    }
    int kernel = atoi(argv[1]);

    unsigned int blockSize = BLOCK_SIZE_1D;
    if (argc > 2) {
        blockSize = atoi(argv[2]);
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
    cudaErrorCheck(
        cudaMemcpy(AGPU, A, M * K * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&BGPU, K * N * sizeof(float)));
    cudaErrorCheck(
        cudaMemcpy(BGPU, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&CGPU, M * N * sizeof(float)));
    cudaErrorCheck(
        cudaMemcpy(CGPU, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc(&DGPU, M * N * sizeof(float)));

    float elapsedTime;
    gemm_cpu(A, B, C, D, M, N, K);

    switch (kernel) {
        case 1:
            gemm_kernel1<<<ceilDiv(M, blockSize), blockSize>>>(AGPU, BGPU, CGPU,
                                                               DGPU, M, N, K);
            break;
        case 2: {
            blockSize = BLOCK_SIZE_2D * BLOCK_SIZE_2D;
            dim3 blockDim(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            dim3 gridDim(ceilDiv(N, BLOCK_SIZE_2D), ceilDiv(M, BLOCK_SIZE_2D));
            gemm_kernel2<BLOCK_SIZE_2D>
                <<<gridDim, blockDim>>>(AGPU, BGPU, CGPU, DGPU, M, N, K);
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
    cudaErrorCheck(cudaMemcpy(resFromGPU, DGPU, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    if (checkResults(D, resFromGPU, M * N)) {
        switch (kernel) {
            case 1:
                benchmarkKernel(gemm_kernel1, ceilDiv(M, blockSize), blockSize,
                                &elapsedTime, AGPU, BGPU, CGPU, DGPU, M, N, K);
                break;
            case 2:
                benchmarkKernel(
                    gemm_kernel2<BLOCK_SIZE_2D>,
                    dim3(ceilDiv(N, BLOCK_SIZE_2D), ceilDiv(M, BLOCK_SIZE_2D)),
                    dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D), &elapsedTime, AGPU,
                    BGPU, CGPU, DGPU, M, N, K);
                break;
            case 3:
                benchmarkKernel(
                    gemm_kernel3<BLOCK_SIZE_2D, STRIDE_KERNEL3>,
                    dim3(ceilDiv(N, BLOCK_SIZE_2D * STRIDE_KERNEL3),
                         ceilDiv(M, BLOCK_SIZE_2D * STRIDE_KERNEL3)),
                    dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D), &elapsedTime, AGPU,
                    BGPU, CGPU, DGPU, M, N, K);
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