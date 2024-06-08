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
  // tiling + shared memory
  __shared__ float sharedA[blockSize][blockSize];
  __shared__ float sharedB[blockSize][blockSize];

  for (int row = blockIdx.y * blockDim.y + threadIdx.y;
       row < M + blockDim.y - 1; row += blockDim.y * gridDim.y) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < N + blockDim.x - 1; col += blockDim.x * gridDim.x) {
      float val = 0.f;
      for (int k = 0; k < ceilDiv(K, blockSize); ++k) {
        sharedA[threadIdx.y][threadIdx.x] =
            A[row * K + k * blockSize + threadIdx.x];
        sharedB[threadIdx.y][threadIdx.x] =
            B[(k * blockSize + threadIdx.y) * N + col];
        __syncthreads();
        for (int i = 0; i < blockSize; ++i) {
          val += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        __syncthreads();
      }
      if (row < M && col < N) D[row * N + col] = val + C[row * N + col];
    }
  }
}

#define M 512
#define K 256
#define N 512
#define BLOCK_SIZE 256

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: gemm_forward <kernel> [blockSize]\n");
    return EXIT_FAILURE;
  }
  int kernel = atoi(argv[1]);

  int blockSize = BLOCK_SIZE;
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
      gemm_kernel1<<<ceilDiv(M, blockSize), blockSize>>>(AGPU, BGPU, CGPU, DGPU,
                                                         M, N, K);
      break;
    case 2: {
      blockSize = 16 * 16;
      dim3 blockDim(16, 16);
      dim3 gridDim(ceilDiv(N, 16), ceilDiv(M, 16));
      gemm_kernel2<16><<<gridDim, blockDim>>>(AGPU, BGPU, CGPU, DGPU, M, N, K);
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
        benchmarkKernel(gemm_kernel2<16>, dim3(M / 16, N / 16), dim3(16, 16),
                        &elapsedTime, AGPU, BGPU, CGPU, DGPU, M, N, K);
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