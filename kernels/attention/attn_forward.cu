#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <cub/cub.cuh>

#include "common.h"

/* Attention Forward Implementation

Usage: ./attn_forward <kernel> [blockSize]
e.g. ./attn_forward 1

attn_forward_cpu(): CPU implementation
attn_forward_kernel1(): Basic CUDA implementation

Q, K, V are the query, key and value matrices
out is the output matrix
B is batch size, H is number of heads, T is sequence length, D is head dimension
*/

void attn_forward_cpu(float* Q, float* K, float* V, float* out, int B, int H,
                      int T, int D) {
  // Shape: Q,K,V: [B,H,T,D], out: [B,H,T,D]
  float* tmp = new float[T * T];  // Store Q*K^T

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      // Compute attention scores: Q * K^T / sqrt(D)
      float scale = 1.0f / sqrtf(D);
      for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
          float sum = 0.0f;
          for (int d = 0; d < D; d++) {
            sum += Q[b * H * T * D + h * T * D + i * D + d] *
                   K[b * H * T * D + h * T * D + j * D + d];
          }
          tmp[i * T + j] = sum * scale;
        }
      }

      // Softmax
      for (int i = 0; i < T; i++) {
        float max_val = tmp[i * T];
        for (int j = 1; j < T; j++) {
          max_val = max(max_val, tmp[i * T + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < T; j++) {
          tmp[i * T + j] = exp(tmp[i * T + j] - max_val);
          sum += tmp[i * T + j];
        }

        for (int j = 0; j < T; j++) {
          tmp[i * T + j] /= sum;
        }
      }

      // Multiply with V
      for (int i = 0; i < T; i++) {
        for (int d = 0; d < D; d++) {
          float sum = 0.0f;
          for (int j = 0; j < T; j++) {
            sum += tmp[i * T + j] * V[b * H * T * D + h * T * D + j * D + d];
          }
          out[b * H * T * D + h * T * D + i * D + d] = sum;
        }
      }
    }
  }
  delete[] tmp;
}

__global__ void attn_forward_kernel1(float* Q, float* K, float* V, float* out,
                                     int B, int H, int T, int D) {
  // Basic implementation - one thread handles one output element
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = B * H * T * D;

  if (idx < total) {
    int d = idx % D;
    int t = (idx / D) % T;
    int h = (idx / (T * D)) % H;
    int b = idx / (H * T * D);

    int offset = b * H * T * D + h * T * D;
    float* q = Q + offset;
    float* k = K + offset;
    float* v = V + offset;
    float sum = 0.0f;
    float scale = 1.0f / sqrtf(D);

    // Compute attention scores for this position
    float scores[1024];  // Assuming max T=1024 for simplicity
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Calculate scores and find max
    for (int i = 0; i < T; i++) {
      float score = 0.0f;
      for (int j = 0; j < D; j++) {
        score += q[t * D + j] * k[i * D + j];
      }
      score *= scale;
      scores[i] = score;
      max_score = max(max_score, score);
    }

    // Compute softmax
    for (int i = 0; i < T; i++) {
      scores[i] = exp(scores[i] - max_score);
      sum_exp += scores[i];
    }

    // Normalize and multiply with V
    for (int i = 0; i < T; i++) {
      float attn_prob = scores[i] / sum_exp;
      sum += attn_prob * v[i * D + d];
    }

    out[idx] = sum;
  }
}

#define B 32
#define H 8
#define T 32
#define D 32
#define BLOCK_SIZE 256
#define REPEAT_TIMES 100

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(
        stderr,
        "Usage: attn_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
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
  float* Q = (float*)malloc(B * H * T * D * sizeof(float));
  float* K = (float*)malloc(B * H * T * D * sizeof(float));
  float* V = (float*)malloc(B * H * T * D * sizeof(float));
  float* out = (float*)malloc(B * H * T * D * sizeof(float));
  float* resFromGPU = (float*)malloc(B * H * T * D * sizeof(float));

  // Initialize arrays
  initArrFloat(Q, B * H * T * D);
  initArrFloat(K, B * H * T * D);
  initArrFloat(V, B * H * T * D);
  zeros(out, B * H * T * D);

  // Allocate device memory
  float *Q_gpu, *K_gpu, *V_gpu, *out_gpu;
  cudaErrorCheck(cudaMalloc(&Q_gpu, B * H * T * D * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&K_gpu, B * H * T * D * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&V_gpu, B * H * T * D * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&out_gpu, B * H * T * D * sizeof(float)));

  // Copy data to device
  cudaErrorCheck(cudaMemcpy(Q_gpu, Q, B * H * T * D * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(K_gpu, K, B * H * T * D * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(V_gpu, V, B * H * T * D * sizeof(float),
                            cudaMemcpyHostToDevice));

  float elapsedTime = 0.0f;

  // Compute reference result on CPU
  attn_forward_cpu(Q, K, V, out, B, H, T, D);

  // Run selected kernel
  switch (kernel) {
    case 1: {
      int gridSize = (B * H * T * D + blockSize - 1) / blockSize;
      attn_forward_kernel1<<<gridSize, blockSize>>>(Q_gpu, K_gpu, V_gpu,
                                                    out_gpu, B, H, T, D);
      break;
    }
    default:
      printf("Error: Invalid kernel type: %d\n", kernel);
      return EXIT_FAILURE;
  }

  // Copy result back to host
  cudaErrorCheck(cudaMemcpy(resFromGPU, out_gpu, B * H * T * D * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cudaErrorCheck(cudaDeviceSynchronize());

  // Verify results and benchmark
  if (checkResults(out, resFromGPU, B * H * T * D)) {
    switch (kernel) {
      case 1: {
        int gridSize = (B * H * T * D + blockSize - 1) / blockSize;
        benchmarkKernel(repeatTimes, attn_forward_kernel1, gridSize, blockSize,
                        0, 0, &elapsedTime, Q_gpu, K_gpu, V_gpu, out_gpu, B, H,
                        T, D);
        break;
      }
    }
    printf(
        "attn_forward kernel: %d | B=%d H=%d T=%d D=%d | Time: %f ms | "
        "blockSize: %d\n",
        kernel, B, H, T, D, elapsedTime, blockSize);
  }

  // Cleanup
  free(Q);
  free(K);
  free(V);
  free(out);
  free(resFromGPU);
  cudaFree(Q_gpu);
  cudaFree(K_gpu);
  cudaFree(V_gpu);
  cudaFree(out_gpu);

  return EXIT_SUCCESS;
}