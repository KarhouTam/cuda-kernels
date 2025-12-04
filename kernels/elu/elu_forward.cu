#include "common.h"

#define ALPHA 1.0f

__device__ __forceinline__ float elu(float x) {
  return x > 0 ? x : ALPHA * (expf(x) - 1.0f);
}

void elu_cpu(float* input, float* output, const int M, const int N) {
  for (int m = 0; m < M; m++) {
    float* in = input + m * N;
    float* out = output + m * N;
    for (int n = 0; n < N; n++) {
      out[n] = in[n] > 0 ? in[n] : ALPHA * (expf(in[n]) - 1.0f);
    }
  }
}

__global__ void elu_forward_kernel1(const float* input, float* output,
                                    const int M, const int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < M) {
    const float* const in = input + tid * N;
    float* const out = output + tid * N;
    for (int i = 0; i < N; i++) {
      out[i] = elu(in[i]);
    }
  }
}

#define M 8192
#define N 8192
#define BLOCK_SIZE 128
#define REPEAT_TIMES 100

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: elu_forward <kernel> [blockSize] [benchmarkRepeatTimes]\n");
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
  cudaErrorCheck(cudaMemcpy(inputGPU, input, M * N * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMalloc(&outputGPU, M * N * sizeof(float)));

  float elapsedTime = 0.0f;

  elu_cpu(input, output, M, N);

  switch (kernel) {
    case 1:
      elu_forward_kernel1<<<M * N / blockSize, blockSize>>>(inputGPU, outputGPU,
                                                            M, N);
      break;
    default:
      printf("Error: Invalid kernel type: %i\n", kernel);
      return EXIT_FAILURE;
  }
  cudaErrorCheck(cudaMemcpy(resFromGPU, outputGPU, M * N * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cudaErrorCheck(cudaDeviceSynchronize());

  if (checkResults(output, resFromGPU, M * N)) {
    switch (kernel) {
      case 1:
        benchmarkKernel(repeatTimes, elu_forward_kernel1, M * N / blockSize,
                        blockSize, 0, 0, &elapsedTime, inputGPU, outputGPU, M,
                        N);
        break;
    }
    printf(
        "elu_forward kernel: %i | matrixSize: %i x %i | Times: %f ms | "
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
