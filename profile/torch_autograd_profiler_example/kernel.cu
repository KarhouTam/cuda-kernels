#include <cuda_runtime.h>

template <typename T>
__device__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename T>
__device__ T warpReduceMax(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__global__ void softmax_kernel(float* input, float* output, const int M,
                               const int N) {
  const int tid = threadIdx.x;
  const int warpId = tid / warpSize;
  const int laneId = tid % warpSize;
  const int warpsPerBlock = blockDim.x / warpSize;
  const int numWarps = warpsPerBlock * gridDim.x;
  const int idx = warpsPerBlock * blockIdx.x + warpId;
  for (int m = idx; m < M; m += numWarps) {
    // each lane (thread in a warp) calculate the maxval among
    // data with indices [landId, landId + 32, laneId + 64, ...]
    const float* x = input + m * N;
    float* const y = output + m * N;

    float maxval = -INFINITY;
    for (int i = laneId; i < N; i += warpSize) {
      maxval = fmaxf(maxval, x[i]);
    }
    // warp-reduce to calculate the MAX of maxval among all lanes
    // and the 0-th lane will store the result
    maxval = warpReduceMax(maxval);

    float sum = 0.0f;
    for (int i = laneId; i < N; i += warpSize) {
      sum += expf(x[i] - maxval);
    }

    sum = warpReduceSum(sum);
    for (int i = laneId; i < N; i += warpSize) {
      y[i] = expf(x[i] - maxval) / sum;
    }
  }
}
