#include <torch/types.h>

__global__ void softmax_kernel(float* input, float* output, const int M,
                               const int N);

torch::Tensor softmax(torch::Tensor input) {
  const int blockSize = 128;
  const auto M = input.size(0);
  const auto N = input.size(1);

  auto result = torch::empty_like(input);

  dim3 blockDim(blockSize);
  dim3 gridDim(M * 32 / blockDim.x);

  softmax_kernel<<<gridDim, blockDim>>>(input.data_ptr<float>(),
                                        result.data_ptr<float>(), M, N);
  return result;
}
