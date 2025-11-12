
#include <stdio.h>
#define NUM_TENSOR 1024
#define SIZE_TENSOR 128

struct TensorListMetaData {
  float* addresses[3][NUM_TENSOR];
};

void multi_tensor_add_cpu(float** src, float** other, float** dst, int size) {
  for (int j = 0; j < NUM_TENSOR; ++j) {
    for (int i = 0; i < size; ++i) {
      dst[j][i] = src[j][i] + other[j][i];
    }
  }
}

__global__ void multi_tensor_add_kernel_nested_array(float** src, float** other,
                                                     float** dst, int size) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int j = 0; j < NUM_TENSOR; ++j) {
    if (idx < size) {
      for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        dst[j][i] = src[j][i] + other[j][i];
      }
    }
  }
}

__global__ void multi_tensor_add_kernel_meta_data(TensorListMetaData* meta,
                                                  int size) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int j = 0; j < NUM_TENSOR; ++j) {
    float* const src = meta->addresses[0][j];
    float* const other = meta->addresses[1][j];
    float* const dst = meta->addresses[2][j];
    if (idx < size) {
      for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        dst[i] = src[i] + other[i];
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    printf("Error: Missing input argument for mode.\n");
    return -1;
  }
  const int mode = argv[1][0] - '0';
  const size_t bytes_for_each_tensor = sizeof(float) * SIZE_TENSOR;
  float** src = (float**)malloc(sizeof(float*) * NUM_TENSOR);
  float** other = (float**)malloc(sizeof(float*) * NUM_TENSOR);
  float** dst = (float**)malloc(sizeof(float*) * NUM_TENSOR);
  for (int i = 0; i < NUM_TENSOR; ++i) {
    src[i] = (float*)malloc(bytes_for_each_tensor);
    other[i] = (float*)malloc(bytes_for_each_tensor);
    dst[i] = (float*)malloc(bytes_for_each_tensor);
    for (int j = 0; j < SIZE_TENSOR; ++j) {
      src[i][j] = rand() % 100 / 100.0f;
      other[i][j] = (float)(rand() % 100) / 100.0f;
      dst[i][j] = 0.0f;  // Initialize destination tensor
    }
  }
  multi_tensor_add_cpu(src, other, dst, SIZE_TENSOR);
  bool correct = true;
  if (mode == 0) {
    float **src_gpu, **other_gpu, **dst_gpu;
    cudaMalloc(&src_gpu, sizeof(float*) * NUM_TENSOR);
    cudaMalloc(&other_gpu, sizeof(float*) * NUM_TENSOR);
    cudaMalloc(&dst_gpu, sizeof(float*) * NUM_TENSOR);
    float* h_src_ptrs[NUM_TENSOR];
    float* h_other_ptrs[NUM_TENSOR];
    float* h_dst_ptrs[NUM_TENSOR];
    for (int i = 0; i < NUM_TENSOR; ++i) {
      float *srcArray, *otherArray, *dstArray;
      cudaMalloc(&srcArray, bytes_for_each_tensor);
      cudaMalloc(&otherArray, bytes_for_each_tensor);
      cudaMalloc(&dstArray, bytes_for_each_tensor);
      cudaMemcpy(srcArray, src[i], bytes_for_each_tensor,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(otherArray, other[i], bytes_for_each_tensor,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(dstArray, dst[i], bytes_for_each_tensor,
                 cudaMemcpyHostToDevice);
      h_src_ptrs[i] = srcArray;
      h_other_ptrs[i] = otherArray;
      h_dst_ptrs[i] = dstArray;
    }
    cudaMemcpy(src_gpu, h_src_ptrs, sizeof(float*) * NUM_TENSOR,
               cudaMemcpyHostToDevice);
    cudaMemcpy(other_gpu, h_other_ptrs, sizeof(float*) * NUM_TENSOR,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dst_gpu, h_dst_ptrs, sizeof(float*) * NUM_TENSOR,
               cudaMemcpyHostToDevice);

    multi_tensor_add_kernel_nested_array<<<32, 32>>>(src_gpu, other_gpu,
                                                     dst_gpu, SIZE_TENSOR);
    cudaDeviceSynchronize();
    float* result = (float*)malloc(bytes_for_each_tensor);
    // Copy device pointer arrays back to host
    float* h_dst_ptrs_check[NUM_TENSOR];
    cudaMemcpy(h_dst_ptrs_check, dst_gpu, sizeof(float*) * NUM_TENSOR,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_TENSOR; ++i) {
      cudaMemcpy(result, h_dst_ptrs_check[i], bytes_for_each_tensor,
                 cudaMemcpyDeviceToHost);
      for (int j = 0; j < SIZE_TENSOR; ++j) {
        if (fabs(result[j] - dst[i][j]) > 1e-5) {
          correct = false;
          printf("Mismatch at tensor %d, index %d: CPU=%f, GPU=%f\n", i, j,
                 dst[i][j], result[j]);
        }
      }
    }
    free(result);
  } else {
    TensorListMetaData* meta =
        (TensorListMetaData*)malloc(sizeof(TensorListMetaData));
    for (int i = 0; i < NUM_TENSOR; ++i) {
      cudaMalloc(&(meta->addresses[0][i]), bytes_for_each_tensor);
      cudaMalloc(&(meta->addresses[1][i]), bytes_for_each_tensor);
      cudaMalloc(&(meta->addresses[2][i]), bytes_for_each_tensor);
      cudaMemcpy(meta->addresses[0][i], src[i], bytes_for_each_tensor,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(meta->addresses[1][i], other[i], bytes_for_each_tensor,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(meta->addresses[2][i], dst[i], bytes_for_each_tensor,
                 cudaMemcpyHostToDevice);
    }
    TensorListMetaData* d_meta;
    cudaMalloc(&d_meta, sizeof(TensorListMetaData));
    cudaMemcpy(d_meta, meta, sizeof(TensorListMetaData),
               cudaMemcpyHostToDevice);
    multi_tensor_add_kernel_meta_data<<<32, 32>>>(d_meta, SIZE_TENSOR);
    cudaDeviceSynchronize();
    TensorListMetaData* h_meta =
        (TensorListMetaData*)malloc(sizeof(TensorListMetaData));
    cudaMemcpy(h_meta, d_meta, sizeof(TensorListMetaData),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_TENSOR; ++i) {
      h_meta->addresses[0][i] = (float*)malloc(bytes_for_each_tensor);
      h_meta->addresses[1][i] = (float*)malloc(bytes_for_each_tensor);
      h_meta->addresses[2][i] = (float*)malloc(bytes_for_each_tensor);
      cudaMemcpy(h_meta->addresses[0][i], meta->addresses[0][i],
                 bytes_for_each_tensor, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_meta->addresses[1][i], meta->addresses[1][i],
                 bytes_for_each_tensor, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_meta->addresses[2][i], meta->addresses[2][i],
                 bytes_for_each_tensor, cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < NUM_TENSOR; ++i) {
      for (int j = 0; j < SIZE_TENSOR; ++j) {
        if (fabs(h_meta->addresses[2][i][j] - dst[i][j]) > 1e-5) {
          correct = false;
          printf("Mismatch at tensor %d, index %d: CPU=%f, GPU=%f\n", i, j,
                 dst[i][j], h_meta->addresses[2][i][j]);
        }
      }
    }
    for (int i = 0; i < NUM_TENSOR; ++i) {
      free(h_meta->addresses[0][i]);
      free(h_meta->addresses[1][i]);
      free(h_meta->addresses[2][i]);
    }
    free(h_meta);
    cudaFree(d_meta);
    free(meta);
  }

  if (correct)
    printf("CPU and GPU results match!\n");
  else
    printf("CPU and GPU results do not match!\n");

  return 0;
}