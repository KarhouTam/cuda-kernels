#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <cub/cub.cuh>

#include "common.h"

/* Attention Forward Implementation

  Implements scaled dot-product attention: softmax(Q @ K^T / sqrt(D)) @ V

  Usage: ./attn_forward <kernel> [blockSize] [benchmarkRepeatTimes]
  e.g. ./attn_forward 1
       ./attn_forward 3 128 200

  Q, K, V are the query, key and value matrices
  out is the output matrix
  B is batch size, H is number of heads, T is sequence length, D is head
  dimension

  attn_forward_cpu(): CPU reference implementation using three-pass softmax
  (find max, compute exp sum, normalize and multiply V).

  attn_forward_kernel1(): Naive CUDA implementation. One thread handles one
  output element (b, h, t_q, d). Two passes over all T keys: first pass finds
  the max score for numerical stability, second pass computes softmax and
  accumulates the V-weighted sum. No shared memory — every thread re-reads K
  and V from global memory.

  attn_forward_kernel2(): One warp per query row. Each warp (32 threads)
  processes one output row (b, h, t_q, :). Threads within the warp collaborate
  on dot products via warpReduceSum, then use online softmax to process keys
  one at a time. Compared to kernel1, this introduces warp-level parallelism
  and online softmax (single pass over keys), but still reads K and V from
  global memory for every query row.

  attn_forward_kernel3(): Flash Attention V1 (tiled + online softmax).
  Splits Q into blocks of Br rows and K/V into blocks of Bc rows, loading each
  tile into shared memory. The attention scores S = Q_tile @ K_tile^T are
  stored in shared memory, then online softmax is applied row-wise to update
  running statistics across tiles. V_tile is then loaded into the same buffer
  that held K_tile, and output accumulators are updated using S and V_tile.
  This avoids materializing the full TxT attention matrix in HBM — each K/V
  tile is read from global memory only once per Q tile.
  Parameters: Br=32, Bc=32, blockDim=128 (4 warps), each warp handles 8 query
  rows. Shared memory: Q_tile[Br*D] + S[Br*Bc] + Kt_Vt[Bc*D] = 20 KB.

  Note: at small T (e.g. 128), kernel3 may be slower than kernel2 due to
  tiling overhead (shared memory loads, extra __syncthreads, S matrix
  management). Flash Attention's benefits become decisive at larger sequence
  lengths (T >= 512) where the O(T^2) HBM traffic of kernel2 dominates.
*/

void attn_forward_cpu(float* Q, float* K, float* V, float* out, int B, int H,
                      int T, int D) {
  // Shape: Q,K,V: [B,H,T,D], out: [B,H,T,D]
  float* tmp = new float[T * T];  // Store attention scores Q @ K^T

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      // Compute attention scores: Q @ K^T / sqrt(D)
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

      // Softmax over keys (row-wise)
      for (int i = 0; i < T; i++) {
        float max_val = tmp[i * T];
        for (int j = 1; j < T; j++) {
          max_val = fmaxf(max_val, tmp[i * T + j]);
        }

        float s = 0.0f;
        for (int j = 0; j < T; j++) {
          tmp[i * T + j] = expf(tmp[i * T + j] - max_val);
          s += tmp[i * T + j];
        }

        for (int j = 0; j < T; j++) {
          tmp[i * T + j] /= s;
        }
      }

      // Multiply softmax probabilities with V
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
  // Naive implementation: one thread handles one output element
  // Two passes over all keys — no scratch memory needed
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = B * H * T * D;

  if (idx >= total)
    return;

  // Decode indices: output element at position (b, h, t_q, d)
  int d = idx % D;
  int t_q = (idx / D) % T;
  int h = (idx / (T * D)) % H;
  int b = idx / (H * T * D);

  int headOffset = b * H * T * D + h * T * D;
  float* q = Q + headOffset + t_q * D;
  float* k = K + headOffset;
  float* v = V + headOffset;

  float scale = rsqrtf((float)D);

  // Pass 1: find max score for numerical stability
  float maxval = -INFINITY;
  for (int t_k = 0; t_k < T; t_k++) {
    float dot = 0.0f;
    for (int j = 0; j < D; j++) {
      dot += q[j] * k[t_k * D + j];
    }
    maxval = fmaxf(maxval, dot * scale);
  }

  // Pass 2: compute softmax sum and accumulate V-weighted output
  float softmaxSum = 0.0f;
  float outVal = 0.0f;
  for (int t_k = 0; t_k < T; t_k++) {
    float dot = 0.0f;
    for (int j = 0; j < D; j++) {
      dot += q[j] * k[t_k * D + j];
    }
    float p = expf(dot * scale - maxval);
    softmaxSum += p;
    outVal += p * v[t_k * D + d];
  }

  out[idx] = outVal / softmaxSum;
}

__global__ void attn_forward_kernel2(float* Q, float* K, float* V, float* out,
                                     int B, int H, int T, int D) {
  // One warp per query row
  // Each warp processes one output row (b, h, t_q, :) using online softmax
  // Threads in the warp collaborate on dot products via warpReduceSum,
  // then independently update their portion of the output vector.
  int tid = threadIdx.x;
  int warpId = tid / warpSize;
  int laneId = tid % warpSize;
  int warpsPerBlock = blockDim.x / warpSize;
  int numWarps = warpsPerBlock * gridDim.x;
  int globalWarpIdx = warpsPerBlock * blockIdx.x + warpId;

  float scale = rsqrtf((float)D);

  // Each warp takes one query row; stride across all rows in the batch
  for (int row = globalWarpIdx; row < B * H * T; row += numWarps) {
    int t_q = row % T;
    int h = (row / T) % H;
    int b = row / (H * T);

    int headOffset = b * H * T * D + h * T * D;
    float* q = Q + headOffset + t_q * D;
    float* k = K + headOffset;
    float* v = V + headOffset;
    float* o = out + headOffset + t_q * D;

    // Per-warp online softmax state
    float maxval = -INFINITY;
    float sumval = 0.0f;

    // Each lane accumulates its portion of the D-dimensional output
    const int dimsPerLane = D / warpSize;
    float outAccum[4];  // up to D/warpSize = 128/32 = 4 for D <= 128
    for (int i = 0; i < dimsPerLane; i++) {
      outAccum[i] = 0.0f;
    }

    // Process all keys sequentially with online softmax
    for (int t_k = 0; t_k < T; t_k++) {
      // Compute partial dot product for this lane's dimensions
      float dot = 0.0f;
      for (int j = laneId; j < D; j += warpSize) {
        dot += q[j] * k[t_k * D + j];
      }
      float score = warpReduceSum(dot) * scale;

      // Online softmax update: rescale running sum and output
      float newMax = fmaxf(maxval, score);
      float correction = expf(maxval - newMax);
      sumval = sumval * correction + expf(score - newMax);

      // Update output accumulator for this lane's dimensions
      for (int j = laneId; j < D; j += warpSize) {
        int idx = j / warpSize;
        outAccum[idx] =
            outAccum[idx] * correction + expf(score - newMax) * v[t_k * D + j];
      }
      maxval = newMax;
    }

    // Write normalized output for this lane's dimensions
    for (int j = laneId; j < D; j += warpSize) {
      o[j] = outAccum[j / warpSize] / sumval;
    }
  }
}

__global__ void attn_forward_kernel3(float* Q, float* K, float* V, float* out,
                                     int B, int H, int T, int D) {
  // Flash Attention V1: tiled matrix multiply with online softmax
  //
  // One thread block handles one Q tile of Br rows for one (batch, head).
  // Inner loop streams K/V tiles through shared memory, accumulating output
  // via online softmax without materializing the full TxT attention matrix.
  //
  // Shared memory layout (20 KB total for Br=32, Bc=32, D=64):
  //   Q_tile  [Br, D]   — query tile, loaded once, stays resident
  //   S       [Br, Bc]  — attention scores Q_tile @ K_tile^T
  //   KV_buf  [Bc, D]   — first holds K tile, then reloaded as V tile
  //
  // Block mapping: each warp handles ceilDiv(Br, warpsPerBlock) query rows.
  // Arrays are sized for the worst case (blockDim >= 64 → ≤16 rows per warp).

  const int Br = 32;  // query rows per tile
  const int Bc = 32;  // key/value rows per tile

  extern __shared__ float smem[];
  float* Q_tile = smem;                     // [Br, D]
  float* S = smem + Br * D;                 // [Br, Bc]
  float* KV_buf = smem + Br * D + Br * Bc;  // [Bc, D]

  const int warpsPerBlock = blockDim.x / warpSize;
  // Require Br to be evenly divisible by warpsPerBlock so each warp
  // processes the same number of query rows (no partial-warps tail).
  // Works for blockDim: 64 (2 warps × 16 rows), 128 (4 × 8), 256 (8 × 4).
  if (Br % warpsPerBlock != 0)
    return;
  const int rowsPerWarp = Br / warpsPerBlock;
  const int warpId = threadIdx.x / warpSize;
  const int laneId = threadIdx.x % warpSize;
  const int dimsPerLane = D / warpSize;  // 2 for D=64

  // Which (batch, head, Q-tile) this block handles
  int numQBlocks = ceilDiv(T, Br);
  int blockIdx_bh = blockIdx.x / numQBlocks;
  int qBlockIdx = blockIdx.x % numQBlocks;
  int b = blockIdx_bh / H;
  int h = blockIdx_bh % H;

  int headOffset = b * H * T * D + h * T * D;
  float* Q_head = Q + headOffset;
  float* K_head = K + headOffset;
  float* V_head = V + headOffset;
  float* O_head = out + headOffset;

  int qStart = qBlockIdx * Br;

  // ---- Load Q tile into shared memory (stays resident) ----
  for (int i = threadIdx.x; i < Br * D; i += blockDim.x) {
    int qr = i / D;
    int col = i % D;
    int globalRow = qStart + qr;
    Q_tile[i] = (globalRow < T) ? Q_head[globalRow * D + col] : 0.0f;
  }
  __syncthreads();

  // Per-query-row running state in registers.
  // Sized for worst case: blockDim >= 64 → warpsPerBlock >= 2 → rowsPerWarp
  // <= 16.
  float maxvals[16];  // running max per row handled by this warp
  float sumvals[16];  // running softmax denominator per row
  float oldMax[16];   // max before processing current tile (for rescaling)
  // Output accumulator: rowsPerWarp x dimsPerLane per thread
  float outAccum[16][4];  // D/32 <= 4 for D <= 128

  for (int r = 0; r < rowsPerWarp; r++) {
    maxvals[r] = -INFINITY;
    sumvals[r] = 0.0f;
    for (int d = 0; d < dimsPerLane; d++) {
      outAccum[r][d] = 0.0f;
    }
  }

  float scale = rsqrtf((float)D);
  int numKBlocks = ceilDiv(T, Bc);

  // ---- Outer loop: stream K/V tiles through shared memory ----
  for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
    int kStart = kBlock * Bc;
    int kRows = min(Bc, T - kStart);

    // ---- Phase 1: Load K tile ----
    for (int i = threadIdx.x; i < Bc * D; i += blockDim.x) {
      int kr = i / D;
      int col = i % D;
      int globalRow = kStart + kr;
      KV_buf[i] = (globalRow < T) ? K_head[globalRow * D + col] : 0.0f;
    }
    __syncthreads();

    // ---- Phase 2: Compute S = Q_tile x K_tile^T into shared memory ----
    // Each thread computes S entries indexed by its thread ID (strided).
    // Total S entries: Br x Bc = 1024. With 128 threads: 8 entries/thread.
    for (int idx = threadIdx.x; idx < Br * Bc; idx += blockDim.x) {
      int qi = idx / Bc;  // query row within tile
      int kj = idx % Bc;  // key row within tile
      float dot = 0.0f;
      for (int d = 0; d < D; d++) {
        dot += Q_tile[qi * D + d] * KV_buf[kj * D + d];
      }
      // Only write valid scores for rows within bounds
      S[qi * Bc + kj] =
          (kj < kRows && (qStart + qi) < T) ? dot * scale : -INFINITY;
    }
    __syncthreads();

    // ---- Phase 3: Online softmax row reduction ----
    // Each warp processes its rowsPerWarp rows independently.
    // For each row, lane j reads S[row][j], warp-reduces for max and sum.
    for (int r = 0; r < rowsPerWarp; r++) {
      int globalQR = qStart + warpId * rowsPerWarp + r;
      if (globalQR >= T)
        continue;

      int qi = warpId * rowsPerWarp + r;

      // Save old max for Phase 5 rescaling
      oldMax[r] = maxvals[r];

      // Lane j reads S[qi][j] from shared memory
      float myScore = (laneId < Bc) ? S[qi * Bc + laneId] : -INFINITY;

      // Warp-reduce to find tile-wide max
      float tileMax = warpReduceMax(myScore);

      // Compute new global max and tile sum
      float newMax = fmaxf(maxvals[r], tileMax);
      float myExp = expf(myScore - newMax);
      float tileSum = warpReduceSum(myExp);

      // Online softmax: rescale running sum, add tile contributions
      float correction = expf(maxvals[r] - newMax);
      sumvals[r] = sumvals[r] * correction + tileSum;
      maxvals[r] = newMax;
    }
    __syncthreads();

    // ---- Phase 4: Load V tile (overwrite K in KV_buf) ----
    for (int i = threadIdx.x; i < Bc * D; i += blockDim.x) {
      int kr = i / D;
      int col = i % D;
      int globalRow = kStart + kr;
      KV_buf[i] = (globalRow < T) ? V_head[globalRow * D + col] : 0.0f;
    }
    __syncthreads();
    float* V_tile = KV_buf;  // alias for clarity

    // ---- Phase 5: Accumulate output ----
    // For each query row: out += sum_j exp(S[qi][j] - max) * V_tile[j]
    // with correction factor applied to old accumulator.
    for (int r = 0; r < rowsPerWarp; r++) {
      int globalQR = qStart + warpId * rowsPerWarp + r;
      if (globalQR >= T)
        continue;

      int qi = warpId * rowsPerWarp + r;
      float rowMax = maxvals[r];

      // Rescale old output accumulators: correction = exp(oldMax - newMax)
      float correction = expf(oldMax[r] - rowMax);

      // Each lane accumulates its portion of the D-dimensional output
      for (int d = laneId, dIdx = 0; d < D; d += warpSize, dIdx++) {
        float accum = 0.0f;
        for (int j = 0; j < kRows; j++) {
          accum += expf(S[qi * Bc + j] - rowMax) * V_tile[j * D + d];
        }
        outAccum[r][dIdx] = outAccum[r][dIdx] * correction + accum;
      }
    }
    __syncthreads();
  }

  // ---- Write final output: normalize by softmax denominator ----
  for (int r = 0; r < rowsPerWarp; r++) {
    int globalQR = qStart + warpId * rowsPerWarp + r;
    if (globalQR >= T)
      continue;
    for (int d = laneId; d < D; d += warpSize) {
      O_head[globalQR * D + d] = outAccum[r][d / warpSize] / sumvals[r];
    }
  }
}

#define B 4
#define H 8
#define T 128
#define D 64
#define BLOCK_SIZE 128
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

  const int N = B * H * T * D;

  // Allocate host memory
  float* Q = (float*)malloc(N * sizeof(float));
  float* K = (float*)malloc(N * sizeof(float));
  float* V = (float*)malloc(N * sizeof(float));
  float* out = (float*)malloc(N * sizeof(float));
  float* resFromGPU = (float*)malloc(N * sizeof(float));

  // Initialize arrays
  initArrFloat(Q, N);
  initArrFloat(K, N);
  initArrFloat(V, N);
  zeros(out, N);

  // Allocate device memory
  float *Q_gpu, *K_gpu, *V_gpu, *out_gpu;
  cudaErrorCheck(cudaMalloc(&Q_gpu, N * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&K_gpu, N * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&V_gpu, N * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&out_gpu, N * sizeof(float)));

  // Copy data to device
  cudaErrorCheck(
      cudaMemcpy(Q_gpu, Q, N * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrorCheck(
      cudaMemcpy(K_gpu, K, N * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrorCheck(
      cudaMemcpy(V_gpu, V, N * sizeof(float), cudaMemcpyHostToDevice));

  float elapsedTime = 0.0f;

  // Compute reference result on CPU
  attn_forward_cpu(Q, K, V, out, B, H, T, D);

  // Run selected kernel
  switch (kernel) {
    case 1: {
      int gridSize = ceilDiv(N, blockSize);
      attn_forward_kernel1<<<gridSize, blockSize>>>(Q_gpu, K_gpu, V_gpu,
                                                    out_gpu, B, H, T, D);
      break;
    }
    case 2: {
      int warpsPerBlock = blockSize / 32;
      int numWarps = ceilDiv(B * H * T, warpsPerBlock);
      attn_forward_kernel2<<<numWarps, blockSize>>>(Q_gpu, K_gpu, V_gpu,
                                                    out_gpu, B, H, T, D);
      break;
    }
    case 3: {
      const int Br = 32;
      const int Bc = 32;
      int numQBlocks = ceilDiv(T, Br);
      int gridSize = B * H * numQBlocks;
      // Shared memory: Q_tile[Br*D] + S[Br*Bc] + KV_buf[Bc*D]
      int smemSize = (Br * D + Br * Bc + Bc * D) * sizeof(float);
      attn_forward_kernel3<<<gridSize, blockSize, smemSize>>>(
          Q_gpu, K_gpu, V_gpu, out_gpu, B, H, T, D);
      break;
    }
    default:
      printf("Error: Invalid kernel type: %d\n", kernel);
      return EXIT_FAILURE;
  }

  // Copy result back to host
  cudaErrorCheck(cudaMemcpy(resFromGPU, out_gpu, N * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cudaErrorCheck(cudaDeviceSynchronize());

  // Verify results and benchmark
  if (checkResults(out, resFromGPU, N)) {
    switch (kernel) {
      case 1: {
        int gridSize = ceilDiv(N, blockSize);
        benchmarkKernel(repeatTimes, attn_forward_kernel1, gridSize, blockSize,
                        0, 0, &elapsedTime, Q_gpu, K_gpu, V_gpu, out_gpu, B, H,
                        T, D);
        break;
      }
      case 2: {
        int warpsPerBlock = blockSize / 32;
        int numWarps = ceilDiv(B * H * T, warpsPerBlock);
        benchmarkKernel(repeatTimes, attn_forward_kernel2, numWarps, blockSize,
                        0, 0, &elapsedTime, Q_gpu, K_gpu, V_gpu, out_gpu, B, H,
                        T, D);
        break;
      }
      case 3: {
        const int Br = 32;
        const int Bc = 32;
        int numQBlocks = ceilDiv(T, Br);
        int gridSize = B * H * numQBlocks;
        int smemSize = (Br * D + Br * Bc + Bc * D) * sizeof(float);
        benchmarkKernel(repeatTimes, attn_forward_kernel3, gridSize, blockSize,
                        smemSize, 0, &elapsedTime, Q_gpu, K_gpu, V_gpu, out_gpu,
                        B, H, T, D);
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
