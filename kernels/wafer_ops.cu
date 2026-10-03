// Custom CUDA kernels for wafer-map preprocessing and filtering, exposed to PyTorch.
//
// Thread/block layout: one 2-D block of 16x16 threads computes a 16x16 output tile; grid.x and
// grid.y tile the image, grid.z is the batch index. Consecutive threads in a warp handle
// consecutive columns of a row, so global loads and stores are coalesced.
//
// The *_tiled kernels stage the input tile plus a one-pixel halo in shared memory, so each input
// value is read from global memory once per tile instead of up to nine times.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------------------------------------------------------- preprocess

// Nearest-neighbour resize (same index rule as PyTorch's "nearest-exact") into float32.
__global__ void resize_kernel(const uint8_t* __restrict__ in, float* __restrict__ out,
                              int n, const int* __restrict__ heights, const int* __restrict__ widths,
                              const long* __restrict__ offsets, int size) {
  int ox = blockIdx.x * TILE + threadIdx.x;
  int oy = blockIdx.y * TILE + threadIdx.y;
  int b = blockIdx.z;
  if (ox >= size || oy >= size) return;
  int h = heights[b], w = widths[b];
  // nearest-exact: src = floor((dst + 0.5) * scale)
  int sy = min((int)((oy + 0.5f) * h / size), h - 1);
  int sx = min((int)((ox + 0.5f) * w / size), w - 1);
  out[((long)b * size + oy) * size + ox] = (float)in[offsets[b] + (long)sy * w + sx];
}

// Per-wafer standardisation: one block per wafer, block-wide reduction for mean and variance.
__global__ void standardise_kernel(float* __restrict__ x, int size) {
  int b = blockIdx.x;
  int n = size * size;
  float* img = x + (long)b * n;
  __shared__ float s_sum[256];
  __shared__ float s_sq[256];
  float sum = 0.f, sq = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = img[i];
    sum += v;
    sq += v * v;
  }
  s_sum[threadIdx.x] = sum;
  s_sq[threadIdx.x] = sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float mean = s_sum[0] / n;
  float var = s_sq[0] / n - mean * mean;
  float inv = 1.f / (sqrtf(fmaxf(var, 0.f)) + 1e-6f);
  for (int i = threadIdx.x; i < n; i += blockDim.x) img[i] = (img[i] - mean) * inv;
}

torch::Tensor preprocess(torch::Tensor flat, torch::Tensor heights, torch::Tensor widths,
                         torch::Tensor offsets, int size) {
  CHECK_CUDA(flat); CHECK_CONTIG(flat);
  int n = heights.size(0);
  auto out = torch::empty({n, size, size}, flat.options().dtype(torch::kFloat32));
  dim3 block(TILE, TILE);
  dim3 grid((size + TILE - 1) / TILE, (size + TILE - 1) / TILE, n);
  resize_kernel<<<grid, block>>>(flat.data_ptr<uint8_t>(), out.data_ptr<float>(), n,
                                 heights.data_ptr<int>(), widths.data_ptr<int>(),
                                 offsets.data_ptr<long>(), size);
  standardise_kernel<<<n, 256>>>(out.data_ptr<float>(), size);
  return out;
}

// ---------------------------------------------------------------- conv3x3

__global__ void conv3x3_naive_kernel(const float* __restrict__ in, float* __restrict__ out,
                                     const float* __restrict__ w, int H, int W) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  int b = blockIdx.z;
  if (x >= W || y >= H) return;
  const float* img = in + (long)b * H * W;
  float acc = 0.f;
#pragma unroll
  for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
      int yy = y + dy, xx = x + dx;
      float v = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[yy * W + xx] : 0.f;  // zero pad
      acc += v * w[(dy + 1) * 3 + (dx + 1)];
    }
  }
  out[((long)b * H + y) * W + x] = acc;
}

__global__ void conv3x3_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                     const float* __restrict__ w, int H, int W) {
  __shared__ float tile[TILE + 2][TILE + 2];
  __shared__ float sw[9];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * TILE + tx;
  int y = blockIdx.y * TILE + ty;
  int b = blockIdx.z;
  const float* img = in + (long)b * H * W;
  if (ty == 0 && tx < 9) sw[tx] = w[tx];
  // Load the tile plus halo: each thread loads its own pixel, edge threads also load the halo.
  for (int ly = ty; ly < TILE + 2; ly += TILE) {
    for (int lx = tx; lx < TILE + 2; lx += TILE) {
      int gy = blockIdx.y * TILE + ly - 1;
      int gx = blockIdx.x * TILE + lx - 1;
      tile[ly][lx] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? img[gy * W + gx] : 0.f;
    }
  }
  __syncthreads();
  if (x >= W || y >= H) return;
  float acc = 0.f;
#pragma unroll
  for (int dy = 0; dy < 3; ++dy)
#pragma unroll
    for (int dx = 0; dx < 3; ++dx) acc += tile[ty + dy][tx + dx] * sw[dy * 3 + dx];
  out[((long)b * H + y) * W + x] = acc;
}

torch::Tensor conv3x3(torch::Tensor x, torch::Tensor weight, bool tiled) {
  CHECK_CUDA(x); CHECK_CONTIG(x); CHECK_CUDA(weight);
  int N = x.size(0), H = x.size(1), W = x.size(2);
  auto w = weight.contiguous().to(torch::kFloat32);
  auto out = torch::empty_like(x);
  dim3 block(TILE, TILE);
  dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE, N);
  if (tiled)
    conv3x3_tiled_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), w.data_ptr<float>(), H, W);
  else
    conv3x3_naive_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), w.data_ptr<float>(), H, W);
  return out;
}

// ---------------------------------------------------------------- median3x3

__device__ __forceinline__ void sort2(float& a, float& b) {
  float lo = fminf(a, b), hi = fmaxf(a, b);
  a = lo; b = hi;
}

// Median of 9 with a fixed 19-comparison sorting network (branch-free, no local arrays in memory).
__device__ __forceinline__ float median9(float v[9]) {
  sort2(v[1], v[2]); sort2(v[4], v[5]); sort2(v[7], v[8]);
  sort2(v[0], v[1]); sort2(v[3], v[4]); sort2(v[6], v[7]);
  sort2(v[1], v[2]); sort2(v[4], v[5]); sort2(v[7], v[8]);
  sort2(v[0], v[3]); sort2(v[5], v[8]); sort2(v[4], v[7]);
  sort2(v[3], v[6]); sort2(v[1], v[4]); sort2(v[2], v[5]);
  sort2(v[4], v[7]); sort2(v[4], v[2]); sort2(v[6], v[4]);
  sort2(v[4], v[2]);
  return v[4];
}

__global__ void median3x3_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int H, int W) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  int b = blockIdx.z;
  if (x >= W || y >= H) return;
  const float* img = in + (long)b * H * W;
  float v[9];
  int k = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int yy = min(max(y + dy, 0), H - 1), xx = min(max(x + dx, 0), W - 1);  // replicate pad
      v[k++] = img[yy * W + xx];
    }
  out[((long)b * H + y) * W + x] = median9(v);
}

__global__ void median3x3_tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int H, int W) {
  __shared__ float tile[TILE + 2][TILE + 2];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * TILE + tx;
  int y = blockIdx.y * TILE + ty;
  int b = blockIdx.z;
  const float* img = in + (long)b * H * W;
  for (int ly = ty; ly < TILE + 2; ly += TILE)
    for (int lx = tx; lx < TILE + 2; lx += TILE) {
      int gy = min(max(blockIdx.y * TILE + ly - 1, 0), H - 1);
      int gx = min(max(blockIdx.x * TILE + lx - 1, 0), W - 1);
      tile[ly][lx] = img[gy * W + gx];
    }
  __syncthreads();
  if (x >= W || y >= H) return;
  float v[9];
  int k = 0;
#pragma unroll
  for (int dy = 0; dy < 3; ++dy)
#pragma unroll
    for (int dx = 0; dx < 3; ++dx) v[k++] = tile[ty + dy][tx + dx];
  out[((long)b * H + y) * W + x] = median9(v);
}

torch::Tensor median3x3(torch::Tensor x, bool tiled) {
  CHECK_CUDA(x); CHECK_CONTIG(x);
  int N = x.size(0), H = x.size(1), W = x.size(2);
  auto out = torch::empty_like(x);
  dim3 block(TILE, TILE);
  dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE, N);
  if (tiled)
    median3x3_tiled_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), H, W);
  else
    median3x3_naive_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), H, W);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "nearest resize + per-wafer standardise (CUDA)");
  m.def("conv3x3", &conv3x3, "3x3 zero-padded conv, naive or shared-memory tiled (CUDA)");
  m.def("median3x3", &median3x3, "3x3 replicate-padded median, naive or tiled (CUDA)");
}
