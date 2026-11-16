// CUDA implementation of the first-stage geometry linearization.
//
// Mirrors RegistrationMode::GeometryOnly in BIEVR/src/ls_optimizer.cpp:
//   * voxel lookup: exact cell, then the 7 neighbour probes of nearestVoxel()
//   * residual: r = p_O.z - bilinear(bump_smoothed)
//   * Jacobian: row 2 of the 3x6 SE(3) Jacobian minus grad(I) * topRows(2)
//   * Huber weight and cost, identical formula
//
// The bilinear sampler reproduces sampleValueAndGradient(): weight-masked, and it
// fails when the 2x2 footprint leaves the image or all four mask weights are zero.
#include "gpu_linearize.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gpu_geo {
namespace {

constexpr int kMaxBlocks = 1024;

__device__ __forceinline__ uint64_t expandBits(uint64_t v) {
  v = (v | (v << 32)) & 0x1f00000000ffffULL;
  v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
  v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
  v = (v | (v << 4)) & 0x10c30c30c30c3ULL;
  v = (v | (v << 2)) & 0x1249249249249249ULL;
  return v;
}

__device__ __forceinline__ uint64_t morton(int x, int y, int z) {
  const uint64_t ux = static_cast<uint64_t>(x + (1 << 20)) & 0x1FFFFFull;
  const uint64_t uy = static_cast<uint64_t>(y + (1 << 20)) & 0x1FFFFFull;
  const uint64_t uz = static_cast<uint64_t>(z + (1 << 20)) & 0x1FFFFFull;
  return expandBits(ux) | (expandBits(uy) << 1) | (expandBits(uz) << 2);
}

__device__ __forceinline__ bool sampleValue(const float* image, const float* weights,
                                            int rows, int cols, double x, double y,
                                            double& value) {
  const int x0 = static_cast<int>(floor(x));
  const int y0 = static_cast<int>(floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  if (x0 < 0 || y0 < 0 || x1 >= cols || y1 >= rows) return false;
  const double dx = x - x0, dy = y - y0;
  const double w00 = (1.0 - dx) * (1.0 - dy) * (weights[y0 * cols + x0] > 0.0f);
  const double w01 = dx * (1.0 - dy) * (weights[y0 * cols + x1] > 0.0f);
  const double w10 = (1.0 - dx) * dy * (weights[y1 * cols + x0] > 0.0f);
  const double w11 = dx * dy * (weights[y1 * cols + x1] > 0.0f);
  const double sum = w00 + w01 + w10 + w11;
  if (sum == 0.0) return false;
  value = (w00 * image[y0 * cols + x0] + w01 * image[y0 * cols + x1] +
           w10 * image[y1 * cols + x0] + w11 * image[y1 * cols + x1]) / sum;
  return true;
}

__device__ __forceinline__ bool sampleAndGradient(const float* image, const float* weights,
                                                  int rows, int cols, double x, double y,
                                                  double& value, double& dIdx, double& dIdy) {
  const int x0 = static_cast<int>(floor(x));
  const int y0 = static_cast<int>(floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const int max_x = cols - 1;
  const int max_y = rows - 1;
  if (x0 < 0 || y0 < 0 || x1 > max_x || y1 > max_y) return false;

  const double dx = x - x0;
  const double dy = y - y0;
  const double dx1 = 1.0 - dx;
  const double dy1 = 1.0 - dy;

  const int v00 = weights[y0 * cols + x0] > 0.f;
  const int v01 = weights[y0 * cols + x1] > 0.f;
  const int v10 = weights[y1 * cols + x0] > 0.f;
  const int v11 = weights[y1 * cols + x1] > 0.f;

  const double w0 = dx1 * dy1 * v00;
  const double w1 = dx * dy1 * v01;
  const double w2 = dx1 * dy * v10;
  const double w3 = dx * dy * v11;
  const double ws = w0 + w1 + w2 + w3;
  if (ws == 0.0) return false;

  const double m00 = image[y0 * cols + x0];
  const double m01 = image[y0 * cols + x1];
  const double m10 = image[y1 * cols + x0];
  const double m11 = image[y1 * cols + x1];
  value = (w0 * m00 + w1 * m01 + w2 * m10 + w3 * m11) / ws;

  // Central-difference style gradients over the full 2x2 stencil, matching the
  // weight-masked arithmetic of the CPU sampler.
  dIdx = 0.0;
  dIdy = 0.0;
  double minus = 0.0, plus = 0.0;
  if (x0 >= 1 && x1 + 1 <= max_x &&
      sampleValue(image, weights, rows, cols, x - 1.0, y, minus) &&
      sampleValue(image, weights, rows, cols, x + 1.0, y, plus))
    dIdx = 0.5 * (plus - minus);
  if (y0 >= 1 && y1 + 1 <= max_y &&
      sampleValue(image, weights, rows, cols, x, y - 1.0, minus) &&
      sampleValue(image, weights, rows, cols, x, y + 1.0, plus))
    dIdy = 0.5 * (plus - minus);
  return true;
}

__device__ int findVoxel(const uint64_t* keys, const VoxelPod* voxels,
                         int n_keys, const double* p_W,
                         double voxel_size, int& slot) {
  // Exact cell.
  const int ix = static_cast<int>(floor(p_W[0] / voxel_size));
  const int iy = static_cast<int>(floor(p_W[1] / voxel_size));
  const int iz = static_cast<int>(floor(p_W[2] / voxel_size));
  const uint64_t key = morton(ix, iy, iz);

  int lo = 0, hi = n_keys - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const uint64_t k = keys[mid];
    if (k == key) {
      slot = mid;
      return 1;
    }
    if (k < key) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  // nearestVoxel(): probe the 6 axial neighbours plus the cell itself, keep the
  // voxel whose centroid is closest to the point.
  const int kOffsets[7][3] = {{0, 0, 0},  {1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                              {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
  double best = 1e300;
  int best_slot = -1;
  for (int o = 0; o < 7; ++o) {
    const uint64_t k2 = morton(ix + kOffsets[o][0], iy + kOffsets[o][1], iz + kOffsets[o][2]);
    int lo2 = 0, hi2 = n_keys - 1;
    int found = -1;
    while (lo2 <= hi2) {
      const int mid = (lo2 + hi2) >> 1;
      const uint64_t k = keys[mid];
      if (k == k2) {
        found = mid;
        break;
      }
      if (k < k2) {
        lo2 = mid + 1;
      } else {
        hi2 = mid - 1;
      }
    }
    if (found < 0) continue;
    const double dx = p_W[0] - voxels[found].centroid[0];
    const double dy = p_W[1] - voxels[found].centroid[1];
    const double dz = p_W[2] - voxels[found].centroid[2];
    const double dist = dx * dx + dy * dy + dz * dz;
    // CPU nearestVoxel() uses <= and therefore chooses the later offset on ties.
    if (dist <= best) { best = dist; best_slot = found; }
  }
  if (best_slot < 0) return 0;
  slot = best_slot;
  return 1;
}

__global__ void linearizeKernel(const double* __restrict__ points, int n_points,
                                const double* __restrict__ T_W_L, double voxel_size,
                                double inv_px_size, double huber_delta,
                                const uint64_t* __restrict__ keys, int n_keys,
                                const VoxelPod* __restrict__ voxels,
                                const float* __restrict__ pool, int compute_jacobians,
                                double* __restrict__ H_out, double* __restrict__ b_out,
                                double* __restrict__ cost_out, int* __restrict__ count_out) {
  __shared__ double sH[21];
  __shared__ double sb[6];
  __shared__ double scost;
  __shared__ int scount;

  for (int i = threadIdx.x; i < 21; i += blockDim.x) sH[i] = 0.0;
  for (int i = threadIdx.x; i < 6; i += blockDim.x) sb[i] = 0.0;
  if (threadIdx.x == 0) {
    scost = 0.0;
    scount = 0;
  }
  __syncthreads();

  double H[21];
  double b[6];
  for (int i = 0; i < 21; ++i) H[i] = 0.0;
  for (int i = 0; i < 6; ++i) b[i] = 0.0;
  double cost = 0.0;
  int count = 0;

  const double R[9] = {T_W_L[0], T_W_L[1], T_W_L[2], T_W_L[4], T_W_L[5],
                       T_W_L[6], T_W_L[8], T_W_L[9], T_W_L[10]};
  const double t[3] = {T_W_L[3], T_W_L[7], T_W_L[11]};

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_points;
       idx += blockDim.x * gridDim.x) {
    const double p[3] = {points[3 * idx], points[3 * idx + 1], points[3 * idx + 2]};
    const double pW[3] = {R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0],
                          R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1],
                          R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]};

    int slot = -1;
    if (!findVoxel(keys, voxels, n_keys, pW, voxel_size, slot)) continue;
    const VoxelPod v = voxels[slot];

    // p_O = T_C_W * p_W
    const double* C = v.T_C_W;
    const double pO[3] = {C[0] * pW[0] + C[1] * pW[1] + C[2] * pW[2] + C[3],
                          C[4] * pW[0] + C[5] * pW[1] + C[6] * pW[2] + C[7],
                          C[8] * pW[0] + C[9] * pW[1] + C[10] * pW[2] + C[11]};
    const double x = pO[0] * inv_px_size;
    const double y = pO[1] * inv_px_size;

    double I = 0.0, dIdx = 0.0, dIdy = 0.0;
    if (!sampleAndGradient(pool + v.image_offset, pool + v.weight_offset, v.height, v.width, x,
                           y, I, dIdx, dIdy)) {
      continue;
    }

    const double r = pO[2] - I;
    const double abs_r = fabs(r);
    const double w = abs_r <= huber_delta ? 1.0 : huber_delta / abs_r;
    cost += abs_r <= huber_delta ? 0.5 * r * r : huber_delta * (abs_r - 0.5 * huber_delta);
    ++count;

    if (!compute_jacobians) continue;

    // J = row2(SE3) - (grad I) * topRows(2), with the gradient scaled by inv_px.
    const double gx = dIdx * inv_px_size;
    const double gy = dIdy * inv_px_size;
    double J[6];
    // rotation block of SE3 for this point: -R_o * [p]_x, using rows of C*R
    // rotation rows of (C_rot * R)
    double RcR[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        RcR[3 * i + j] = C[4 * i] * R[j] + C[4 * i + 1] * R[3 + j] + C[4 * i + 2] * R[6 + j];
      }
    }
    // skew(p) = [[0,-pz,py],[pz,0,-px],[-py,px,0]]
    const double px = p[0], py = p[1], pz = p[2];
    // SE3_Jac top-left = -R_o * skew(p), top-right = -R_o? No: the code uses
    //   block<3,3>(0,3) = R_o   (translation part)
    //   block<3,3>(0,0) = -R_o * skew(p)
    // where R_o = T_C_W.linear() * T_W_L.linear() = RcR.
    double SE3[3][6];
    for (int i = 0; i < 3; ++i) {
      // rotation columns
      SE3[i][0] = -RcR[3 * i + 1] * pz + RcR[3 * i + 2] * py;
      SE3[i][1] = RcR[3 * i + 0] * pz - RcR[3 * i + 2] * px;
      SE3[i][2] = -RcR[3 * i + 0] * py + RcR[3 * i + 1] * px;
      // translation columns
      SE3[i][3] = RcR[3 * i + 0];
      SE3[i][4] = RcR[3 * i + 1];
      SE3[i][5] = RcR[3 * i + 2];
    }
    for (int k = 0; k < 6; ++k) {
      J[k] = SE3[2][k] - (gx * SE3[0][k] + gy * SE3[1][k]);
    }
    // H += w * J^T J (upper triangle), b += w * J^T r
    int at = 0;
    for (int i = 0; i < 6; ++i) {
      for (int j = i; j < 6; ++j) {
        H[at++] += w * J[i] * J[j];
      }
      b[i] += w * J[i] * r;
    }
  }

  // Block reduction.
  for (int i = 0; i < 21; ++i) atomicAdd(&sH[i], H[i]);
  for (int i = 0; i < 6; ++i) atomicAdd(&sb[i], b[i]);
  atomicAdd(&scost, cost);
  atomicAdd(&scount, count);
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < 21; ++i) atomicAdd(&H_out[i], sH[i]);
    for (int i = 0; i < 6; ++i) atomicAdd(&b_out[i], sb[i]);
    atomicAdd(cost_out, scost);
    atomicAdd(count_out, scount);
  }
}

uint64_t* g_keys = nullptr;
VoxelPod* g_voxels = nullptr;
float* g_pool = nullptr;
int g_n_keys = 0;
int g_pool_size = 0;

double* g_points = nullptr;
int g_points_capacity = 0;
double* g_pose = nullptr;
double* g_H = nullptr;
double* g_b = nullptr;
double* g_cost = nullptr;
int* g_count = nullptr;

double g_kernel_ms = 0.0;
double g_transfer_ms = 0.0;

bool ensureScratch(int n_points) {
  if (g_points_capacity >= n_points) return true;
  if (g_points) cudaFree(g_points);
  g_points = nullptr;
  g_points_capacity = n_points;
  if (cudaMalloc(&g_points, sizeof(double) * 3 * n_points) != cudaSuccess) {
    g_points_capacity = 0;
    return false;
  }
  return true;
}

}  // namespace

bool uploadMap(const uint64_t* keys, const VoxelPod* voxels, int n_voxels, const float* pool,
               int pool_size) {
  releaseMap();
  if (!keys || !voxels || !pool || n_voxels <= 0 || pool_size <= 0) return false;
  g_n_keys = n_voxels;
  g_pool_size = pool_size;
  const bool ok =
      cudaMalloc(&g_keys, sizeof(uint64_t) * n_voxels) == cudaSuccess &&
      cudaMalloc(&g_voxels, sizeof(VoxelPod) * n_voxels) == cudaSuccess &&
      cudaMalloc(&g_pool, sizeof(float) * pool_size) == cudaSuccess &&
      cudaMemcpy(g_keys, keys, sizeof(uint64_t) * n_voxels, cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaMemcpy(g_voxels, voxels, sizeof(VoxelPod) * n_voxels, cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaMemcpy(g_pool, pool, sizeof(float) * pool_size, cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaMalloc(&g_pose, sizeof(double) * 12) == cudaSuccess &&
      cudaMalloc(&g_H, sizeof(double) * 21) == cudaSuccess &&
      cudaMalloc(&g_b, sizeof(double) * 6) == cudaSuccess &&
      cudaMalloc(&g_cost, sizeof(double)) == cudaSuccess &&
      cudaMalloc(&g_count, sizeof(int)) == cudaSuccess;
  if (!ok) { releaseMap(); return false; }
  return true;
}

void releaseMap() {
  if (g_keys) cudaFree(g_keys);
  if (g_voxels) cudaFree(g_voxels);
  if (g_pool) cudaFree(g_pool);
  if (g_pose) cudaFree(g_pose);
  if (g_H) cudaFree(g_H);
  if (g_b) cudaFree(g_b);
  if (g_cost) cudaFree(g_cost);
  if (g_count) cudaFree(g_count);
  g_keys = nullptr;
  g_voxels = nullptr;
  g_pool = nullptr;
  g_pose = g_H = g_b = g_cost = nullptr;
  g_count = nullptr;
  g_n_keys = 0;
  g_pool_size = 0;
  g_points_capacity = 0;
  if (g_points) cudaFree(g_points);
  g_points = nullptr;
}

bool linearize(const double* points, int n_points, const double* T_W_L, double voxel_size,
               double inv_px_size, double huber_delta, bool compute_jacobians,
               LinearizeResult& out) {
  if (!g_keys || g_n_keys <= 0) return false;
  if (!points || !T_W_L || n_points <= 0) return false;
  if (!ensureScratch(n_points)) return false;

  cudaEvent_t e0, e1;
  if (cudaEventCreate(&e0) != cudaSuccess) return false;
  if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); return false; }

  cudaEventRecord(e0);
  const bool prepared =
      cudaMemcpy(g_points, points, sizeof(double) * 3 * n_points, cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaMemcpy(g_pose, T_W_L, sizeof(double) * 12, cudaMemcpyHostToDevice) == cudaSuccess &&
      cudaMemset(g_H, 0, sizeof(double) * 21) == cudaSuccess &&
      cudaMemset(g_b, 0, sizeof(double) * 6) == cudaSuccess &&
      cudaMemset(g_cost, 0, sizeof(double)) == cudaSuccess &&
      cudaMemset(g_count, 0, sizeof(int)) == cudaSuccess;
  if (!prepared) { cudaEventDestroy(e0); cudaEventDestroy(e1); return false; }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float transfer = 0.f;
  cudaEventElapsedTime(&transfer, e0, e1);

  const int threads = 256;
  const int blocks = std::min(kMaxBlocks, (n_points + threads - 1) / threads);
  cudaEventRecord(e0);
  linearizeKernel<<<blocks, threads>>>(g_points, n_points, g_pose, voxel_size, inv_px_size,
                                       huber_delta, g_keys, g_n_keys, g_voxels, g_pool,
                                       compute_jacobians ? 1 : 0, g_H, g_b, g_cost, g_count);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  if (cudaGetLastError() != cudaSuccess) {
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return false;
  }
  float kernel = 0.f;
  cudaEventElapsedTime(&kernel, e0, e1);

  const bool downloaded =
      cudaMemcpy(out.H, g_H, sizeof(double) * 21, cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(out.b, g_b, sizeof(double) * 6, cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(&out.cost, g_cost, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(&out.count, g_count, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
  if (!downloaded) { cudaEventDestroy(e0); cudaEventDestroy(e1); return false; }

  g_kernel_ms = kernel;
  g_transfer_ms = transfer;
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return true;
}

double lastKernelMs() { return g_kernel_ms; }
double lastTransferMs() { return g_transfer_ms; }

}  // namespace gpu_geo
