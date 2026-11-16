// GPU prototype for the first-stage (geometry-only) registration.
//
// One thread per source point computes the voxel correspondence, the height-map
// residual, the 1x6 geometric Jacobian and the Huber weight; a block reduction
// produces the 6x6 H_g (21 upper-triangular entries), the 6x1 b_g and the cost.
// The 6x6 LM solve stays on the CPU.
#ifndef GPU_LINEARIZE_H_
#define GPU_LINEARIZE_H_

#include <cstdint>

namespace gpu_geo {

// Flat, transferable copy of one voxel. Mirrors the fields the CPU linearize()
// reads: the plane-to-world transform, the smoothed height image and its mask.
struct VoxelPod {
  double T_C_W[12];  // 3x4 row-major R|t of the image frame
  double centroid[3];
  int width;
  int height;
  int image_offset;  // into the float pool (height values, row-major)
  int weight_offset; // into the float pool (mask, same layout)
};

// Result of one linearization pass.
struct LinearizeResult {
  double cost = 0.0;
  int count = 0;      // points that produced a residual
  double H[21] = {};  // upper triangle, row-major: (0,0)(0,1)...(5,5)
  double b[6] = {};
};

// Upload the map once per frame. Keys must be sorted; keys[s] belongs to voxels[s].
bool uploadMap(const uint64_t* keys, const VoxelPod* voxels, int n_voxels,
               const float* pool, int pool_size);

void releaseMap();

// One geometry linearization at the given pose.
//   compute_jacobians=true  -> H, b and cost
//   compute_jacobians=false -> cost only (LM trial evaluation)
// Returns false when the map has not been uploaded.
bool linearize(const double* points, int n_points, const double* T_W_L,
               double voxel_size, double inv_px_size, double huber_delta,
               bool compute_jacobians, LinearizeResult& out);

// Device-side timings of the last call, milliseconds.
double lastKernelMs();
double lastTransferMs();

}  // namespace gpu_geo

#endif  // GPU_LINEARIZE_H_
