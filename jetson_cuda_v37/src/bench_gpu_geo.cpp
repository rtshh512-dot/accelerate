// Correctness + timing harness for the first-stage GPU prototype.
//
// Three things are compared at the same map, the same source cloud and the same
// starting pose:
//   1. CPU  : the real LsqRegistration(GeometryOnly) solve plus informationAt(T_g)
//   2. GPU  : the same LM loop driven by the CUDA linearization
//   3. checks: effective point count, cost, H_g and final pose
//
// The GPU loop replicates stepLm(): one H/b pass per outer iteration, trial
// evaluations for the damping search, and the same accept/reject rule.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "bievr_lio/bievr_map.h"
#include "bievr_lio/common.h"
#include "bievr_lio/ls_optimizer.h"

#include "gpu_linearize.h"

namespace {

using namespace bievr;

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void buildScene(Pointcloud& cloud, double spacing) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> jitter(-0.02, 0.02);
  const double length = 60.0, width = 8.0, height = 4.0;
  std::vector<Point> surfaces;
  for (double x = 0; x < length; x += spacing)
    for (double y = 0; y < width; y += spacing) {
      surfaces.push_back(Point(x, y, 0.0));
      surfaces.push_back(Point(x, y, height));
    }
  for (double x = 0; x < length; x += spacing)
    for (double z = 0; z < height; z += spacing) {
      surfaces.push_back(Point(x, 0.0, z));
      surfaces.push_back(Point(x, width, z));
    }
  for (double x = 7.0; x < length; x += 11.0)
    for (double y = 0; y < width; y += spacing)
      for (double z = 0; z < height; z += spacing) surfaces.push_back(Point(x, y, z));

  std::shuffle(surfaces.begin(), surfaces.end(), rng);
  cloud.resize(surfaces.size());
  for (size_t i = 0; i < surfaces.size(); ++i) {
    cloud.points().col(static_cast<Eigen::Index>(i)) =
        surfaces[i] + Point(jitter(rng), jitter(rng), jitter(rng));
  }
}

// Same Morton layout as the kernel (independent of the map's own hash, so the
// comparison does not depend on that implementation).
uint64_t expandBits(uint64_t v) {
  v = (v | (v << 32)) & 0x1f00000000ffffULL;
  v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
  v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
  v = (v | (v << 4)) & 0x10c30c30c30c3ULL;
  v = (v | (v << 2)) & 0x1249249249249249ULL;
  return v;
}

uint64_t mortonOf(const V3& p, double voxel_size) {
  const int x = static_cast<int>(std::floor(p.x() / voxel_size));
  const int y = static_cast<int>(std::floor(p.y() / voxel_size));
  const int z = static_cast<int>(std::floor(p.z() / voxel_size));
  const uint64_t ux = static_cast<uint64_t>(x + (1 << 20)) & 0x1FFFFFull;
  const uint64_t uy = static_cast<uint64_t>(y + (1 << 20)) & 0x1FFFFFull;
  const uint64_t uz = static_cast<uint64_t>(z + (1 << 20)) & 0x1FFFFFull;
  return expandBits(ux) | (expandBits(uy) << 1) | (expandBits(uz) << 2);
}

Matrix66 expandH(const double* upper) {
  Matrix66 H = Matrix66::Zero();
  int at = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      H(i, j) = upper[at];
      H(j, i) = upper[at];
      ++at;
    }
  }
  return H;
}

M3 so3Exp(const V3& w) {
  const double theta = w.norm();
  if (theta < 1e-12) {
    M3 R;
    R << 1, -w.z(), w.y(), w.z(), 1, -w.x(), -w.y(), w.x(), 1;
    return R;
  }
  const V3 a = w / theta;
  Eigen::AngleAxisd aa(theta, a);
  return aa.toRotationMatrix();
}

Transform applyDelta(const Transform& T, const V6& d) {
  Transform out = T;
  out.linear() = T.linear() * so3Exp(d.head<3>());
  out.translation() = T.translation() + T.linear() * d.tail<3>();
  return out;
}

bool isConverged(const Transform& delta, const RegistrationConfig& cfg) {
  const M3 rotation_error = delta.linear() - M3::Identity();
  const double r = rotation_error.cwiseAbs().maxCoeff() / cfg.rotation_epsilon;
  const double t = delta.translation().cwiseAbs().maxCoeff() /
                   cfg.transformation_epsilon;
  return std::max(r, t) < 1.0;
}

void packPose(const Transform& T, double out[12]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out[4 * r + c] = T.linear()(r, c);
    out[4 * r + 3] = T.translation()(r);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const size_t n_source = argc > 1 ? static_cast<size_t>(std::atoi(argv[1])) : 8000;

  const double voxel_size = 0.5;
  const double px_size = 0.05;
  const double huber = 0.1;

  Pointcloud cloud;
  buildScene(cloud, 0.2);
  BIEVRMap::Config map_cfg;
  map_cfg.voxel_size = voxel_size;
  map_cfg.px_size = px_size;
  map_cfg.smooth = true;
  map_cfg.weighted = true;
  map_cfg.intensity = false;
  BIEVRMap map(map_cfg);
  std::vector<double> ranges(cloud.size());
  for (size_t i = 0; i < cloud.size(); ++i) ranges[i] = cloud.points().col(i).norm();
  map.integratePoints(cloud, &ranges, nullptr);
  std::printf("map: %zu voxels\n", map.size());

  std::mt19937 rng(3);
  std::uniform_int_distribution<size_t> pick(0, cloud.size() - 1);
  std::normal_distribution<double> noise(0.0, 0.02);
  Pointcloud source;
  source.resize(n_source);
  std::vector<double> pts(3 * n_source);
  for (size_t i = 0; i < n_source; ++i) {
    const Point p = cloud.points().col(static_cast<Eigen::Index>(pick(rng))) +
                    Point(noise(rng), noise(rng), noise(rng));
    source.points().col(static_cast<Eigen::Index>(i)) = p;
    pts[3 * i] = p.x();
    pts[3 * i + 1] = p.y();
    pts[3 * i + 2] = p.z();
  }

  // Export every observed voxel. Source-only export loses the six-neighbor
  // fallback and changes correspondences as soon as the trial pose moves.
  std::vector<std::pair<uint64_t, const Voxel*>> entries;
  const double t_pack0 = nowMs();
  map.forEachVoxel([&](const Voxel& v) {
    if (v.num_points_ == 0) return;
    const V3 centroid = v.sum_ / static_cast<double>(v.num_points_);
    entries.emplace_back(map.hashIndex(centroid), &v);
  });
  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<uint64_t> keys;
  std::vector<gpu_geo::VoxelPod> pods;
  keys.reserve(entries.size());
  pods.reserve(entries.size());
  std::vector<float> pool;
  for (const auto& entry : entries) {
    const Voxel* voxel = entry.second;
    keys.push_back(entry.first);
    pods.emplace_back();
    gpu_geo::VoxelPod& pod = pods.back();
    const M3& R = voxel->T_C_W_.linear();
    const V3& t = voxel->T_C_W_.translation();
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) pod.T_C_W[4 * r + c] = R(r, c);
      pod.T_C_W[4 * r + 3] = t(r);
    }
    const V3 centroid = voxel->sum_ / static_cast<double>(voxel->num_points_);
    for (int j = 0; j < 3; ++j) pod.centroid[j] = centroid(j);
    pod.height = static_cast<int>(voxel->bump_smoothed_.rows());
    pod.width = static_cast<int>(voxel->bump_smoothed_.cols());
    pod.image_offset = static_cast<int>(pool.size());
    for (int r = 0; r < pod.height; ++r)
      for (int c = 0; c < pod.width; ++c) pool.push_back(voxel->bump_smoothed_(r, c));
    pod.weight_offset = static_cast<int>(pool.size());
    for (int r = 0; r < pod.height; ++r)
      for (int c = 0; c < pod.width; ++c) pool.push_back(voxel->bump_weights_(r, c));
  }
  const double pack_ms = nowMs() - t_pack0;
  std::printf("gpu map: %zu voxels, %zu floats (%.2f MB)\n", pods.size(), pool.size(),
              pool.size() * sizeof(float) / 1048576.0);
  std::printf("map pack %.3f ms\n", pack_ms);
  const double t_upload0 = nowMs();
  if (!gpu_geo::uploadMap(keys.data(), pods.data(), static_cast<int>(pods.size()), pool.data(),
                          static_cast<int>(pool.size()))) {
    std::fprintf(stderr, "CUDA map upload failed\n");
    return 1;
  }
  const double upload_ms = nowMs() - t_upload0;
  std::printf("map upload %.3f ms\n", upload_ms);

  Transform T_init = Transform::Identity();
  T_init.translation() = V3(0.03, -0.02, 0.04);

  RegistrationConfig cfg;

  // Compare one linearization at exactly the same pose before comparing LM.
  Matrix66 H0_cpu;
  V6 b0_cpu;
  LsqRegistration initial_reg(map, source, cfg, nullptr, RegistrationMode::GeometryOnly);
  const double cost0_cpu = initial_reg.linearizationAt(T_init, H0_cpu, b0_cpu);
  const int count0_cpu = initial_reg.numEffectivePoints();
  gpu_geo::LinearizeResult initial_gpu;
  double packed_pose[12];
  packPose(T_init, packed_pose);
  if (!gpu_geo::linearize(pts.data(), static_cast<int>(n_source), packed_pose,
                           voxel_size, 1.0 / px_size, huber, true, initial_gpu)) {
    std::fprintf(stderr, "initial CUDA linearization failed\n");
    return 1;
  }
  const double initial_h_diff = (expandH(initial_gpu.H) - H0_cpu).norm() /
                                std::max(1e-12, H0_cpu.norm());
  V6 b0_gpu;
  for (int i = 0; i < 6; ++i) b0_gpu(i) = initial_gpu.b[i];
  const double initial_b_diff = (b0_gpu - b0_cpu).norm() /
                                std::max(1e-12, b0_cpu.norm());
  const double initial_cost_diff = std::abs(initial_gpu.cost - cost0_cpu) /
                                   std::max(1e-12, std::abs(cost0_cpu));
  std::printf("initial linearization: count %d/%d, rel H %.3e, b %.3e, cost %.3e\n",
              count0_cpu, initial_gpu.count, initial_h_diff, initial_b_diff,
              initial_cost_diff);
  if (count0_cpu != initial_gpu.count || initial_h_diff > 1e-5 ||
      initial_b_diff > 1e-5 || initial_cost_diff > 1e-5) {
    std::fprintf(stderr, "CPU/GPU mismatch; stop before timing\n");
    return 2;
  }

  // ---- CPU reference --------------------------------------------------------
  Transform T_cpu = T_init;
  Matrix66 H_cpu = Matrix66::Zero();
  int count_cpu = 0;
  double cost_cpu = 0.0;
  double cpu_stage_ms = 0.0;
  {
    LsqRegistration reg(map, source, cfg, nullptr, RegistrationMode::GeometryOnly);
    const double t0 = nowMs();
    T_cpu = reg.computeTransformation(T_init);
    H_cpu = reg.informationAt(T_cpu);
    cpu_stage_ms = nowMs() - t0;
    count_cpu = reg.numEffectivePoints();
    const RegistrationResiduals res = reg.collectResiduals(T_cpu);
    for (double r : res.geometry) {
      const double a = std::abs(r);
      cost_cpu += a <= huber ? 0.5 * r * r : huber * (a - 0.5 * huber);
    }
  }

  // ---- GPU LM loop ----------------------------------------------------------
  Transform T_gpu = T_init;
  Matrix66 H_gpu = Matrix66::Zero();
  double lambda = -1.0;
  int anchor_iterations = 0;
  double gpu_stage_ms = 0.0;
  double gpu_kernel_ms_total = 0.0;
  double gpu_transfer_ms_total = 0.0;
  int count_gpu = 0;
  double cost_gpu = 0.0;
  {
    const double t0 = nowMs();
    bool converged = false;
    for (int it = 0; it < cfg.max_iterations && !converged; ++it) {
      gpu_geo::LinearizeResult lin;
      packPose(T_gpu, packed_pose);
      if (!gpu_geo::linearize(pts.data(), static_cast<int>(n_source), packed_pose,
                              voxel_size, 1.0 / px_size, huber, true, lin)) {
        std::printf("GPU linearize failed\n");
        return 1;
      }
      gpu_kernel_ms_total += gpu_geo::lastKernelMs();
      gpu_transfer_ms_total += gpu_geo::lastTransferMs();
      const Matrix66 H = expandH(lin.H);
      V6 b;
      for (int i = 0; i < 6; ++i) b(i) = lin.b[i];
      double y0 = lin.cost;
      count_gpu = lin.count;
      if (lambda < 0.0) {
        lambda = std::max(1e-9, cfg.lm_init_lambda_factor * H.diagonal().cwiseAbs().maxCoeff());
      }
      double nu = 2.0;
      bool accepted = false;
      for (int inner = 0; inner < cfg.lm_max_iterations; ++inner) {
        const Matrix66 system = H + lambda * Matrix66::Identity();
        const V6 d = system.ldlt().solve(-b);
        if (!d.allFinite()) break;
        if (d.squaredNorm() < 1e-24) {
          converged = true;
          break;
        }
        const Transform xi = applyDelta(T_gpu, d);
        gpu_geo::LinearizeResult trial;
        packPose(xi, packed_pose);
        if (!gpu_geo::linearize(pts.data(), static_cast<int>(n_source), packed_pose, voxel_size,
                                1.0 / px_size, huber, false, trial)) {
          std::fprintf(stderr, "CUDA trial failed\n");
          return 1;
        }
        gpu_kernel_ms_total += gpu_geo::lastKernelMs();
        gpu_transfer_ms_total += gpu_geo::lastTransferMs();
        const double rho = (y0 - trial.cost) / d.dot(lambda * d - b);
        if (rho < 0.0) {
          if (isConverged(applyDelta(Transform::Identity(), d), cfg)) {
            converged = true;
            break;
          }
          lambda *= nu;
          nu *= 2.0;
          continue;
        }
        T_gpu = xi;
        y0 = trial.cost;
        lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2 * rho - 1, 3));
        converged = isConverged(applyDelta(Transform::Identity(), d), cfg);
        accepted = true;
        break;
      }
      if (!accepted) break;
      ++anchor_iterations;
    }
    gpu_geo::LinearizeResult final_lin;
    packPose(T_gpu, packed_pose);
    if (!gpu_geo::linearize(pts.data(), static_cast<int>(n_source), packed_pose, voxel_size,
                            1.0 / px_size, huber, true, final_lin)) {
      std::fprintf(stderr, "CUDA final linearization failed\n");
      return 1;
    }
    H_gpu = expandH(final_lin.H);
    cost_gpu = final_lin.cost;
    count_gpu = final_lin.count;
    gpu_kernel_ms_total += gpu_geo::lastKernelMs();
    gpu_transfer_ms_total += gpu_geo::lastTransferMs();
    gpu_stage_ms = nowMs() - t0;
  }

  const V3 dt = T_gpu.translation() - T_cpu.translation();
  const double dr = Eigen::AngleAxisd(T_cpu.linear().transpose() * T_gpu.linear()).angle() *
                    180.0 / 3.141592653589793;
  double h_rel = 0.0;
  const double h_norm = H_cpu.norm();
  if (h_norm > 0) h_rel = (H_gpu - H_cpu).norm() / h_norm;

  std::printf("\n--- correctness at the same map/pose ---\n");
  std::printf("effective points  cpu=%d  gpu=%d\n", count_cpu, count_gpu);
  std::printf("cost              cpu=%.6e  gpu=%.6e  rel=%.2e\n", cost_cpu, cost_gpu,
              cost_cpu != 0 ? std::abs(cost_gpu - cost_cpu) / std::abs(cost_cpu) : 0.0);
  std::printf("H_g relative diff %.3e\n", h_rel);
  std::printf("final pose        dt=%.3e m  dR=%.3e deg\n", dt.norm(), dr);
  std::printf("\n--- timing ---\n");
  std::printf("cpu stage (solve + informationAt) %8.3f ms\n", cpu_stage_ms);
  std::printf("gpu stage (all launches+transfers) %8.3f ms\n", gpu_stage_ms);
  std::printf("  of which kernel    %8.3f ms\n", gpu_kernel_ms_total);
  std::printf("  of which transfers %8.3f ms\n", gpu_transfer_ms_total);
  std::printf("gpu lm iterations %d\n", anchor_iterations);

  gpu_geo::releaseMap();
  return 0;
}
