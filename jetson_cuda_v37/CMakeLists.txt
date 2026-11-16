cmake_minimum_required(VERSION 3.18)
project(gpu_geo_proto LANGUAGES CXX CUDA)

# Point COIN_ROOT at the COIN-BIEVR working tree (the directory that contains
# BIEVR/include and BIEVR/src).
set(COIN_ROOT "" CACHE PATH "COIN-BIEVR repository root")
if(NOT EXISTS "${COIN_ROOT}/BIEVR/src/ls_optimizer.cpp")
  message(FATAL_ERROR "Pass -DCOIN_ROOT=/absolute/path/to/COIN-BIEVR source")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin Nano
endif()

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(Ceres REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CPU reference: the real optimizer, exactly as the ROS package compiles it.
add_library(bievr_cpu STATIC
  "${COIN_ROOT}/BIEVR/src/ls_optimizer.cpp"
  "${COIN_ROOT}/BIEVR/src/bievr_map.cpp"
  "${COIN_ROOT}/BIEVR/src/common.cpp"
  "${COIN_ROOT}/BIEVR/src/utils.cpp"
)
target_include_directories(bievr_cpu PUBLIC "${COIN_ROOT}/BIEVR/include")
target_link_libraries(bievr_cpu PUBLIC Eigen3::Eigen TBB::tbb Ceres::ceres)
target_compile_definitions(bievr_cpu PUBLIC MODE_LPP)

add_library(gpu_geo STATIC src/gpu_linearize.cu)
target_include_directories(gpu_geo PUBLIC src)
target_link_libraries(gpu_geo PUBLIC CUDA::cudart)

add_executable(bench_gpu_geo src/bench_gpu_geo.cpp)
target_link_libraries(bench_gpu_geo PRIVATE gpu_geo bievr_cpu)
target_include_directories(bench_gpu_geo PRIVATE src)
