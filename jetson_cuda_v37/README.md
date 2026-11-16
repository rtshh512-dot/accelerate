# Jetson Orin Nano CUDA 几何配准原型（v37）

这是**第一阶段的独立正确性与耗时实验**：对同一合成地图、点云和初始位姿，比较原项目的 CPU 几何配准与 CUDA 几何配准。CUDA 负责逐点体素查找、高度图采样、残差、雅可比和 H/b 累加；6×6 LM 求解仍在 CPU。它尚未接入完整 COIN-BIEVR 轨迹流程，不能产生 ATE，也不能代表最终 GPU 加速效果。

## 为什么先做这个原型

现有旧原型存在 Morton 编码、位姿内存布局、图像梯度、邻域回退、地图导出及循环括号等错误，直接测速度会误判。v37 修正这些问题，并在计时前比较**同一位姿**上的匹配点数、代价、H、b；不一致就退出。

## 在 Orin Nano 上编译

需要板上已安装 JetPack/CUDA、CMake、Eigen、TBB、Ceres 和 C++ 编译器。先将 [COIN-BIEVR-unofficial](https://github.com/qiyu-lu/COIN-BIEVR-unofficial) 源码克隆并切到 `c560f3b`，再应用本仓库 `baseline_v32/COIN-BIEVR-geometry-probe-v32-full.patch`，使 CPU 参考与 B 组源码一致。`COIN_ROOT` 指向打好补丁的源码根目录。本原型只调用核心库源文件，不要求先启动 ROS。

```bash
git clone https://github.com/qiyu-lu/COIN-BIEVR-unofficial.git coin-bievr-v32
cd coin-bievr-v32
git checkout c560f3b
git apply /path/to/accelerate/baseline_v32/COIN-BIEVR-geometry-probe-v32-full.patch
cd /path/to/accelerate
```

```bash
cmake -S . -B build -DCOIN_ROOT=/absolute/path/to/COIN-BIEVR-unofficial \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/bench_gpu_geo 8000
```

Orin Nano 的 CUDA 架构为 `sm_87`。如果依赖由 ROS 工作区提供，需要先 source 对应的 `setup.bash`，让 CMake 能找到 Ceres 等依赖。不同 JetPack/CUDA 版本可能需要调整编译选项；**这份代码尚未在有 CUDA 的设备上编译**。

## 输出如何理解

- `initial linearization`：同一位姿下的 CPU/GPU 点数及 H、b、代价差；这项不通过，程序停止。
- `map pack`、`map upload`：把 CPU 体素对象转换成连续数组并上传的时间。
- `cpu stage`、`gpu stage`：单次合成场景的几何求解时间；本原型的 GPU stage 没有包含地图打包和上传，比较整帧时**必须把这两项加回去**。
- `final pose`：两套 LM 求解的最终位姿差。由于浮点归约次序和 CPU/GPU 计算细节不同，要求先看同位姿 H/b，再看位姿差；不要只凭最终耗时判断正确性。

当前程序每次只测一遍；要判断稳定速度差，需要在相同功耗模式下重复运行并汇总，而不能把单次输出当成结论。

## 与用户指定的三个实验组的关系

正式整轨实验应有 A=原版 COIN-BIEVR CPU，B=自适应 λ＋弱方向＋v32 几何探测＋直接体素查询复用 CPU，C=与 B 算法相同但使用 CUDA 的版本。**本原型尚不是 C 组**：它只验证几何 CUDA 核函数，未接入强度联合阶段、动态地图同步或完整轨迹。尤其在 v32 的 RunwayD 实验中，大多数帧跳过完整几何迭代；要获得明显整帧收益，后续重点应是联合阶段与地图同步，而非仅优化几何阶段。

## 仍需上板验证

当前 Windows 环境没有 `nvcc`，不能验证 CUDA 编译、GPU 运行或加速比。首次运行请保存完整 CMake 构建日志和程序输出。若同位姿对照失败，先检查点数，再检查体素键、邻域关联、图像采样和雅可比；不要绕过对照直接报告性能。
