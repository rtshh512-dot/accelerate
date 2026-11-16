# Jetson Orin Nano CUDA 实验口径（v37）

日期：2026-11-16。此文件定义对照组和板上准备步骤；尚无 CUDA 实现或板上测量结果。

## 对照组

| 组 | 算法配置 | 执行设备 | 用途 |
|---|---|---|---|
| A | 原版 COIN-BIEVR | CPU | 原始系统参照 |
| B | 自适应 λ＋弱方向＋v32 几何探测＋直接体素查询复用 | CPU | **CUDA 改造的直接基线** |
| C | 与 B 完全相同的算法和参数，仅逐点配准后端换成 CUDA | CPU＋GPU | 测设备加速的净收益 |

原版与 B 的差异包含算法改进，不能把 A→C 的全部时间变化称为 GPU 收益。CUDA 加速比用 B/C 比较；A 只显示整个系统相对原版的最终结果。三组都必须在同一块 Jetson、同一 JetPack、同一 ROS、同一数据集及功耗模式下运行。先以 RunwayD 和 Shield5 检查速度与精度，再扩到 IntersectionD 和 Flat Surfaces。

## 首次上板

1. 在 Jetson 上运行 `bash CHECK_JETSON_CUDA_V37.sh | tee jetson_precheck.log`，保存输出。脚本只读，不更改系统。Jetson 上没有 `nvidia-smi` 不等于没有 CUDA；以 CUDA `deviceQuery` 或最小核函数成功为准。
2. 确认系统版本。JetPack 6.2.1 的根文件系统为 Ubuntu 22.04，适合先评估 ROS 2 Humble；本仓库 README 已测试的 ROS 2 版本为 Jazzy，不能直接假设 Humble 零改动可编译。如果板上已有可用 ROS 环境，先以实际环境为准。
3. 在 Jetson 本机编译并跑通 B 组的 CPU 版；每组至少重复三次，记录轨迹完整性、ATE、RPE、`05_registration`、整帧平均/P95、各阶段迭代次数、功耗模式、GPU/CPU 温度及显存占用。
4. A 组使用未启用自适应、弱方向、几何探测及查询复用的原始实现与原始配置。B 组使用 v32 的 `probe_lookup` 配置。不要把 v32 脚本里的 `baseline` 模式当作 A；该模式仍保留自适应 λ 和弱方向，仅关闭 v32 两项加速。

## CUDA 组的验收顺序

1. 几何阶段单次线性化：对齐关联点数、残差损失、H、b。GPU 浮点归约顺序可能与 CPU 不同，因此检查相对误差并关注近退化特征值。
2. 完整几何阶段：对齐每帧起点、迭代次数、LM 接受/拒绝次数及几何位姿。
3. 联合阶段：保留同一 λ、同一弱方向点和同一地图快照，检查强度 H/b、最终位姿。
4. 整轨：在精度满足要求后比较耗时，分开统计地图上传、GPU 查询、核函数、归约和同步。若仅核函数快而整帧不快，说明 GPU 后端尚未达成目标。

## 当前尚缺

- Jetson 的实际连接方式、JetPack/ROS 版本和板上 CPU 日志。
- 可在 Jetson 上编译、运行及验证的 CUDA 后端。Windows 工作区没有 `nvcc`，现有 VMware 客户机也没有可见 NVIDIA 设备；本机不能证明 CUDA 编译通过，更不能报告加速比。
- A/B 两组在 Jetson 上同机复测的数据。此前 VM 或其他主机的毫秒数不作为 Jetson 基线。

## 参考位置

- v32 对照脚本：`脚本/RUN_GEOMETRY_PROBE_V32.sh`。其中 `probe_lookup` 是 B 组算法参考，`baseline` 不是原版 A 组。
- v32 补丁：`脚本/COIN-BIEVR-geometry-probe-v32-full.patch`。
- CUDA 改造设计：`CUDA_GPU_ACCELERATION_DIRECTION_REPORT.md`。
