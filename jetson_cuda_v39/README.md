# v39：第二阶段几何＋强度联合配准的 CUDA 后端

v39 基于 v38。第一阶段完整几何迭代仍可使用 CUDA；新增独立开关 `intensity.joint_cuda_enabled`，让第二阶段 Joint / joint_weak_photo 的逐点线性化尝试使用 GPU。CPU 保留单遍几何探测、λ 计算、弱方向选点、6×6 LM 求解和地图更新。

## 计算路径

CPU 先计算并固定每个点的弱方向质量权重。GPU 在每个 LM 试探位姿下查询体素、高度图及强度图，分别累加几何和强度残差的代价、H 与 b。强度残差仍按 `sqrt(quality) * photo_scale` 缩放。GPU 返回总 H/b、几何与强度信息块以及匹配点数，CPU 继续求解位姿增量。

`joint_cuda_enabled` 与 `geometry_cuda_enabled` 独立。v39 只覆盖当前主要实验使用的 Joint 模式、有效高度图残差及雅可比、无入射角残差、无 photo_confidence、无冻结几何雅可比、`lm_trial_stride=1`。其他配置自动使用 CPU。第二阶段 CPU 的直接体素指针复用不用于 GPU；GPU 用 Morton 排序表查询。地图每次求解都重新打包上传，尚无 GPU 常驻地图。

默认 `joint_cuda_validate: True`。同一初始位姿分别计算 CPU/GPU 的几何点数、强度点数、总 H/b/代价及分开信息块。差异超出绝对 `1e-6` 加相对 `1e-4` 的容差时，当前求解回退 CPU。CUDA 调用失败也回退 CPU。校验组包含额外 CPU 线性化开销；测纯性能时另跑 `CUDA_VALIDATE=false`。

## 实验组

`RUN_JETSON_CUDA_V39.sh` 生成四组：原版 CPU、自适应 v32 CPU、仅第一阶段几何 CUDA、两阶段 CUDA。比较 GPU 的净影响时，重点看 `geometry_cuda` 与 `both_cuda`；比较整个改造系统与原版则看 `original` 与 `both_cuda`。各组需在同一块 Orin Nano、相同功耗模式和数据集上运行。

日志增加 `cuda_joint`、`cuda_joint_check`、`cuda_joint_valid` 和第二阶段打包/上传/核函数/回传/校验耗时。`cuda_joint=0` 可能意味着没有进入第二阶段、配置不满足 CUDA 条件、校验失败或 CUDA 运行失败，需要结合日志判断。

## 当前验证边界

已检查完整补丁能应用于上游 `c560f3b` 的干净源码，差异无空白错误。Windows 主机没有 nvcc，VM 没有 NVIDIA GPU；**尚未编译 CUDA、运行板上轨迹或验证 ATE 与耗时**。首次板上运行先保持校验开启，检查匹配点数、H/b 和回退警告，再关闭校验测速度。
