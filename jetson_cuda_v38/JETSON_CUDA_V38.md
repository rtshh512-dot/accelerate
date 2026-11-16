# v38：将 v37 几何 CUDA 原型接入 COIN-BIEVR

日期：2026-11-16。v38 从 v32（自适应 λ、弱方向、几何探测、直接体素查询复用）派生，新增**可选的第一阶段几何 CUDA LM 求解**。尚未在 CUDA 设备上编译和运行，不能声称已加速或保持 ATE。

## 精确改动范围

- `BIEVR_WITH_CUDA=OFF` 是默认构建；不开 CUDA 时保留 v32 的 CPU 路径。
- `intensity.geometry_cuda_enabled: True` 且构建时 `-DBIEVR_WITH_CUDA=ON`，完整的几何 LM 迭代尝试使用 CUDA。GPU 做每点的体素关联、高度图插值、几何残差、雅可比、Huber 和 H/b 归约；CPU 做 6×6 LM 求解与位姿更新。
- v32 的**单遍几何探测**、最终 `informationAt`（也负责填充第二阶段的体素查询缓存）、λ 计算、弱方向选点、第二阶段几何＋强度联合优化、地图更新均仍在 CPU。
- 为确保地图快照一致，流水线重叠模式下自动关闭此 CUDA 分支。`lm_trial_stride>1`、禁用高度图残差/雅可比或启用冻结几何雅可比时也回到 CPU。
- 每次 CUDA 求解前把所有已观测体素的变换、质心、高度图和有效掩码打包上传；本版以正确性为先，没有跨帧常驻或增量地图同步。数据准备及上传的耗时必须算进 `geo_ms` 和端到端时间。

## 正确性保护

默认 `intensity.geometry_cuda_validate: True`：每次尝试 GPU 几何求解前，在**相同初始位姿**上各做一次 CPU/GPU 线性化，比较匹配点数和 H、b、代价；相对误差大于 `1e-4` 或点数不同，就记录警告并对该帧使用 CPU。该验证会增加一遍 CPU 计算，初次上板应保持开启；证明正确后，可设 `False` 单独测纯性能。即使关闭验证，CUDA 调用失败也会回退 CPU。

`CACHE_FRAME` 新增：`cuda_geo`（该帧完整几何求解是否使用 GPU）、`cuda_pack_ms`、`cuda_upload_ms`、`cuda_kernel_ms`、`cuda_transfer_ms`、`cuda_validate_ms`、`cuda_check`、`cuda_valid`。`cuda_check=0` 时没有执行同位姿校验，`cuda_valid=0` 不表示校验失败。`cuda_geo=0` 常见于几何探测直接跳过完整求解，不能据此判定 CUDA 未成功编译。

## 板上实验

完整补丁：`脚本/COIN-BIEVR-jetson-cuda-v38-full.patch`，基于上游 `c560f3b`。脚本：`脚本/RUN_JETSON_CUDA_V38.sh`。脚本有三组：

| 组 | 代码与配置 |
|---|---|
| `original` | 上游 `c560f3b` CPU + 用户提供的原版配置 |
| `adaptive_cpu` | v38 代码 + 自适应/弱方向/v32 两项加速 + CUDA 关闭 |
| `adaptive_cuda` | 与 `adaptive_cpu` 相同，唯一开关差异为 `geometry_cuda_enabled: True` |

脚本需设置 `JETSON_ORIGINAL_CONF`、`JETSON_ADAPTIVE_CONF`，分别指向板上的原版配置和现有自适应弱方向配置。默认 RunwayD 路径、Enwide 杆臂与固定时间偏移来自此前实验。可以用 `RUNWAYD_BAG`、`RUNWAYD_GT`、`COIN_EVALUATOR` 和 `COIN_WS_ROOT` 覆盖。默认 `REPS=3`、几何探测门限 0.02 m/0.5°、`CUDA_VALIDATE=true`。测试性能时再额外跑一组 `CUDA_VALIDATE=false`，与 `adaptive_cpu` 比；先保存验证组的结果。

```bash
export JETSON_ORIGINAL_CONF=/path/to/original.yaml
export JETSON_ADAPTIVE_CONF=/path/to/adaptive_joint_weak_photo.yaml
export UNDERLAY_ROS=/opt/ros/humble/setup.bash  # 依板上实际版本调整
bash RUN_JETSON_CUDA_V38.sh build
bash RUN_JETSON_CUDA_V38.sh run
bash RUN_JETSON_CUDA_V38.sh evaluate
bash RUN_JETSON_CUDA_V38.sh summary
```

v32 在 RunwayD 的既有实验中，完整几何迭代被探测跳过的比例约 97%。所以仅将这一段搬到 GPU，整帧加速上限很低，地图打包/上传还可能让它更慢。v38 的价值首先是验证 CUDA 与现有体素地图、优化器的正确接入和测出真实开销；若正确但无速度收益，下一步应考虑第二阶段联合求解或地图常驻 GPU，而不能把 v38 的负结果解释成“GPU 对整个系统无用”。

## 当前验证边界

已做：补丁应用检查、配置生成和 YAML 解析、静态差异检查。未做：CUDA/C++ 编译、板上运行、CPU/GPU 同位姿对照、ATE 与真实耗时。当前 Windows 主机无 `nvcc`，VM 也无 NVIDIA 设备。
