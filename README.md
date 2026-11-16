# COIN-BIEVR acceleration experiments

Jetson Orin Nano 上的 CUDA 几何配准原型与 CPU 对照材料：

- [v38：接入 COIN-BIEVR 配准流程的可选 CUDA 几何后端](jetson_cuda_v38/JETSON_CUDA_V38.md)
- [v38 完整补丁与 A/B/C 实验脚本](jetson_cuda_v38/)
- [v37 CUDA 原型、编译步骤与限制](jetson_cuda_v37/README.md)
- [v32 CPU 基线补丁与运行脚本](jetson_cuda_v37/baseline_v32/)
- [三组实验口径](docs/JETSON_CUDA_V37_EXPERIMENT.md)
- [CUDA 加速方向报告](docs/CUDA_GPU_ACCELERATION_DIRECTION_REPORT.md)

v38 已将 v37 的几何 CUDA 计算作为可选后端接入第一阶段；第二阶段联合优化仍使用 CPU。尚未在 Jetson 上编译或测得加速比；板上首先保留同位姿 CPU/GPU 校验，再比较整轨 ATE 与端到端耗时。
