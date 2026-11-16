# COIN-BIEVR acceleration experiments

Jetson Orin Nano 上的 CUDA 几何配准原型与 CPU 对照材料：

- [v37 CUDA 原型、编译步骤与限制](jetson_cuda_v37/README.md)
- [v32 CPU 基线补丁与运行脚本](jetson_cuda_v37/baseline_v32/)
- [三组实验口径](docs/JETSON_CUDA_V37_EXPERIMENT.md)
- [CUDA 加速方向报告](docs/CUDA_GPU_ACCELERATION_DIRECTION_REPORT.md)

当前 CUDA 原型尚未接入完整轨迹流程，也未在 Jetson 上编译或测得加速比；请先完成同位姿 CPU/GPU 线性化对照，再进行整轨集成。
