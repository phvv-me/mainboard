# Providers

Provider 检测特定 vendor 的硬件和遥测信息，同时保持公共 API 概念优先。

| provider | 平台 | 状态 |
|---|---|---|
| `AppleGPU` | Apple Silicon macOS | GPU 型号、核心、Metal 支持、统一内存 |
| `AppleNPU` | Apple Silicon macOS | Neural Engine 标识和 Core ML 后端 |
| `NvidiaGPU` | Linux + CUDA | CUDA 架构、SM 数量、内存，以及在支持时的时钟频率 |

AMD、Intel 和 Qualcomm provider 目前都是导入安全的桩。它们返回不可用状态，因此导入和 CI 都不需要硬件或 vendor SDK。

Provider 细节应当增加遥测信息，而不是重命名公共概念。GPU 仍然是 `GPU`；CUDA、Metal、ROCm、Level Zero 和 Core ML 都是后端细节。
