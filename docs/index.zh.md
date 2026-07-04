<div align="center" markdown>
![mainboard banner](assets/banner.png){ width="760" }
</div>

# mainboard

**面向 Python 的硬件拓扑与机器快照。**

mainboard 回答一个简单的问题：这台机器由什么组成，以及 Python 能安全地了解它的哪些信息？它将 CPU、GPU 和 NPU 暴露为带类型的 `Unit`，共享统一的快照语义，并将主机探测为一个可序列化的 `MachineSnapshot`，涵盖 cpu、内存、gpus、npus、environment、board 和 toolchain，而不会强制每台机器都套用仅限 CUDA 的模型。

## 快速开始

```sh
pip install mainboard
mainboard
```

基础安装轻量且为纯 Python，覆盖 CPU 与 Apple 探测，不会引入任何 CUDA 相关依赖。如需 NVIDIA GPU 检测与遥测，请安装 `cuda` 附加项，即 `pip install mainboard[cuda]`，它会在 Linux 上引入 CUDA Python 绑定。当绑定或硬件缺失时，检测会优雅降级为没有 NVIDIA 设备。

在 [chefe](https://phvv.me/chefe) 项目中工作？`chefe add mainboard -l python`。

## CLI

```sh
mainboard
python -m mainboard
mainboard --color=False
```

两个命令渲染相同的机器示意图。`--color=False` 适用于日志和不支持颜色的终端。

## Python

```python
from mainboard import Machine

machine = Machine()
print(machine.cpu.label)
print(machine.gpus)
print(machine.npus)
print(machine.board.model)
print(machine.model_dump_json(indent=2))
```

## mainboard 为你提供什么

| 特性 | 含义 |
|---|---|
| 概念优先的 unit | `CPU`、`GPU` 和 `NPU` 共享 `kind`、`vendor` 和 `snapshot()` |
| Provider 隔离 | Apple 和 NVIDIA 的细节都隐藏在 provider 类之后 |
| 安全导入 | 未来的 AMD、Intel 和 Qualcomm provider 都是导入安全的桩 |
| 终端视图 | `mainboard` 渲染一份内存与计算 unit 的 Rich 示意图 |
| 工具链发现 | 可扩展的探测注册表报告 PATH 上的 C/C++/CUDA 编译器和构建系统及其版本 |
| 一次调用快照 | `Machine().model_dump_json()` 返回 cpu、内存、gpus、npus、environment（用户、组、调度器）、board（主板、BIOS）和 toolchain（编译器、构建系统） |

## 平台

| 平台 | 状态 |
|---|---|
| Apple Silicon macOS | CPU、Apple GPU 和 Apple Neural Engine 检测 |
| Linux + NVIDIA CUDA | CPU 和 NVIDIA GPU 检测 |
| 其他平台 | CPU 回退，外加惰性的未来 provider 桩 |

!!! warning "mainboard 尚处早期（`0.0.x`）"
    公共 API 有意保持精简，但 provider 的遥测细节仍可能变化。

接下来，请阅读 [units](units.md) 指南和[探测与快照](probe.md)，或直接跳转到 [environment](environment.md)、[toolchain](toolchain.md)、[board](board.md) 和 [providers](providers.md) 参考。
