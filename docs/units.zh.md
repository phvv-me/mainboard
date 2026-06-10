# Units

`CPU`、`GPU` 和 `NPU` 都继承自 `Unit`。它们共享相同的结构，因此工具可以统一地对待每个计算设备，而不会强制机器套用仅限 CUDA 的模型。

```python
from mainboard import Machine

for unit in Machine().units:
    print(unit.kind, unit.vendor, unit.name)
    print(unit.snapshot())
```

每个 unit 都暴露：

| 属性 | 含义 |
|---|---|
| `kind` | 类别标识：`UnitKind.CPU`、`UnitKind.GPU` 或 `UnitKind.NPU` |
| `vendor` | provider 标识，例如 `Vendor.APPLE` 或 `Vendor.NVIDIA` |
| `name` | 人类可读的型号名称 |
| `architecture` | 已知时的架构字符串 |
| `memory` | `Memory` 读数：该单元可见内存的总量、已用和空闲字节 |
| `snapshot()` | 带类型的遥测快照 |

每个单元都以相同方式报告内存，因此 `unit.memory.total_bytes` 和 `unit.memory.used_gb` 在 CPU、GPU 和 NPU 上用法一致。

## 快照

`Unit.snapshot()` 返回 `UnitSnapshot`。`GPU.snapshot()` 返回 `GPUSnapshot`，它扩展了 `UnitSnapshot`，加入 GPU 专属的遥测信息。

```python
machine = Machine()
cpu_snapshot = machine.cpu.snapshot("setup")
gpu_snapshot = machine.gpus[0].snapshot("kernel")
```

快照模型是 Pydantic 模型，可以用 `model_dump()` 序列化。
