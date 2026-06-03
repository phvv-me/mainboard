# 探测与快照

主入口是 `Machine`。

```python
from mainboard import Machine

machine = Machine()
```

## Machine

| 属性 | 类型 | 含义 |
|---|---|---|
| `cpu` | `CPU` | 检测到的主机 CPU |
| `gpus` | `tuple[GPU, ...]` | 检测到的 GPU |
| `npus` | `tuple[NPU, ...]` | 检测到的 NPU |
| `units` | `tuple[Unit, ...]` | 将 CPU、GPU 和 NPU 合并为一个序列 |
| `environment` | `Environment` | 主机上的用户、组和作业调度器 |
| `board` | `Board` | 主机的主板与固件标识 |
| `snapshot()` | `MachineSnapshot` | 一次调用即可探测整台主机 |
| `model_dump_json()` | `str` | 序列化为 JSON 的快照 |

## 快照

`Machine.snapshot()` 一次调用即探测主机，并返回一个 `MachineSnapshot`——一个冻结的 Pydantic 模型，聚合了 CPU 标识、系统内存、每一个 GPU 和 NPU、主机的执行环境以及系统主板。`Machine.model_dump_json()` 是用于序列化它的便捷方法。

```python
from mainboard import Machine

print(Machine().model_dump_json(indent=2))
```

```json
{
  "timestamp_ns": 625513707924083,
  "hostname": "host.local",
  "cpu": {
    "name": "Apple M4 Pro",
    "architecture": "arm64",
    "vendor": "apple",
    "logical_cores": 14,
    "physical_cores": 14,
    "total_memory_bytes": 51539607552,
    "current_clock_mhz": 4.0
  },
  "memory": {
    "scope": "system",
    "total_bytes": 51539607552,
    "used_bytes": 25495814144,
    "free_bytes": 20745732096,
    "source": "psutil"
  },
  "environment": {
    "user": "pedro",
    "group": "staff",
    "groups": ["staff", "admin"],
    "scheduler": "none"
  },
  "board": {
    "vendor": "Apple",
    "model": "Mac16,8",
    "version": "MacBook Pro",
    "bios_vendor": "",
    "bios_version": ""
  },
  "gpus": [],
  "npus": []
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `hostname` | `str` | 被探测主机的网络名称 |
| `cpu` | `CpuSnapshot` | CPU 名称、架构、vendor、核心数、内存 |
| `memory` | `MemoryUsage` | 探测时的系统 RAM 使用情况 |
| `environment` | `Environment` | 主机上可用的用户、组和作业调度器 |
| `board` | `Board` | 主板与固件标识 |
| `gpus` | `tuple[GPUSnapshot, ...]` | 每个 GPU 的遥测信息，在无 GPU 的主机上为空 |
| `npus` | `tuple[UnitSnapshot, ...]` | 每个 NPU 的遥测信息，在无 NPU 的主机上为空 |
| `unit_count` | `int` | CPU 加上每一个 GPU 和 NPU |
| `kinds` | `tuple[UnitKind, ...]` | 存在的不同 unit 类别 |

探测采取尽力而为的策略：没有加速器的主机会返回空的 `gpus` 和 `npus`，而不会抛出异常。该模型可通过 `MachineSnapshot.model_validate_json(...)` 往返还原，因此下游工具可以将 JSON 通过网络传输并重建快照。
