# Units

`CPU`, `GPU`, and `NPU` inherit from `Unit`. They share the same shape so a tool can treat every compute device uniformly, without forcing the machine through a CUDA-only model.

```python
from mainboard import Machine

for unit in Machine().units:
    print(unit.kind, unit.vendor, unit.name)
    print(unit.snapshot())
```

Every unit exposes:

| attribute | meaning |
|---|---|
| `kind` | class identity: `UnitKind.CPU`, `UnitKind.GPU`, or `UnitKind.NPU` |
| `vendor` | provider identity such as `Vendor.APPLE` or `Vendor.NVIDIA` |
| `name` | human-readable model name |
| `architecture` | architecture string when known |
| `snapshot()` | typed telemetry snapshot |

## Snapshots

`Unit.snapshot()` returns `UnitSnapshot`. `GPU.snapshot()` returns `GPUSnapshot`, which extends `UnitSnapshot` with GPU-specific telemetry.

```python
machine = Machine()
cpu_snapshot = machine.cpu.snapshot("setup")
gpu_snapshot = machine.gpus[0].snapshot("kernel")
```

Snapshot models are Pydantic models and can be serialized with `model_dump()`.
