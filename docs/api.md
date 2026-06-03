# API

The main entry point is `Machine`.

```python
from mainboard import Machine

machine = Machine()
```

## Machine

| attribute | type | meaning |
|---|---|---|
| `cpu` | `CPU` | detected host CPU |
| `gpus` | `tuple[GPU, ...]` | detected GPUs |
| `npus` | `tuple[NPU, ...]` | detected NPUs |
| `units` | `tuple[Unit, ...]` | CPU, GPUs, and NPUs in one sequence |
| `environment` | `Environment` | the user, group(s), and job scheduler on the host |
| `snapshot()` | `MachineSnapshot` | one-call probe of the whole host |
| `model_dump_json()` | `str` | snapshot serialized to JSON |

## Probe

`Machine.snapshot()` probes the host in one call and returns a `MachineSnapshot`, a frozen Pydantic model that aggregates CPU identity, system memory, every GPU and NPU, and the host's execution environment. `Machine.model_dump_json()` is the convenience that serializes it.

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
  "gpus": [],
  "npus": []
}
```

| field | type | meaning |
|---|---|---|
| `hostname` | `str` | network name of the probed host |
| `cpu` | `CpuSnapshot` | CPU name, architecture, vendor, core counts, memory |
| `memory` | `MemoryUsage` | system RAM usage at probe time |
| `environment` | `Environment` | user, group(s), and job scheduler available on the host |
| `gpus` | `tuple[GPUSnapshot, ...]` | per-GPU telemetry, empty on a host with no GPU |
| `npus` | `tuple[UnitSnapshot, ...]` | per-NPU telemetry, empty on a host with no NPU |
| `unit_count` | `int` | CPU plus every GPU and NPU |
| `kinds` | `tuple[UnitKind, ...]` | distinct unit kinds present |

Probing is best-effort: a host with no accelerator returns empty `gpus` and `npus` rather than raising. The model round-trips with `MachineSnapshot.model_validate_json(...)`, so a downstream tool can ship the JSON across the wire and rebuild the snapshot.

## Environment

`Machine.environment`, also carried in the snapshot, reports who is running and what job scheduler the host offers, so a tool can route work without re-detecting it.

| field | type | meaning |
|---|---|---|
| `user` | `str` | login name of the current user |
| `group` | `str` | primary group name |
| `groups` | `tuple[str, ...]` | every group the user belongs to |
| `scheduler` | `Scheduler` | job scheduler on PATH: `slurm`, `pbs`, `pueue`, or `none` |

Cluster schedulers take priority over pueue when more than one is on PATH.

## Units

`CPU`, `GPU`, and `NPU` inherit from `Unit`.

```python
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
cpu_snapshot = machine.cpu.snapshot("setup")
gpu_snapshot = machine.gpus[0].snapshot("kernel")
```

Snapshot models are Pydantic models and can be serialized with `model_dump()`.

## Providers

Providers detect vendor-specific hardware and telemetry while keeping the public API concept-first.

| provider | platform | status |
|---|---|---|
| `AppleGPU` | Apple Silicon macOS | GPU model, cores, Metal support, unified memory |
| `AppleNPU` | Apple Silicon macOS | Neural Engine identity and Core ML backend |
| `NvidiaGPU` | Linux + CUDA | CUDA architecture, SM count, memory, clocks where supported |

AMD, Intel, and Qualcomm providers are import-safe stubs today. They return unavailable so imports and CI do not require hardware or vendor SDKs.

Provider details should add telemetry, not rename the public concepts. A GPU is still a `GPU`; CUDA, Metal, ROCm, Level Zero, and Core ML are backend details.
