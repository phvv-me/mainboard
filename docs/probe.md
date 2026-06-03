# Probe and Snapshot

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
| `board` | `Board` | the host's motherboard and firmware identity |
| `snapshot()` | `MachineSnapshot` | one-call probe of the whole host |
| `model_dump_json()` | `str` | snapshot serialized to JSON |

## Snapshot

`Machine.snapshot()` probes the host in one call and returns a `MachineSnapshot`, a frozen Pydantic model that aggregates CPU identity, system memory, every GPU and NPU, the host's execution environment, and the system board. `Machine.model_dump_json()` is the convenience that serializes it.

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

| field | type | meaning |
|---|---|---|
| `hostname` | `str` | network name of the probed host |
| `cpu` | `CpuSnapshot` | CPU name, architecture, vendor, core counts, memory |
| `memory` | `MemoryUsage` | system RAM usage at probe time |
| `environment` | `Environment` | user, group(s), and job scheduler available on the host |
| `board` | `Board` | motherboard and firmware identity |
| `gpus` | `tuple[GPUSnapshot, ...]` | per-GPU telemetry, empty on a host with no GPU |
| `npus` | `tuple[UnitSnapshot, ...]` | per-NPU telemetry, empty on a host with no NPU |
| `unit_count` | `int` | CPU plus every GPU and NPU |
| `kinds` | `tuple[UnitKind, ...]` | distinct unit kinds present |

Probing is best-effort: a host with no accelerator returns empty `gpus` and `npus` rather than raising. The model round-trips with `MachineSnapshot.model_validate_json(...)`, so a downstream tool can ship the JSON across the wire and rebuild the snapshot.
