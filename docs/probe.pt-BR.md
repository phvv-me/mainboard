# Probe e snapshot

O ponto de entrada principal é `Machine`.

```python
from mainboard import Machine

machine = Machine()
```

## Machine

| atributo | tipo | significado |
|---|---|---|
| `cpu` | `CPU` | CPU do host detectada |
| `gpus` | `tuple[GPU, ...]` | GPUs detectadas |
| `npus` | `tuple[NPU, ...]` | NPUs detectadas |
| `units` | `tuple[Unit, ...]` | CPU, GPUs e NPUs em uma única sequência |
| `environment` | `Environment` | o user, o(s) group(s) e o job scheduler do host |
| `board` | `Board` | a identidade da placa-mãe e do firmware do host |
| `snapshot()` | `MachineSnapshot` | probe de todo o host em uma única chamada |
| `model_dump_json()` | `str` | snapshot serializado em JSON |

## Snapshot

`Machine.snapshot()` sonda o host em uma única chamada e retorna um `MachineSnapshot`, um modelo Pydantic congelado que agrega a identidade da CPU, a memória do sistema, cada GPU e NPU, o environment de execução do host e a board do sistema. `Machine.model_dump_json()` é o atalho que o serializa.

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

| campo | tipo | significado |
|---|---|---|
| `hostname` | `str` | nome de rede do host sondado |
| `cpu` | `CpuSnapshot` | nome, arquitetura, vendor, contagem de núcleos e memória da CPU |
| `memory` | `MemoryUsage` | uso da RAM do sistema no momento do probe |
| `environment` | `Environment` | user, group(s) e job scheduler disponíveis no host |
| `board` | `Board` | identidade da placa-mãe e do firmware |
| `gpus` | `tuple[GPUSnapshot, ...]` | telemetria por GPU, vazia em um host sem GPU |
| `npus` | `tuple[UnitSnapshot, ...]` | telemetria por NPU, vazia em um host sem NPU |
| `unit_count` | `int` | a CPU mais cada GPU e NPU |
| `kinds` | `tuple[UnitKind, ...]` | tipos distintos de unit presentes |

O probe é best-effort: um host sem acelerador retorna `gpus` e `npus` vazias em vez de levantar exceção. O modelo faz round-trip com `MachineSnapshot.model_validate_json(...)`, de modo que uma ferramenta downstream pode enviar o JSON pela rede e reconstruir o snapshot.
