# Sonda e instantánea

El punto de entrada principal es `Machine`.

```python
from mainboard import Machine

machine = Machine()
```

## Machine

| atributo | tipo | significado |
|---|---|---|
| `cpu` | `CPU` | CPU del host detectada |
| `gpus` | `tuple[GPU, ...]` | GPU detectadas |
| `npus` | `tuple[NPU, ...]` | NPU detectadas |
| `units` | `tuple[Unit, ...]` | CPU, GPU y NPU en una sola secuencia |
| `environment` | `Environment` | el usuario, grupo(s) y planificador de trabajos en el host |
| `board` | `Board` | la placa madre e identidad de firmware del host |
| `snapshot()` | `MachineSnapshot` | sonda en una sola llamada de todo el host |
| `model_dump_json()` | `str` | instantánea serializada a JSON |

## Instantánea

`Machine.snapshot()` sondea el host en una sola llamada y devuelve una `MachineSnapshot`, un modelo de Pydantic congelado que agrega la identidad de la CPU, la memoria del sistema, cada GPU y NPU, el entorno de ejecución del host y la placa del sistema. `Machine.model_dump_json()` es la conveniencia que la serializa.

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
| `hostname` | `str` | nombre de red del host sondeado |
| `cpu` | `CpuSnapshot` | nombre, arquitectura, proveedor, conteos de núcleos y memoria de la CPU |
| `memory` | `MemoryUsage` | uso de RAM del sistema al momento del sondeo |
| `environment` | `Environment` | usuario, grupo(s) y planificador de trabajos disponible en el host |
| `board` | `Board` | identidad de la placa madre y el firmware |
| `gpus` | `tuple[GPUSnapshot, ...]` | telemetría por GPU, vacía en un host sin GPU |
| `npus` | `tuple[UnitSnapshot, ...]` | telemetría por NPU, vacía en un host sin NPU |
| `unit_count` | `int` | CPU más cada GPU y NPU |
| `kinds` | `tuple[UnitKind, ...]` | tipos de unidad distintos presentes |

El sondeo es de mejor esfuerzo: un host sin acelerador devuelve `gpus` y `npus` vacías en lugar de lanzar un error. El modelo hace round-trip con `MachineSnapshot.model_validate_json(...)`, así que una herramienta posterior puede enviar el JSON por la red y reconstruir la instantánea.
