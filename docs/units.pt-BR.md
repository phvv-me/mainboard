# Units

`CPU`, `GPU` e `NPU` herdam de `Unit`. Eles compartilham o mesmo formato para que uma ferramenta possa tratar todo dispositivo de computação de maneira uniforme, sem forçar a máquina a passar por um modelo exclusivo de CUDA.

```python
from mainboard import Machine

for unit in Machine().units:
    print(unit.kind, unit.vendor, unit.name)
    print(unit.snapshot())
```

Toda unit expõe:

| atributo | significado |
|---|---|
| `kind` | identidade da classe: `UnitKind.CPU`, `UnitKind.GPU` ou `UnitKind.NPU` |
| `vendor` | identidade do provider, como `Vendor.APPLE` ou `Vendor.NVIDIA` |
| `name` | nome legível do modelo |
| `architecture` | string da arquitetura, quando conhecida |
| `snapshot()` | snapshot tipado de telemetria |

## Snapshots

`Unit.snapshot()` retorna `UnitSnapshot`. `GPU.snapshot()` retorna `GPUSnapshot`, que estende `UnitSnapshot` com telemetria específica de GPU.

```python
machine = Machine()
cpu_snapshot = machine.cpu.snapshot("setup")
gpu_snapshot = machine.gpus[0].snapshot("kernel")
```

Os modelos de snapshot são modelos Pydantic e podem ser serializados com `model_dump()`.
