# Unidades

`CPU`, `GPU` y `NPU` heredan de `Unit`. Comparten la misma forma para que una herramienta pueda tratar cada dispositivo de cómputo de manera uniforme, sin obligar a la máquina a pasar por un modelo exclusivo de CUDA.

```python
from mainboard import Machine

for unit in Machine().units:
    print(unit.kind, unit.vendor, unit.name)
    print(unit.snapshot())
```

Cada unidad expone:

| atributo | significado |
|---|---|
| `kind` | identidad de clase: `UnitKind.CPU`, `UnitKind.GPU` o `UnitKind.NPU` |
| `vendor` | identidad del proveedor, como `Vendor.APPLE` o `Vendor.NVIDIA` |
| `name` | nombre de modelo legible por humanos |
| `architecture` | cadena de arquitectura cuando se conoce |
| `snapshot()` | instantánea de telemetría tipada |

## Instantáneas

`Unit.snapshot()` devuelve `UnitSnapshot`. `GPU.snapshot()` devuelve `GPUSnapshot`, que extiende `UnitSnapshot` con telemetría específica de GPU.

```python
machine = Machine()
cpu_snapshot = machine.cpu.snapshot("setup")
gpu_snapshot = machine.gpus[0].snapshot("kernel")
```

Los modelos de instantánea son modelos de Pydantic y se pueden serializar con `model_dump()`.
