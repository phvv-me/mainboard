<div align="center" markdown>
![mainboard banner](assets/banner.png){ width="760" }
</div>

# mainboard

**Topología de hardware e instantáneas de máquinas para Python.**

mainboard responde una pregunta sencilla: ¿de qué está hecha esta máquina y qué puede saber Python de ella de forma segura? Expone CPU, GPU y NPU como `Unit`s tipados con semántica de instantánea compartida, y sondea el host hacia una única `MachineSnapshot` serializable que cubre cpu, memoria, gpus, npus, entorno, placa y toolchain, sin obligar a cada máquina a pasar por un modelo exclusivo de CUDA.

## Inicio rápido

```sh
pip install mainboard
mainboard
```

En Linux, los bindings de CUDA se instalan automáticamente, por lo que las GPU NVIDIA se detectan sin configuración adicional. En macOS no se instala nada relacionado con CUDA, y la detección degrada con normalidad en hosts sin GPU NVIDIA.

¿Trabajas en un proyecto [chefe](https://phvv.me/chefe)? `chefe add mainboard -l python`.

## CLI

```sh
mainboard
python -m mainboard
mainboard --color=False
```

Ambos comandos renderizan el mismo esquema de la máquina. `--color=False` es útil para registros y terminales sin soporte de color.

## Python

```python
from mainboard import Machine

machine = Machine()
print(machine.cpu.name)
print(machine.gpus)
print(machine.npus)
print(machine.board.model)
print(machine.model_dump_json(indent=2))
```

## Qué te ofrece mainboard

| característica | qué significa |
|---|---|
| Unidades centradas en el concepto | `CPU`, `GPU` y `NPU` comparten `kind`, `vendor` y `snapshot()` |
| Aislamiento de proveedores | Los detalles de Apple y NVIDIA quedan detrás de clases de proveedor |
| Importaciones seguras | Los futuros proveedores AMD, Intel y Qualcomm son stubs seguros para importar |
| Vista de terminal | `mainboard` renderiza un esquema Rich de la memoria y las unidades de cómputo |
| Descubrimiento de toolchain | Un registro de sondeos ampliable informa los compiladores C/C++/CUDA y los sistemas de compilación en el PATH, con sus versiones |
| Instantánea en una llamada | `Machine().model_dump_json()` devuelve cpu, memoria, gpus, npus, entorno (usuario, grupo, planificador), placa (placa madre, BIOS) y toolchain (compiladores, sistemas de compilación) |

## Plataformas

| plataforma | estado |
|---|---|
| macOS con Apple Silicon | Detección de CPU, GPU de Apple y Apple Neural Engine |
| Linux + NVIDIA CUDA | Detección de CPU y GPU NVIDIA |
| Otras plataformas | Respaldo de CPU más stubs inertes de futuros proveedores |

!!! warning "mainboard es temprano (`0.0.x`)"
    La API pública es intencionalmente pequeña, pero los detalles de telemetría de los proveedores aún pueden cambiar.

A continuación, lee la guía sobre [unidades](units.md) y la [sonda e instantánea](probe.md), o salta a la referencia de [entorno](environment.md), [toolchain](toolchain.md), [placa](board.md) y [proveedores](providers.md).
