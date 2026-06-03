# Mainboard

**Hardware topology and machine snapshots for Python.**

Mainboard answers a simple question: what is this machine made of, and what can Python safely know about it? It exposes CPUs, GPUs, and NPUs as typed `Unit`s with shared snapshot semantics, and probes the host into one serializable `MachineSnapshot` covering cpu, memory, gpus, npus, environment, and board, without forcing every machine through a CUDA-only model.

## Quickstart

```sh
pip install mainboard
mainboard
```

On Linux machines with NVIDIA GPUs, install the CUDA provider extra:

```sh
pip install "mainboard[nvidia]"
```

Working in a [chefe](https://phvv.me/chefe) project? `chefe add mainboard --pypi`.

## CLI

```sh
mainboard
python -m mainboard
mainboard --color=False
```

Both commands render the same machine schematic. `--color=False` is useful for logs and terminals without color support.

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

## What Mainboard Gives You

| feature | what it means |
|---|---|
| Concept-first units | `CPU`, `GPU`, and `NPU` share `kind`, `vendor`, and `snapshot()` |
| Provider isolation | Apple and NVIDIA details stay behind provider classes |
| Safe imports | Future AMD, Intel, and Qualcomm providers are import-safe stubs |
| Terminal view | `mainboard` renders a Rich schematic of memory and compute units |
| One-call snapshot | `Machine().model_dump_json()` returns cpu, memory, gpus, npus, environment (user, group, scheduler), and board (motherboard, BIOS) |

## Platforms

| platform | status |
|---|---|
| Apple Silicon macOS | CPU, Apple GPU, and Apple Neural Engine detection |
| Linux + NVIDIA CUDA | CPU and NVIDIA GPU detection |
| Other platforms | CPU fallback plus inert future-provider stubs |

!!! warning "Mainboard is early (`0.0.x`)"
    The public API is intentionally small, but provider telemetry details may still change.

Next, read the guide on [units](units.md) and the [probe and snapshot](probe.md), or jump to the [environment](environment.md), [board](board.md), and [providers](providers.md) reference.
