# Mainboard

**CPU, GPU, and NPU hardware topology for Python.**

Mainboard answers a simple question: what compute units does this machine have, and what can Python safely know about them? It exposes CPUs, GPUs, and NPUs as typed `Unit`s with shared snapshot semantics, without forcing every machine through a CUDA-only model.

## Quickstart

```sh
pip install mainboard
mainboard
```

On Linux machines with NVIDIA GPUs, install the CUDA provider extra:

```sh
pip install "mainboard[nvidia]"
```

For persistent CLI use, `uv tool install mainboard` is also a good fit.

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
```

## What Mainboard Gives You

| feature | what it means |
|---|---|
| Concept-first units | `CPU`, `GPU`, and `NPU` share `kind`, `vendor`, and `snapshot()` |
| Provider isolation | Apple and NVIDIA details stay behind provider classes |
| Safe imports | Future AMD, Intel, and Qualcomm providers are import-safe stubs |
| Terminal view | `mainboard` renders a Rich schematic of memory and compute units |
| One-call snapshot | `Machine().model_dump_json()` returns CPU, memory, GPUs, NPUs, and the host environment (user, group, scheduler) |

## Platforms

| platform | status |
|---|---|
| Apple Silicon macOS | CPU, Apple GPU, and Apple Neural Engine detection |
| Linux + NVIDIA CUDA | CPU and NVIDIA GPU detection |
| Other platforms | CPU fallback plus inert future-provider stubs |

!!! warning "Mainboard is early (`0.0.x`)"
    The public API is intentionally small, but provider telemetry details may still change.

Next, see the [API reference](api.md).
