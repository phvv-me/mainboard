<div align="center">

[![mainboard banner](https://raw.githubusercontent.com/phvv-me/mainboard/main/docs/assets/banner.png)](https://phvv.me/mainboard)

[![CI](https://github.com/phvv-me/mainboard/actions/workflows/ci.yml/badge.svg)](https://github.com/phvv-me/mainboard/actions/workflows/ci.yml)
[![Publish](https://github.com/phvv-me/mainboard/actions/workflows/publish.yml/badge.svg)](https://github.com/phvv-me/mainboard/actions/workflows/publish.yml)
[![PyPI](https://img.shields.io/pypi/v/mainboard)](https://pypi.org/project/mainboard/)
[![Python](https://img.shields.io/pypi/pyversions/mainboard)](https://pypi.org/project/mainboard/)
[![Docs](https://img.shields.io/badge/docs-phvv.me%2Fmainboard-15803d)](https://phvv.me/mainboard)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/phvv-me/mainboard/actions/workflows/ci.yml)

</div>

## Installation

```sh
pip install mainboard         # CPU and Apple probing, pure Python
pip install mainboard[cuda]   # adds NVIDIA telemetry via the CUDA Python bindings
```

Working in a [chefe](https://phvv.me/chefe) project? Add it to your manifest:

```sh
chefe add mainboard -l python
```

The base install is light and pure Python, so GPU-less Linux hosts like a Raspberry Pi pull nothing CUDA-related. The `cuda` extra installs the NVIDIA bindings on Linux for full GPU detection and telemetry, and provider detection degrades gracefully to no NVIDIA devices whenever the bindings or the hardware are absent.

## What it is

mainboard tells Python what compute is on the current machine, without assuming the world is only CUDA. It models CPUs, GPUs, and NPUs as `Unit`s, keeps vendor-specific probing behind providers (Apple and NVIDIA today), and gives you the whole board in one call.

```python
from mainboard import Machine

print(Machine().model_dump_json(indent=2))   # cpu, memory, gpus, npus, and the host environment
```

## Usage

```python
machine = Machine()
machine.cpu.snapshot()             # CPU identity and capacity
machine.gpus[0].snapshot()         # per-GPU telemetry
machine.environment                # user, group(s), and job scheduler on the host
machine.model_dump_json()          # one-call JSON probe of the whole machine
```

The CLI renders a Rich schematic of the board:

```sh
mainboard
```

## Timing spans

`span` is a lightweight, nestable timer for production code, off by default so a disabled span costs one boolean check. `with span("extract"):` or `@span` times a block or a function (sync or async, nesting tracked per-`asyncio.Task` via a `ContextVar` so concurrent tasks never share a stack), and every closed span folds into a `Collector` — the process-wide default, or one you own for a scoped window such as a single request.

```python
from mainboard.profiling import enable_spans, span

enable_spans()

with span("pipeline"):
    with span("extract"):
        ...
    with span("embed"):
        ...
```

Read the result with `Collector.records()` (the raw per-occurrence log) or `Collector.stats()` (per dotted-path count/total/mean/p50/p95/max), and print either with `.report()` or `.show()`. See [`docs/profiling.md`](docs/profiling.md) for the full profiling API, including the GPU-sampling `Profiler`.
