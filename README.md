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
pip install mainboard
```

Working in a [chefe](https://phvv.me/chefe) project? Add it to your manifest:

```sh
chefe add mainboard -l python
```

On Linux the CUDA bindings install automatically, so NVIDIA GPUs are detected out of the box. On macOS nothing CUDA-related is pulled in, and provider detection degrades gracefully when no NVIDIA GPU is present.

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
