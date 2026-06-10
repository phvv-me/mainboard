<div align="center" markdown>
![mainboard banner](assets/banner.png){ width="760" }
</div>

# mainboard

**Topologia de hardware e snapshots de máquina para Python.**

O mainboard responde a uma pergunta simples: do que esta máquina é feita e o que o Python pode saber sobre ela com segurança? Ele expõe CPUs, GPUs e NPUs como `Unit`s tipados com semântica de snapshot compartilhada, e sonda o host gerando um único `MachineSnapshot` serializável que cobre cpu, memória, gpus, npus, environment, board e toolchain, sem forçar toda máquina a passar por um modelo exclusivo de CUDA.

## Início rápido

```sh
pip install mainboard
mainboard
```

No Linux, as bindings do CUDA são instaladas automaticamente, então GPUs NVIDIA são detectadas sem configuração extra. No macOS nada relacionado ao CUDA é instalado, e a detecção degrada normalmente em hosts sem GPU NVIDIA.

Trabalhando em um projeto [chefe](https://phvv.me/chefe)? `chefe add mainboard -l python`.

## CLI

```sh
mainboard
python -m mainboard
mainboard --color=False
```

Ambos os comandos renderizam o mesmo esquema da máquina. `--color=False` é útil para logs e terminais sem suporte a cores.

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

## O que o mainboard oferece

| recurso | o que significa |
|---|---|
| Units centradas no conceito | `CPU`, `GPU` e `NPU` compartilham `kind`, `vendor` e `snapshot()` |
| Isolamento de provider | Detalhes de Apple e NVIDIA ficam por trás de classes de provider |
| Imports seguros | Os futuros providers AMD, Intel e Qualcomm são stubs seguros para importação |
| Visão no terminal | `mainboard` renderiza um esquema Rich da memória e das units de computação |
| Descoberta de toolchain | Um registro de sondas expansível reporta os compiladores C/C++/CUDA e os build systems no PATH, com versões |
| Snapshot em uma chamada | `Machine().model_dump_json()` retorna cpu, memória, gpus, npus, environment (user, group, scheduler), board (placa-mãe, BIOS) e toolchain (compiladores, build systems) |

## Plataformas

| plataforma | status |
|---|---|
| macOS com Apple Silicon | Detecção de CPU, GPU Apple e Apple Neural Engine |
| Linux + NVIDIA CUDA | Detecção de CPU e GPU NVIDIA |
| Outras plataformas | Fallback de CPU mais stubs inertes de futuros providers |

!!! warning "O mainboard é recente (`0.0.x`)"
    A API pública é propositalmente pequena, mas detalhes de telemetria dos providers ainda podem mudar.

A seguir, leia o guia sobre [units](units.md) e o [probe e snapshot](probe.md), ou pule direto para a referência de [environment](environment.md), [toolchain](toolchain.md), [board](board.md) e [providers](providers.md).
