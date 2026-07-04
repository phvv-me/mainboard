# Units

`CPU`、`GPU`、`NPU` は `Unit` を継承します。これらは同じ形を共有するため、ツールはマシンを CUDA 専用モデルに押し込めることなく、すべての計算デバイスを一様に扱えます。

```python
from mainboard import Machine

for unit in Machine().units:
    print(unit.kind, unit.vendor, unit.label)
    print(unit.snapshot())
```

すべてのユニットは次を公開します。

| 属性 | 意味 |
|---|---|
| `kind` | クラスの識別子: `UnitKind.CPU`、`UnitKind.GPU`、`UnitKind.NPU` |
| `vendor` | プロバイダの識別子（例: `Vendor.APPLE` や `Vendor.NVIDIA`） |
| `name` | 人間が読めるモデル名 |
| `architecture` | 判明している場合のアーキテクチャ文字列 |
| `memory` | `Memory` の読み取り値: ユニットから見えるメモリの total / used / free バイト |
| `snapshot()` | 型付きのテレメトリスナップショット |

すべてのユニットが同じ方法でメモリを報告するため、`unit.memory.total_bytes` と `unit.memory.used_gb` は CPU・GPU・NPU で一様に使えます。

## スナップショット

`Unit.snapshot()` は `UnitSnapshot` を返します。`GPU.snapshot()` は `GPUSnapshot` を返し、これは `UnitSnapshot` を GPU 固有のテレメトリで拡張したものです。

```python
machine = Machine()
cpu_snapshot = machine.cpu.snapshot("setup")
gpu_snapshot = machine.gpus[0].snapshot("kernel")
```

スナップショットモデルは Pydantic モデルであり、`model_dump()` でシリアライズできます。
