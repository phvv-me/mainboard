<div align="center" markdown>
![mainboard banner](assets/banner.png){ width="760" }
</div>

# mainboard

**Python 向けのハードウェアトポロジーとマシンスナップショット。**

mainboard はシンプルな問いに答えます。このマシンは何でできていて、Python から安全に何を知ることができるのか、です。CPU、GPU、NPU を共通のスナップショットセマンティクスを持つ型付き `Unit` として公開し、すべてのマシンを CUDA 専用モデルに押し込めることなく、ホストを cpu、memory、gpus、npus、environment、board、toolchain をカバーする 1 つのシリアライズ可能な `MachineSnapshot` へと探査します。

## クイックスタート

```sh
pip install mainboard
mainboard
```

Linux では CUDA バインディングが自動的にインストールされるため、NVIDIA GPU はそのまま検出されます。macOS では CUDA 関連は一切取り込まれず、NVIDIA GPU がないホストでも検出は適切にフォールバックします。

[chefe](https://phvv.me/chefe) プロジェクトで作業していますか？ `chefe add mainboard -l python`。

## CLI

```sh
mainboard
python -m mainboard
mainboard --color=False
```

どちらのコマンドも同じマシンの模式図を描画します。`--color=False` は、ログや色をサポートしないターミナルで役立ちます。

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

## mainboard が提供するもの

| 機能 | 意味 |
|---|---|
| コンセプトファーストなユニット | `CPU`、`GPU`、`NPU` が `kind`、`vendor`、`snapshot()` を共有します |
| プロバイダの分離 | Apple や NVIDIA の詳細はプロバイダクラスの背後に留まります |
| 安全なインポート | 将来の AMD、Intel、Qualcomm プロバイダはインポート安全なスタブです |
| ターミナルビュー | `mainboard` がメモリと計算ユニットの Rich な模式図を描画します |
| ツールチェーン検出 | 拡張可能な探査レジストリが、PATH 上の C/C++/CUDA コンパイラとビルドシステムをバージョン付きで報告します |
| ワンコールスナップショット | `Machine().model_dump_json()` が cpu、memory、gpus、npus、environment（user、group、scheduler）、board（マザーボード、BIOS）、toolchain（コンパイラ、ビルドシステム）を返します |

## プラットフォーム

| プラットフォーム | ステータス |
|---|---|
| Apple Silicon macOS | CPU、Apple GPU、Apple Neural Engine の検出 |
| Linux + NVIDIA CUDA | CPU と NVIDIA GPU の検出 |
| その他のプラットフォーム | CPU のフォールバックと、不活性な将来プロバイダのスタブ |

!!! warning "mainboard はまだ初期段階です（`0.0.x`）"
    公開 API は意図的に小さく保たれていますが、プロバイダのテレメトリの詳細は今後変更される可能性があります。

次に、[units](units.md) と [probe and snapshot](probe.md) のガイドを読むか、[environment](environment.md)、[toolchain](toolchain.md)、[board](board.md)、[providers](providers.md) のリファレンスに進んでください。
