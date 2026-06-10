# Probe and Snapshot

主要なエントリポイントは `Machine` です。

```python
from mainboard import Machine

machine = Machine()
```

## Machine

| 属性 | 型 | 意味 |
|---|---|---|
| `cpu` | `CPU` | 検出されたホスト CPU |
| `gpus` | `tuple[GPU, ...]` | 検出された GPU |
| `npus` | `tuple[NPU, ...]` | 検出された NPU |
| `units` | `tuple[Unit, ...]` | CPU、GPU、NPU を 1 つのシーケンスにまとめたもの |
| `environment` | `Environment` | ホスト上のユーザー、グループ、ジョブスケジューラ |
| `board` | `Board` | ホストのマザーボードとファームウェアの識別情報 |
| `snapshot()` | `MachineSnapshot` | ホスト全体をワンコールで探査 |
| `model_dump_json()` | `str` | スナップショットを JSON にシリアライズ |

## Snapshot

`Machine.snapshot()` はホストをワンコールで探査し、`MachineSnapshot` を返します。これは CPU の識別情報、システムメモリ、すべての GPU と NPU、ホストの実行環境、そしてシステムボードを集約した、凍結された Pydantic モデルです。`Machine.model_dump_json()` はそれをシリアライズする便利メソッドです。

```python
from mainboard import Machine

print(Machine().model_dump_json(indent=2))
```

```json
{
  "timestamp_ns": 625513707924083,
  "hostname": "host.local",
  "cpu": {
    "name": "Apple M4 Pro",
    "architecture": "arm64",
    "vendor": "apple",
    "logical_cores": 14,
    "physical_cores": 14,
    "total_memory_bytes": 51539607552,
    "current_clock_mhz": 4.0
  },
  "memory": {
    "total_bytes": 51539607552,
    "used_bytes": 25495814144,
    "free_bytes": 20745732096,
    "scope": "system",
    "unified": false,
    "source": "psutil",
    "supported": true
  },
  "environment": {
    "user": "pedro",
    "group": "staff",
    "groups": ["staff", "admin"],
    "scheduler": "none"
  },
  "board": {
    "vendor": "Apple",
    "model": "Mac16,8",
    "version": "MacBook Pro",
    "bios_vendor": "",
    "bios_version": ""
  },
  "gpus": [],
  "npus": []
}
```

| フィールド | 型 | 意味 |
|---|---|---|
| `hostname` | `str` | 探査したホストのネットワーク名 |
| `cpu` | `CpuSnapshot` | CPU 名、アーキテクチャ、ベンダー、コア数、メモリ |
| `memory` | `Memory` | 探査時点のシステム RAM 使用状況 |
| `environment` | `Environment` | ホスト上で利用可能なユーザー、グループ、ジョブスケジューラ |
| `board` | `Board` | マザーボードとファームウェアの識別情報 |
| `gpus` | `tuple[GPUSnapshot, ...]` | GPU ごとのテレメトリ。GPU のないホストでは空 |
| `npus` | `tuple[UnitSnapshot, ...]` | NPU ごとのテレメトリ。NPU のないホストでは空 |
| `unit_count` | `int` | CPU とすべての GPU、NPU の合計 |
| `kinds` | `tuple[UnitKind, ...]` | 存在する個別のユニット種別 |

探査はベストエフォートです。アクセラレータのないホストは例外を投げる代わりに空の `gpus` と `npus` を返します。モデルは `MachineSnapshot.model_validate_json(...)` でラウンドトリップするため、下流のツールはその JSON をネットワーク越しに送り、スナップショットを再構築できます。
