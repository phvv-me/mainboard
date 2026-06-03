# Board

`Machine.board` は、スナップショットにも含まれており、ホストのマザーボードとファームウェアの識別情報を報告します。これにより、スナップショットはどの物理システムボードと BIOS から取得されたかを記録します。

```python
from mainboard import Machine

board = Machine().board
print(board.vendor, board.model, board.bios_version)
```

| フィールド | 型 | 意味 |
|---|---|---|
| `vendor` | `str` | マザーボードの製造元 |
| `model` | `str` | マザーボードの製品名 |
| `version` | `str` | マザーボードのリビジョンまたはモデル識別子 |
| `bios_vendor` | `str` | ファームウェアのベンダー |
| `bios_version` | `str` | ファームウェアのバージョン文字列 |

## どのように探査されるか

| プラットフォーム | ソース |
|---|---|
| Linux | `/sys/class/dmi/id/{board_vendor,board_name,board_version,bios_vendor,bios_version}` |
| macOS | `system_profiler SPHardwareDataType -json`、ベンダーは `Apple` |

探査はベストエフォートであり、決して例外を投げません。ボード情報を読み取れないホストでは、すべてのフィールドが空文字列のままになります。macOS のパスには BIOS の概念がないため、そこでは `bios_vendor` と `bios_version` は空のままです。
