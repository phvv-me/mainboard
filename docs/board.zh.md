# Board

`Machine.board`（也包含在快照中）报告主机的主板与固件标识，因此快照会记录它来自哪一块物理系统主板和 BIOS。

```python
from mainboard import Machine

board = Machine().board
print(board.vendor, board.model, board.bios_version)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `vendor` | `str` | 主板制造商 |
| `model` | `str` | 主板产品名称 |
| `version` | `str` | 主板修订版本或型号标识 |
| `bios_vendor` | `str` | 固件 vendor |
| `bios_version` | `str` | 固件版本字符串 |

## 如何探测

| 平台 | 来源 |
|---|---|
| Linux | `/sys/class/dmi/id/{board_vendor,board_name,board_version,bios_vendor,bios_version}` |
| macOS | `system_profiler SPHardwareDataType -json`，以 `Apple` 作为 vendor |

探测采取尽力而为的策略，绝不抛出异常。在无法读取主板信息的主机上，每个字段都保持为空字符串。macOS 路径没有 BIOS 的概念，因此那里的 `bios_vendor` 和 `bios_version` 留空。
