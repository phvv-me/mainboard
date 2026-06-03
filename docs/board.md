# Board

`Machine.board`, also carried in the snapshot, reports the host's motherboard and firmware identity, so a snapshot records which physical system board and BIOS it came from.

```python
from mainboard import Machine

board = Machine().board
print(board.vendor, board.model, board.bios_version)
```

| field | type | meaning |
|---|---|---|
| `vendor` | `str` | motherboard manufacturer |
| `model` | `str` | motherboard product name |
| `version` | `str` | motherboard revision or model identifier |
| `bios_vendor` | `str` | firmware vendor |
| `bios_version` | `str` | firmware version string |

## How it is probed

| platform | source |
|---|---|
| Linux | `/sys/class/dmi/id/{board_vendor,board_name,board_version,bios_vendor,bios_version}` |
| macOS | `system_profiler SPHardwareDataType -json`, with `Apple` as the vendor |

Probing is best-effort and never raises. On a host where the board information is unreadable, every field stays an empty string. The macOS path has no BIOS notion, so `bios_vendor` and `bios_version` are left empty there.
