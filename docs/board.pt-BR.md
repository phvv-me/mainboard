# Board

`Machine.board`, também presente no snapshot, informa a identidade da placa-mãe e do firmware do host, de modo que um snapshot registra de qual board física do sistema e de qual BIOS ele veio.

```python
from mainboard import Machine

board = Machine().board
print(board.vendor, board.model, board.bios_version)
```

| campo | tipo | significado |
|---|---|---|
| `vendor` | `str` | fabricante da placa-mãe |
| `model` | `str` | nome de produto da placa-mãe |
| `version` | `str` | revisão ou identificador de modelo da placa-mãe |
| `bios_vendor` | `str` | fornecedor do firmware |
| `bios_version` | `str` | string de versão do firmware |

## Como é sondado

| plataforma | fonte |
|---|---|
| Linux | `/sys/class/dmi/id/{board_vendor,board_name,board_version,bios_vendor,bios_version}` |
| macOS | `system_profiler SPHardwareDataType -json`, com `Apple` como vendor |

O probe é best-effort e nunca levanta exceção. Em um host onde as informações da board são ilegíveis, todos os campos permanecem como string vazia. O caminho do macOS não tem noção de BIOS, então `bios_vendor` e `bios_version` ficam vazios ali.
