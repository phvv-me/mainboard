# Placa

`Machine.board`, también incluido en la instantánea, informa la identidad de la placa madre y el firmware del host, para que una instantánea registre de qué placa física del sistema y BIOS provino.

```python
from mainboard import Machine

board = Machine().board
print(board.vendor, board.model, board.bios_version)
```

| campo | tipo | significado |
|---|---|---|
| `vendor` | `str` | fabricante de la placa madre |
| `model` | `str` | nombre de producto de la placa madre |
| `version` | `str` | revisión o identificador de modelo de la placa madre |
| `bios_vendor` | `str` | proveedor del firmware |
| `bios_version` | `str` | cadena de versión del firmware |

## Cómo se sondea

| plataforma | fuente |
|---|---|
| Linux | `/sys/class/dmi/id/{board_vendor,board_name,board_version,bios_vendor,bios_version}` |
| macOS | `system_profiler SPHardwareDataType -json`, con `Apple` como proveedor |

El sondeo es de mejor esfuerzo y nunca lanza un error. En un host donde la información de la placa es ilegible, cada campo permanece como una cadena vacía. La ruta de macOS no tiene noción de BIOS, así que `bios_vendor` y `bios_version` quedan vacíos allí.
