import re
from contextlib import suppress
from pathlib import Path

from patos import FrozenModel

_INFINIBAND_ROOT = Path("/sys/class/infiniband")


def _read_field(path: Path) -> str:
    """Return stripped sysfs text, or an empty string when the file is absent or unreadable.

    path: sysfs file to read, e.g. a port's `state` or `rate` file.
    """
    with suppress(OSError):
        return path.read_text(encoding="utf-8").strip()
    return ""


class FabricPort(FrozenModel):
    """One InfiniBand or RoCE fabric port.

    device: the HCA device name, e.g. `mlx5_0`.
    port: the 1-based port number on the device.
    state: raw link state, e.g. `4: ACTIVE`.
    rate: raw link rate, e.g. `400 Gb/sec (4X NDR)`.
    link_layer: the fabric technology, `InfiniBand` or `Ethernet` (RoCE).
    """

    device: str
    port: int
    state: str = ""
    rate: str = ""
    link_layer: str = ""


def _device_order(device_dir: Path) -> tuple[str | int, ...]:
    """A sort key reading a device name's digits as numbers, so `mlx5_2` precedes `mlx5_10`.

    Splitting on digit runs alternates text and number, and the split always starts with text,
    so two names compare field by field with matching kinds throughout.

    device_dir: the HCA device directory being ordered.
    """
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", device_dir.name)
    )


class Fabric:
    """InfiniBand and RoCE fabric ports detected in sysfs."""

    @staticmethod
    def port_dirs(device_dir: Path) -> list[Path]:
        """Numbered port directories under one HCA device, in port order."""
        try:
            return sorted(
                (p for p in (device_dir / "ports").iterdir() if p.name.isdigit()),
                key=lambda p: int(p.name),
            )
        except OSError:
            return []

    @classmethod
    def probe(cls, root: Path = _INFINIBAND_ROOT) -> tuple[FabricPort, ...]:
        """Every fabric port found under `root`, empty when that sysfs tree is absent.

        root: the `infiniband` class directory to scan, a test feeds a fake tmp tree here.
        """
        try:
            device_dirs = sorted(root.iterdir(), key=_device_order)
        except OSError:
            return ()
        return tuple(
            FabricPort(
                device=device_dir.name,
                port=int(port_dir.name),
                state=_read_field(port_dir / "state"),
                rate=_read_field(port_dir / "rate"),
                link_layer=_read_field(port_dir / "link_layer"),
            )
            for device_dir in device_dirs
            for port_dir in cls.port_dirs(device_dir)
        )
