from __future__ import annotations

from functools import cached_property
from pathlib import Path

from ..enums import DiskKind
from .base import FrozenModel
from .partition_info import PartitionInfo

_SKIP_PREFIXES = frozenset({"loop", "dm-", "sr", "ram", "zram", "fd"})
_SYS_PLACEHOLDER = frozenset({"unknown", "not specified", "none", "n/a"})


class DriveInfo(FrozenModel):
    """One physical block device detected in `/sys/block/`.

    name: kernel device name, e.g. `nvme0n1`.
    """

    name: str

    @property
    def device(self) -> str:
        """Block device path, e.g. `/dev/nvme0n1`."""
        return f"/dev/{self.name}"

    @cached_property
    def model(self) -> str | None:
        """Drive model string from sysfs; None if unavailable."""
        return self._read_sys(Path(f"/sys/block/{self.name}/device/model"))

    @cached_property
    def kind(self) -> DiskKind:
        """Drive technology — NVMe, SSD, HDD, or Unknown."""
        if self.name.startswith("nvme"):
            return DiskKind.NVME
        rotational = Path(f"/sys/block/{self.name}/queue/rotational")
        if rotational.exists():
            return DiskKind.HDD if rotational.read_text().strip() == "1" else DiskKind.SSD
        return DiskKind.UNKNOWN

    @cached_property
    def size_bytes(self) -> int:
        """Total device capacity in bytes."""
        size_path = Path(f"/sys/block/{self.name}/size")
        return int(size_path.read_text().strip()) * 512 if size_path.exists() else 0

    @cached_property
    def serial(self) -> str | None:
        """Serial number from sysfs; None if unavailable."""
        return self._read_sys(Path(f"/sys/block/{self.name}/device/serial"))

    @cached_property
    def partitions(self) -> tuple[PartitionInfo, ...]:
        """Mounted partitions that belong to this drive."""
        return tuple(
            p
            for p in PartitionInfo.all()
            if Path(p.device).name.startswith(self.name) or p.device == self.device
        )

    @property
    def size_gb(self) -> float:
        """Total device capacity in gibibytes."""
        return self.size_bytes / 1024**3

    @staticmethod
    def _read_sys(path: Path) -> str | None:
        """Return stripped sysfs text, or None if absent or a placeholder value."""
        try:
            value = path.read_text().strip()
            return value if value and value.lower() not in _SYS_PLACEHOLDER else None
        except OSError:
            return None
