from contextlib import suppress
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
        return self._read_sys(f"/sys/block/{self.name}/device/model")

    @cached_property
    def kind(self) -> DiskKind:
        """Drive technology — NVMe, SSD, HDD, or Unknown."""
        if self.name.startswith("nvme"):
            return DiskKind.NVME
        rotational = self._read_sys(f"/sys/block/{self.name}/queue/rotational")
        if not rotational:
            return DiskKind.UNKNOWN
        return DiskKind.HDD if rotational == "1" else DiskKind.SSD

    @cached_property
    def size_bytes(self) -> int:
        """Total device capacity in bytes."""
        size = self._read_sys(f"/sys/block/{self.name}/size")
        return int(size) * 512 if size else 0

    @cached_property
    def serial(self) -> str | None:
        """Serial number from sysfs; None if unavailable."""
        return self._read_sys(f"/sys/block/{self.name}/device/serial")

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
    def _read_sys(path: str) -> str | None:
        """Return stripped sysfs text, or None if absent or a placeholder value.

        Tolerates missing or unreadable pseudo-files so callers can probe
        Linux-only `/sys/block/` entries without guarding their existence.

        path: sysfs file to read, e.g. `/sys/block/nvme0n1/size`.
        """
        with suppress(OSError):
            value = Path(path).read_text(encoding="utf-8").strip()
            return value if value and value.lower() not in _SYS_PLACEHOLDER else None
        return None
