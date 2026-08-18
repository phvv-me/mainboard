from contextlib import suppress
from functools import cached_property
from pathlib import Path

from patos import FrozenModel

from ..enums import DiskKind
from .partition_info import PartitionInfo

SYS_BLOCK = Path("/sys/block")
SKIP_PREFIXES = frozenset({"loop", "dm-", "sr", "ram", "zram", "fd"})
_SYS_PLACEHOLDER = frozenset({"unknown", "not specified", "none", "n/a"})


def read_sys(path: Path) -> str | None:
    """Return stripped sysfs text, or None if absent or a placeholder value.

    Tolerates missing or unreadable pseudo-files so callers can probe
    Linux-only sysfs entries without guarding their existence first.

    path: sysfs file to read, e.g. `SYS_BLOCK / "nvme0n1" / "size"`.
    """
    with suppress(OSError):
        value = path.read_text(encoding="utf-8").strip()
        return value if value and value.lower() not in _SYS_PLACEHOLDER else None
    return None


class DriveInfo(FrozenModel):
    """One physical block device detected in `SYS_BLOCK`.

    name: kernel device name, e.g. `nvme0n1`.
    """

    name: str

    @property
    def device(self) -> str:
        """Block device path, e.g. `/dev/nvme0n1`."""
        return f"/dev/{self.name}"

    @cached_property
    def kind(self) -> DiskKind:
        """Drive technology, NVMe, SSD, HDD, or Unknown."""
        if self.name.startswith("nvme"):
            return DiskKind.NVME
        rotational = read_sys(SYS_BLOCK / self.name / "queue" / "rotational")
        if not rotational:
            return DiskKind.UNKNOWN
        return DiskKind.HDD if rotational == "1" else DiskKind.SSD

    @cached_property
    def model(self) -> str | None:
        """Drive model string from sysfs, or None if unavailable."""
        return read_sys(SYS_BLOCK / self.name / "device" / "model")

    @cached_property
    def partitions(self) -> tuple[PartitionInfo, ...]:
        """Mounted partitions that belong to this drive."""
        return tuple(
            p
            for p in PartitionInfo.all()
            if Path(p.device).name.startswith(self.name) or p.device == self.device
        )

    @cached_property
    def serial(self) -> str | None:
        """Serial number from sysfs, or None if unavailable."""
        return read_sys(SYS_BLOCK / self.name / "device" / "serial")

    @cached_property
    def size_bytes(self) -> int:
        """Total device capacity in bytes."""
        size = read_sys(SYS_BLOCK / self.name / "size")
        return int(size) * 512 if size else 0

    @property
    def size_gb(self) -> float:
        """Total device capacity in gibibytes."""
        return self.size_bytes / 1024**3
