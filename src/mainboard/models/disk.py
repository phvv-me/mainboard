from __future__ import annotations

from pathlib import Path

from ..enums import DiskKind
from .base import FrozenModel


def _read_sys(path: Path) -> str | None:
    """Return stripped text from a sysfs file, or None if absent/unreadable."""
    try:
        value = path.read_text().strip()
        return (
            value if value and value.lower() not in {"unknown", "not specified", "none"} else None
        )
    except OSError:
        return None


class PartitionInfo(FrozenModel):
    """One mounted filesystem partition.

    device: block device path, e.g. `/dev/nvme0n1p1`.
    mountpoint: filesystem mount path, e.g. `/`.
    fstype: filesystem type, e.g. `ext4`.
    readonly: True when mounted read-only.
    total_bytes / used_bytes / free_bytes: capacity from `statvfs`.
    """

    device: str
    mountpoint: str
    fstype: str
    readonly: bool = False
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0

    @property
    def total_gb(self) -> float:
        """Total capacity in gibibytes."""
        return self.total_bytes / 1024**3

    @property
    def used_gb(self) -> float:
        """Used space in gibibytes."""
        return self.used_bytes / 1024**3

    @property
    def free_gb(self) -> float:
        """Free space in gibibytes."""
        return self.free_bytes / 1024**3

    @property
    def utilization_pct(self) -> float:
        """Percentage of total capacity currently used."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes * 100


class DriveInfo(FrozenModel):
    """One physical block device detected in `/sys/block/`.

    device: device path, e.g. `/dev/nvme0n1`.
    model: drive model string from sysfs; None if unavailable.
    kind: drive technology — NVMe, SSD, HDD, or Unknown.
    size_bytes: total device capacity in bytes.
    serial: serial number from sysfs; None if unavailable.
    partitions: mounted partitions that belong to this drive.
    """

    device: str
    model: str | None = None
    kind: DiskKind = DiskKind.UNKNOWN
    size_bytes: int = 0
    serial: str | None = None
    partitions: tuple[PartitionInfo, ...] = ()

    @property
    def size_gb(self) -> float:
        """Total device capacity in gibibytes."""
        return self.size_bytes / 1024**3


class HostDisk(FrozenModel):
    """All physical drives detected on the host.

    cards: one DriveInfo per physical block device; each carries its mounted
    partitions for capacity and filesystem details.
    """

    cards: tuple[DriveInfo, ...]

    @property
    def total_bytes(self) -> int:
        """Combined raw capacity of all drives in bytes."""
        return sum(d.size_bytes for d in self.cards)

    @property
    def total_gb(self) -> float:
        """Combined raw capacity of all drives in gibibytes."""
        return self.total_bytes / 1024**3
