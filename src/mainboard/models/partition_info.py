from functools import cached_property

import psutil

from .base import FrozenModel
from .psutil_results import DiskUsage


class PartitionInfo(FrozenModel):
    """One mounted filesystem partition.

    device: block device path, e.g. `/dev/nvme0n1p1`.
    mountpoint: filesystem mount path, e.g. `/`.
    fstype: filesystem type, e.g. `ext4`.
    opts: raw mount options string from psutil, e.g. `rw,relatime`.
    """

    device: str
    mountpoint: str
    fstype: str
    opts: str = ""

    @classmethod
    def all(cls) -> tuple[PartitionInfo, ...]:
        """Return all mounted physical partitions."""
        return tuple(
            cls(
                device=p.device,
                mountpoint=p.mountpoint,
                fstype=p.fstype,
                opts=p.opts,
            )
            for p in psutil.disk_partitions(all=False)
        )

    @property
    def readonly(self) -> bool:
        """True when mounted read-only."""
        return "ro" in self.opts.split(",")

    @cached_property
    def _usage(self) -> DiskUsage | None:
        """Disk usage from `statvfs`; None if the mount is inaccessible."""
        try:
            usage: DiskUsage = psutil.disk_usage(self.mountpoint)
            return usage
        except PermissionError, OSError:
            return None

    @property
    def total_bytes(self) -> int:
        """Total partition capacity in bytes."""
        return self._usage.total if self._usage else 0

    @property
    def used_bytes(self) -> int:
        """Used capacity in bytes."""
        return self._usage.used if self._usage else 0

    @property
    def free_bytes(self) -> int:
        """Free capacity in bytes."""
        return self._usage.free if self._usage else 0

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
