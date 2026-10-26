from functools import cached_property

import psutil
from patos import FrozenModel

from ....constants import BYTES_PER_KIBIBYTE


class DiskUsage(FrozenModel):
    """The `psutil.disk_usage()` fields read here, in bytes."""

    free: int = 0
    total: int = 0
    used: int = 0


class PartitionInfo(FrozenModel):
    """One mounted filesystem partition.

    opts: raw mount options string from psutil, e.g. `rw,relatime`.
    """

    device: str
    mountpoint: str
    fstype: str
    opts: str = ""

    @cached_property
    def usage(self) -> DiskUsage:
        """Disk usage from `statvfs`, all zero when the mount is inaccessible."""
        try:
            found = psutil.disk_usage(self.mountpoint)
        except OSError:
            return DiskUsage()
        return DiskUsage(free=found.free, total=found.total, used=found.used)

    @classmethod
    def all(cls) -> tuple[PartitionInfo, ...]:
        """Return all mounted physical partitions."""
        return tuple(
            cls(device=p.device, mountpoint=p.mountpoint, fstype=p.fstype, opts=p.opts)
            for p in psutil.disk_partitions(all=False)
        )

    @property
    def free_bytes(self) -> int:
        """Free capacity in bytes."""
        return self.usage.free

    @property
    def free_gb(self) -> float:
        """Free space in gibibytes."""
        return self.free_bytes / BYTES_PER_KIBIBYTE**3

    @property
    def readonly(self) -> bool:
        """True when mounted read-only."""
        return "ro" in self.opts.split(",")

    @property
    def total_bytes(self) -> int:
        """Total partition capacity in bytes."""
        return self.usage.total

    @property
    def total_gb(self) -> float:
        """Total capacity in gibibytes."""
        return self.total_bytes / BYTES_PER_KIBIBYTE**3

    @property
    def used_bytes(self) -> int:
        """Used capacity in bytes."""
        return self.usage.used

    @property
    def used_gb(self) -> float:
        """Used space in gibibytes."""
        return self.used_bytes / BYTES_PER_KIBIBYTE**3

    @property
    def utilization_pct(self) -> float:
        """Percentage of total capacity currently used, 0 when the total is unknown."""
        return self.used_bytes / self.total_bytes * 100 if self.total_bytes else 0.0
