from functools import cached_property
from typing import Protocol

import psutil
from patos import FrozenModel


class DiskUsage(Protocol):
    """The `psutil.disk_usage()` fields read here, total, used, and free bytes."""

    @property
    def free(self) -> int: ...
    @property
    def total(self) -> int: ...
    @property
    def used(self) -> int: ...


class PartitionInfo(FrozenModel):
    """One mounted filesystem partition.

    opts: raw mount options string from psutil, e.g. `rw,relatime`.
    """

    device: str
    mountpoint: str
    fstype: str
    opts: str = ""

    @property
    def free_bytes(self) -> int:
        """Free capacity in bytes."""
        return self.usage.free if self.usage else 0

    @property
    def free_gb(self) -> float:
        """Free space in gibibytes."""
        return self.free_bytes / 1024**3

    @property
    def readonly(self) -> bool:
        """True when mounted read-only."""
        return "ro" in self.opts.split(",")

    @property
    def total_bytes(self) -> int:
        """Total partition capacity in bytes."""
        return self.usage.total if self.usage else 0

    @property
    def total_gb(self) -> float:
        """Total capacity in gibibytes."""
        return self.total_bytes / 1024**3

    @cached_property
    def usage(self) -> DiskUsage | None:
        """Disk usage from `statvfs`, or None if the mount is inaccessible."""
        try:
            return psutil.disk_usage(self.mountpoint)
        except OSError:
            return None

    @property
    def used_bytes(self) -> int:
        """Used capacity in bytes."""
        return self.usage.used if self.usage else 0

    @property
    def used_gb(self) -> float:
        """Used space in gibibytes."""
        return self.used_bytes / 1024**3

    @property
    def utilization_pct(self) -> float:
        """Percentage of total capacity currently used, 0 when the total is unknown."""
        return self.used_bytes / self.total_bytes * 100 if self.total_bytes else 0.0

    @classmethod
    def all(cls) -> tuple[PartitionInfo, ...]:
        """Return all mounted physical partitions."""
        return tuple(
            cls(device=p.device, mountpoint=p.mountpoint, fstype=p.fstype, opts=p.opts)
            for p in psutil.disk_partitions(all=False)
        )
