from pathlib import Path

from .base import FrozenModel
from .drive_info import _SKIP_PREFIXES, DriveInfo


class HostDisk(FrozenModel):
    """All physical drives detected on the host.

    cards: one DriveInfo per physical block device; each carries its mounted
    partitions for capacity and filesystem details.
    """

    @property
    def cards(self) -> tuple[DriveInfo, ...]:
        """Physical block devices enumerated from `/sys/block/`; empty when sysfs is absent."""
        try:
            dev_dirs = sorted(Path("/sys/block").iterdir())
        except FileNotFoundError:
            return ()
        return tuple(
            DriveInfo(name=dev_dir.name)
            for dev_dir in dev_dirs
            if not any(dev_dir.name.startswith(pfx) for pfx in _SKIP_PREFIXES)
            and (size := DriveInfo._read_sys(f"/sys/block/{dev_dir.name}/size"))
            and int(size) * 512 > 0
        )

    @property
    def total_bytes(self) -> int:
        """Combined raw capacity of all drives in bytes."""
        return sum(d.size_bytes for d in self.cards)

    @property
    def total_gb(self) -> float:
        """Combined raw capacity of all drives in gibibytes."""
        return self.total_bytes / 1024**3
