from . import drive_info as drive_info_mod
from .drive_info import DriveInfo


class HostDisk:
    """All physical drives detected on the host, each carrying its mounted partitions."""

    @property
    def cards(self) -> tuple[DriveInfo, ...]:
        """Non-empty physical block devices in sysfs, pseudo devices skipped; empty off Linux.

        Reads `drive_info.SYS_BLOCK` at call time (not a copied import) so the root a test
        points at also governs every `DriveInfo` field it builds.
        """
        try:
            dev_dirs = sorted(drive_info_mod.SYS_BLOCK.iterdir())
        except OSError:
            return ()
        return tuple(
            DriveInfo(name=dev_dir.name)
            for dev_dir in dev_dirs
            if not any(dev_dir.name.startswith(pfx) for pfx in drive_info_mod.SKIP_PREFIXES)
            and drive_info_mod.capacity_bytes(dev_dir) > 0
        )

    @property
    def total_bytes(self) -> int:
        """Combined raw capacity of all drives in bytes."""
        return sum(d.size_bytes for d in self.cards)

    @property
    def total_gb(self) -> float:
        """Combined raw capacity of all drives in gibibytes."""
        return self.total_bytes / 1024**3
