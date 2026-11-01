import shutil
from pathlib import Path

from patos import FrozenModel

from ..constants.units import BYTES_PER_GIBIBYTE, DISK_FLOOR
from ..errors import MissionError
from ..project import Project


class Disk(FrozenModel):
    """The filesystem holding `path`, against the floor of free space every large writer leaves.

    A full disk fails every write at once, the half-written file included, so a writer that can
    write a gigabyte or more asks `require` before it starts and refuses while it can still say
    why. The rule is the same wherever the disk is mounted, so it follows the center to any host.

    path: any path on the disk, existing or not; the nearest existing folder at or above it is
        measured.
    """

    path: Path

    @property
    def device(self) -> int:
        """The filesystem's id, the same for every path on it."""
        return self.existing.stat().st_dev

    @property
    def existing(self) -> Path:
        """The nearest existing folder at or above `path`, the one measured."""
        return next(folder for folder in (self.path, *self.path.parents) if folder.exists())

    @property
    def free(self) -> int:
        """The bytes free on the disk now."""
        return shutil.disk_usage(self.existing).free

    def allowance(self, shares: int, *, doing: str) -> int:
        """One of `shares` equal parts of the space free above the floor.

        It is the most a writer may take whose bytes land on this disk `shares` times over (a
        transfer staged, extracted, then kept). Refused as `require` refuses.
        """
        self.require(doing=doing)
        return (self.free - DISK_FLOOR) // shares

    def require(self, need: int = 0, *, doing: str) -> None:
        """Refuse unless the disk keeps the floor free once `need` more bytes are written.

        doing: what was about to write, the start of the refusal (`collecting gold:runs/x`).
        """
        free = self.free
        if free - need >= DISK_FLOOR:
            return
        writes = f" and this writes up to {need / BYTES_PER_GIBIBYTE:.1f} GiB" if need else ""
        raise MissionError(
            f"{doing} would leave {self.path} below the {DISK_FLOOR // BYTES_PER_GIBIBYTE} GiB "
            f"of free disk every large writer keeps: it has {free / BYTES_PER_GIBIBYTE:.1f} GiB "
            f"free{writes}. Free {(need + DISK_FLOOR - free) / BYTES_PER_GIBIBYTE:.1f} GiB on "
            f"that disk first (`{Project().name} code doctor` names the biggest folders)"
        )
