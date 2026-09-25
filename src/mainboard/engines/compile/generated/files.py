from contextlib import contextmanager

# `Path` backs a pydantic field below, so it must resolve at class-creation time.
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..state import SyncState
from .writer import Writer

if TYPE_CHECKING:
    from collections.abc import Generator

    from filelock import FileLock

# One FileLock per generated directory, since filelock is reentrant per instance only: a
# transaction (stale-check plus recompile) inside `provision`'s lock would deadlock on a second
# instance. Cross-process exclusion lives in the OS lock underneath.
_LOCKS: dict[Path, FileLock] = {}


class GeneratedFiles(FrozenModel):
    """Atomic generated-file writes guarded by one workspace sync lock."""

    directory: Path

    @property
    def inputs(self) -> tuple[Path, ...]:
        """Generated install files shared by shipment, hashing, and prefix copying.

        State, hidden bookkeeping, installed directories and the host-specific activation script
        are not inputs to an addressed environment.
        """
        return tuple(
            entry
            for entry in sorted(self.directory.glob("*"))
            if entry.is_file()
            and not entry.name.startswith(".")
            and entry.name not in ("activate.sh", SyncState.path(self.directory).name)
        )

    @contextmanager
    def locked(self) -> Generator[Writer]:
        """Serialize compilers that target the same generated directory.

        The `Writer` exists only inside this context, so holding the lock is the only way to
        write. filelock is imported here because it drags in asyncio, 13 ms of every cold start.
        """
        from filelock import FileLock

        # Parents too: a shard's lock may be taken before anything ever compiled into it.
        self.directory.mkdir(parents=True, exist_ok=True)
        key = self.directory.resolve()
        lock = _LOCKS.setdefault(key, FileLock(key / ".sync.lock"))
        with lock:
            yield Writer(lock)
