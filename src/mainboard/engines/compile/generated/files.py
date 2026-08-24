from contextlib import contextmanager

# `Path` backs a pydantic field below, so it must resolve at class-creation time. See the
# matching comment on `Toml` in ../platforms.py for why ruff's flake8-type-checking cannot tell.
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from patos import FrozenModel

from .writer import Writer

if TYPE_CHECKING:
    from collections.abc import Generator

    from filelock import FileLock

# One FileLock instance per generated directory, shared across every in-process acquisition.
# filelock is reentrant per INSTANCE, so a caller may open a transaction (stale-check plus
# recompile) while `provision` already holds the same lock around the whole install, where two
# separate instances on the same path would deadlock. Cross-process exclusion is unchanged, it
# lives in the OS lock underneath.
_LOCKS: dict[Path, FileLock] = {}


class GeneratedFiles(FrozenModel):
    """Atomic generated-file writes guarded by one workspace sync lock."""

    directory: Path

    @contextmanager
    def locked(self) -> Generator[Writer]:
        """Serialize compilers that target the same generated directory.

        The :class:`Writer` exists only for the body of this context, so holding the lock is
        not a convention a caller can forget but the only way to reach a write at all.

        filelock is imported here rather than at the top of the file because it drags asyncio in
        behind it, 13 ms of a cold start this package's entry point pays on every command, and
        the only commands that write a generated file are the ones that reach this line.
        """
        from filelock import FileLock

        self.directory.mkdir(exist_ok=True)
        key = self.directory.resolve()
        lock = _LOCKS.setdefault(key, FileLock(key / ".sync.lock"))
        with lock:
            yield Writer(lock)
