import os
import platform
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ....core import MissionError

if TYPE_CHECKING:
    from io import IOBase

    from filelock import FileLock


class Writer:
    """Generated-file edits, handed out by `GeneratedFiles.locked()` and valid only under its lock.

    Every edit re-checks the lock, so an instance stashed past its block fails loudly instead of
    racing whoever holds the lock now.
    """

    def __init__(self, lock: FileLock) -> None:
        self.lock = lock

    def held(self) -> None:
        if not self.lock.is_locked:
            raise MissionError(
                "The workspace sync lock is no longer held, so nothing may be written."
            )

    def remove(self, path: Path) -> None:
        """Drop a file, a link (never followed to its source), or a tree such as a vendored dep."""
        self.held()
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
            return
        shutil.rmtree(path)

    def link(self, path: Path, target: Path) -> None:
        """Point one generated symlink at `target`, replacing whatever stands there now.

        A vendored path dependency is a real directory of links, so an edit under the source
        reaches the next import with nothing to re-vendor, while a resolver handed the directory
        cannot record somewhere else.
        """
        self.held()
        if path.is_symlink() and path.readlink() == target:
            return
        self.remove(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target, target_is_directory=target.is_dir())

    def write(self, path: Path, text: str | bytes) -> None:
        """Replace a generated file atomically, encoding text as UTF-8 without newline changes."""
        self.held()
        content = text.encode("utf-8") if isinstance(text, str) else text
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            existing = None
        if existing == content:
            return
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                self._make_portable(stream)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _make_portable(stream: IOBase) -> None:
        """Set mode 0644, independent of the umask, except on Windows.

        There Python 3.14's chmod turns 0644 into a protected owner-only DACL instead of the
        inherited one, leaving the file unreadable to another process identity.
        """
        if platform.system() != "Windows":
            os.fchmod(stream.fileno(), 0o644)
