import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from ....core import MissionError

if TYPE_CHECKING:
    from io import TextIOWrapper

    from filelock import FileLock


class Writer:
    """Generated-file edits, valid only while the sync lock it was handed is still held.

    A caller cannot build one of these, only receive it from `GeneratedFiles.locked()`, and
    every edit re-checks that lock, so an instance stashed past its block fails loudly instead
    of racing the process that holds the lock now.
    """

    def __init__(self, lock: FileLock) -> None:
        self.lock = lock

    def held(self) -> None:
        """Refuse to touch a generated file once the sync lock has been released."""
        if not self.lock.is_locked:
            raise MissionError(
                "The workspace sync lock is no longer held, so nothing may be written."
            )

    def remove(self, path: Path) -> None:
        """Drop a generated file the manifest no longer asks for, if it is still there."""
        self.held()
        path.unlink(missing_ok=True)

    def write(self, path: Path, text: str) -> None:
        """Replace one generated text file only after its complete contents reach disk."""
        self.held()
        try:
            existing = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        if existing == text:
            return
        with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as directory:
            temporary = Path(directory) / path.name
            with temporary.open("w", encoding="utf-8") as stream:
                self._make_portable(stream)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)

    @staticmethod
    def _make_portable(stream: TextIOWrapper) -> None:
        """Set a public generated-file mode without severing Windows ACL inheritance.

        Python 3.14 implements the full chmod mode surface on Windows. Applying POSIX ``0644``
        there creates a protected owner-only DACL rather than the ordinary inherited ACL of the
        workspace directory, making a generated manifest unreadable to another process identity.
        Windows files therefore keep the ACL inherited at creation; POSIX retains the explicit
        mode that makes a generated artifact independent of the caller's umask.
        """
        if platform.system() != "Windows":
            os.fchmod(stream.fileno(), 0o644)
