import os
import platform
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

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
        """Drop what the manifest no longer asks for: a file, a link, or a whole tree.

        A tree because a vendored path dependency is a directory of links, and a distribution
        the manifest stopped declaring has to leave with the same call that retires a generated
        script. A link is unlinked rather than followed, so retiring one never reaches the
        source it points at.
        """
        self.held()
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
            return
        shutil.rmtree(path)

    def link(self, path: Path, target: Path) -> None:
        """Point one generated symlink at `target`, replacing whatever stands there now.

        How a vendored path dependency reaches its source: the entry is a link, so an edit under
        the source is seen by the next import with nothing to re-vendor, while the directory
        holding it stays real and a resolver handed it cannot record somewhere else.
        """
        self.held()
        if path.is_symlink() and path.readlink() == target:
            return
        self.remove(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target, target_is_directory=target.is_dir())

    def write(self, path: Path, text: str) -> None:
        """Replace one generated text file only after its complete contents reach disk."""
        self.held()
        try:
            existing = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        if existing == text:
            return
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                self._make_portable(stream)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

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
