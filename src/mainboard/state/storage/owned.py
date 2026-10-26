# A FILE OF THE TREE THE TOOL MAY READ, WRITE AND REMOVE, and nothing a link could lead to.
#
# Checking a path and then using it leaves a gap in which a folder of it can be swapped for a
# symlink, so `OwnedFile` opens every folder below an owned root through the descriptor of the one
# above it, refusing links, and does the work on that descriptor. The root itself is opened as
# spelled, since its owner chose it. `tree_of` lists what a folder holds on the same terms.

import hashlib
import os
import secrets
import stat
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager, suppress
from io import BufferedReader, BufferedWriter
from pathlib import Path

from ...core.errors import MissionError


class OwnedFile:
    """A file under an owned `root`, reached through directory descriptors that follow no link.

    A link anywhere below the root refuses with MissionError, and so does a file that is not a
    regular one.
    """

    @contextmanager
    def _open(self, directory: int) -> Generator[BufferedReader]:
        try:
            descriptor = os.open(
                self.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
        except FileNotFoundError:
            raise
        except OSError as fault:
            raise MissionError(f"unsafe evidence file {self.path}: {fault}") from fault
        with open(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise MissionError(f"{self.path} is not a regular evidence file")
            yield stream

    @contextmanager
    def parent(self, *, create: bool = False) -> Generator[int]:
        """The descriptor of the folder holding the file, made first below the root when asked.

        Raises FileNotFoundError when a folder is missing and was not to be made.
        """
        with ExitStack() as cleanup:
            if create:
                self.root.mkdir(parents=True, exist_ok=True)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            cleanup.callback(os.close, directory)
            for component in self.relative.parts[:-1]:
                directory = self._descend(component, directory=directory, create=create)
                cleanup.callback(os.close, directory)
            yield directory

    @contextmanager
    def quarantine(self) -> Generator[int]:
        """Move the exact file aside into a private folder beside it, yielding that folder.

        Leaving the block returns whatever is still there to the file's name and removes the
        folder. If a new file has taken the name meanwhile nothing is overwritten: the moved file
        stays in its folder and MissionError names both.
        """
        with self.parent() as directory, ExitStack() as cleanup:
            name = f".mb-custody-{secrets.token_hex(16)}"
            os.mkdir(name, mode=0o700, dir_fd=directory)
            isolated = self._descend(name, directory=directory, create=False)
            cleanup.callback(os.close, isolated)
            cleanup.callback(self._return, directory=directory, isolated=isolated, name=name)
            os.rename(self.path.name, self.path.name, src_dir_fd=directory, dst_dir_fd=isolated)
            yield isolated

    @contextmanager
    def read(self) -> Generator[BufferedReader]:
        """The file opened for reading.

        Raises FileNotFoundError when it is missing.
        """
        with self.parent() as directory, self._open(directory) as stream:
            yield stream

    @contextmanager
    def staging(self, *, directory: int) -> Generator[tuple[str, BufferedWriter]]:
        """A new private file in `directory`, as its name and an open writer, removed on exit.

        It is created exclusively and owner-only, whatever stands at that name.
        """
        name = f".mb-{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        with ExitStack() as cleanup:
            descriptor = os.open(name, flags, 0o600, dir_fd=directory)
            cleanup.callback(self._unlink, name, directory)
            yield name, cleanup.enter_context(os.fdopen(descriptor, "wb"))

    @staticmethod
    def _unlink(name: str, directory: int) -> None:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=directory)

    def __init__(self, path: Path, *, root: Path) -> None:
        self.root = Path(os.path.abspath(root))
        self.path = Path(os.path.abspath(path))
        try:
            self.relative = self.path.relative_to(self.root)
        except ValueError:
            raise MissionError(f"{path} lies outside {root}") from None
        if not self.relative.parts:
            raise MissionError(f"{path} is not a file")

    def _confirm(self, digest: str, *, directory: int) -> None:
        if (actual := self.digest_at(directory)) != digest:
            raise MissionError(
                f"{self.path}: destination {actual}; lake {digest}; refusing overwrite"
            )

    def _descend(self, component: str, *, directory: int, create: bool) -> int:
        if create:
            with suppress(FileExistsError):
                os.mkdir(component, dir_fd=directory)
        try:
            return os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
        except FileNotFoundError:
            raise
        except OSError as fault:
            raise MissionError(f"unsafe evidence path {self.path}: {fault}") from fault

    def _link_back(self, *, directory: int, isolated: int, name: str) -> None:
        try:
            os.link(
                self.path.name,
                self.path.name,
                src_dir_fd=isolated,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise MissionError(
                f"{self.path}: retained candidate at {self.path.parent / name / self.path.name} "
                f"(candidate {self.digest_at(isolated)}; destination "
                f"{self.digest_at(directory)}); refusing overwrite"
            ) from None

    def _return(self, *, directory: int, isolated: int, name: str) -> None:
        with suppress(FileNotFoundError):
            self._link_back(directory=directory, isolated=isolated, name=name)
            self.unlink_at(isolated)
        os.rmdir(name, dir_fd=directory)

    def digest(self) -> str:
        """The SHA-256 of the file.

        Raises FileNotFoundError when it is missing.
        """
        with self.parent() as directory:
            return self.digest_at(directory)

    def digest_at(self, directory: int) -> str:
        """The SHA-256 of the file as the folder `directory` holds it."""
        with self._open(directory) as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def publish(self, staged: str, *, directory: int, digest: str, replace: bool) -> None:
        """Make the file staged in `directory` the file, unless a different one already is.

        A different file refuses with MissionError naming both digests, unless `replace` swaps
        it in atomically. The staged name stays for its owner to remove.
        """
        if replace:
            os.replace(staged, self.path.name, src_dir_fd=directory, dst_dir_fd=directory)
            return
        try:
            os.link(staged, self.path.name, src_dir_fd=directory, dst_dir_fd=directory)
        except FileExistsError:
            self._confirm(digest, directory=directory)

    def stat_at(self, directory: int) -> os.stat_result:
        """The file's metadata as the folder `directory` holds it, a link not followed."""
        return os.stat(self.path.name, dir_fd=directory, follow_symlinks=False)

    def unlink_at(self, directory: int) -> None:
        """Remove the file from the folder `directory`."""
        os.unlink(self.path.name, dir_fd=directory)


def link_in(path: Path, *, root: Path) -> Path | None:
    """The first link among `path` and its folders below `root`, None when none is a link."""
    absolute = Path(os.path.abspath(path))
    return next(
        (
            part
            for part in (absolute, *absolute.parents)
            if part != root and part.is_relative_to(root) and part.is_symlink()
        ),
        None,
    )


def tree_of(path: Path, *, root: Path) -> list[Path]:
    """The regular files at or under `path`, none of them behind a link.

    Names starting `.mb-` are the tool's own aside (a file being written, a candidate retained
    while an eviction failed), never evidence. Raises MissionError for a link at `path`, among
    its folders below `root` or inside it, and for a path with nothing at it.
    """
    if link := link_in(path, root=root):
        raise MissionError(f"{link} is a link, and evidence never follows one")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise MissionError(f"no evidence at {path}")
    return sorted(_members(path))


def _members(folder: Path) -> Iterator[Path]:
    with os.scandir(folder) as scanned:
        entries = [entry for entry in scanned if not entry.name.startswith(".mb-")]
    if links := [entry.path for entry in entries if entry.is_symlink()]:
        raise MissionError(f"{links[0]} is a link, and evidence never follows one")
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            yield from _members(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):
            yield Path(entry.path)
