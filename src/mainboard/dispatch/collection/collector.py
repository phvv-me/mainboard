"""Portable, staged collection of project-owned evidence through existing OpenSSH."""

import filecmp
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from filelock import FileLock

from ...core.project import Project
from ...trials.artifacts import relative_path
from ..transport import SshTransport


class KnownDigests:
    """The digests of published evidence, remembered by each file's size and modification time.

    Collection never rewrites a published file, so one whose size and timestamp have not moved
    still holds the bytes it was hashed with. Without this memory every sweep re-read every byte
    under every unsettled run's results path, and one 28 GB evidence tree held a `wait` inside a
    single pass for minutes past its own timeout.
    """

    def __init__(self, path: Path) -> None:
        """path: the JSON file the memory lives in, created on the first save."""
        self.path = path
        try:
            self.held: dict[str, list[int | str]] = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.held = {}

    def of(self, file: Path, *, key: str) -> str:
        """`file`'s SHA-256, read off the memory while its size and timestamp stand."""
        status = file.stat()
        stamp = [status.st_size, status.st_mtime_ns]
        remembered = self.held.get(key)
        if remembered is not None and remembered[:2] == stamp:
            return str(remembered[2])
        with file.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        self.held[key] = [*stamp, digest]
        return digest

    def save(self) -> None:
        """Publish the memory by rename, so a reader never meets half a document."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_suffix(f".{os.getpid()}.tmp")
        staged.write_text(json.dumps(self.held), encoding="utf-8")
        staged.replace(self.path)


class Collector:
    """Collect remote evidence without replacing local files.

    Each transfer is staged before publication, so interrupted downloads cannot appear as
    complete results and an older node cannot replace newer evidence merely by changing a
    file's timestamp. Collection never deletes evidence.
    """

    def __init__(self, root: Path, transport: SshTransport | None = None) -> None:
        self.root = root.resolve()
        self.transport = transport or SshTransport()

    def merge(self, archive: Path, *, path: str) -> int:
        """Validate the transfer, then publish new files without overwriting existing bytes.

        Local evidence wins any conflict, independent of source-machine paths or timestamps.
        Event snapshots retain their original offsets so Results can deduplicate overlapping
        captures. Publication is per file; a retry can finish an interrupted import.
        """
        scope = relative_path(path)
        with TemporaryDirectory(prefix=".mainboard-collect-", dir=self.root) as temporary:
            staged = Path(temporary)
            self._extract(archive, staged=staged, scope=scope.as_posix())
            lock = self.root / Project().out_dir / "collection.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(lock, timeout=self.transport.deadline):
                pending = self._pending(staged)
                return sum(self._publish(source, target=target) for source, target in pending)

    def pull(self, host: str, *, root: str, path: str, python: str = "python3") -> int:
        """Collect a remote path and return the number of new local files.

        root: destination workspace, interpreted remotely.
        path: project-relative selection with forward slashes.
        python: trusted interpreter command in the remote SSH login shell.
            Path arguments travel through standard input instead of shell interpolation.
        """
        relative = relative_path(path)
        script = Path(__file__).with_name("pack.py").read_text(encoding="utf-8")
        known = self._known(relative)
        script += f"\npack({root!r}, relative={relative.as_posix()!r}, known={known!r})\n"
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".mainboard-collect-", dir=self.root) as temporary:
            archive = Path(temporary) / "transfer.zip"
            self.transport.run(
                ("ssh", *self.transport.options, self.transport.destination(host), f"{python} -"),
                host,
                operation="collect",
                input_text=script,
                output=archive,
            )
            return self.merge(archive, path=relative.as_posix())

    def _known(self, relative: PurePosixPath) -> dict[str, str]:
        """Skip only byte-identical published files; live event snapshots still transfer."""
        digests = KnownDigests(self.root / Project().out_dir / "collection.digests.json")
        published = (
            path
            for path in (self.root / relative).rglob("*")
            if not path.is_symlink() and path.is_file() and path.parent.name != "events"
        )
        known = {
            key: digests.of(path, key=key)
            for path in published
            for key in (path.relative_to(self.root).as_posix(),)
        }
        digests.save()
        return known

    @staticmethod
    def _duplicate(source: Path, *, target: Path) -> bool:
        """Validate a racing publication without counting it as a new local file."""
        if not filecmp.cmp(source, target, shallow=False):
            raise ValueError(f"conflicting collected evidence, local copy preserved: {target}")
        return False

    @staticmethod
    def _extract(archive: Path, *, staged: Path, scope: str) -> None:
        """Check every path and ZIP checksum in staging before publication can begin."""
        with ZipFile(archive) as source:
            for entry in source.infolist():
                relative = relative_path(entry.filename)
                if not relative.is_relative_to(scope) or entry.is_dir():
                    raise ValueError(f"unexpected collection entry: {relative}")
                target = staged / relative
                if target.exists():
                    raise ValueError(f"duplicate collection entry: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as incoming, target.open("xb") as output:
                    shutil.copyfileobj(incoming, output)
                    output.flush()
                    os.fsync(output.fileno())

    def _changed(self, source: Path, *, target: Path) -> bool:
        return not target.exists() or self._duplicate(source, target=target)

    def _pending(self, staged: Path) -> list[tuple[Path, Path]]:
        """Preflight conflicts while retaining existing evidence and logical project paths."""
        pending = []
        for source in (path for path in staged.rglob("*") if path.is_file()):
            target = self.root / source.relative_to(staged)
            if not target.resolve().is_relative_to(self.root):
                raise ValueError(f"collection destination escapes workspace: {target}")
            if self._changed(source, target=target):
                pending.append((source, target))
        return pending

    def _publish(self, source: Path, *, target: Path) -> bool:
        """A concurrent writer can publish first, but its file can never be replaced here."""
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            return self._duplicate(source, target=target)
        return True
