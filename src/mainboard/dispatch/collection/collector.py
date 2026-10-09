"""Portable, staged collection of project-owned evidence through existing OpenSSH."""

import base64
import hashlib
import json
import os
import shutil
import zlib
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from functools import cached_property
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import psutil
from filelock import FileLock

from ...core.project import Project
from ...state.evidence import Evidence
from ...state.lake import Lake
from ..state.digests import KeptDigests
from ..transport import SshTransport
from .pack import HELD


def relative_path(value: str) -> PurePosixPath:
    """`value` checked as a portable project-relative path, the trials package's rule; imported
    on use, since that package loads a dataframe engine no other verb needs."""
    from ...trials.artifacts import relative_path as checked

    return checked(value)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Collector:
    """Collect remote evidence into the workspace lake without replacing what the center holds.

    Each transfer is staged before it is kept, so an interrupted download cannot appear as
    complete results and an older node cannot replace newer evidence merely by changing a file's
    timestamp. Collection never deletes evidence and never writes the tree: the lake is the
    evidence's home, readers recall from it, and `mb lake materialize` writes a file back for a
    reader that needs its path.
    """

    def __init__(self, root: Path, transport: SshTransport | None = None) -> None:
        self.root = root.resolve()
        self.transport = transport or SshTransport()

    @property
    def digests(self) -> str:
        """The workspace-relative digest cache a host keeps for collection and `link` alike."""
        return (
            (Project().out(self.root) / "run" / "collect-digests.json")
            .relative_to(self.root)
            .as_posix()
        )

    @cached_property
    def evidence(self) -> Evidence:
        return Evidence(Lake.at(self.root))

    def merge(self, archive: Path, *, path: str) -> int:
        """Validate the transfer, then keep it in the lake; how many paths are new to the center.

        Local evidence wins any conflict, on disk or in the lake, independent of source-machine
        paths or timestamps. A file the host left unsent because the lake keeps its bytes (named
        in `HELD`) is indexed at its path from that object. Event snapshots retain their original
        offsets so Results can deduplicate overlapping captures. Keeping is idempotent, so a
        retry finishes an interrupted import.
        """
        scope = relative_path(path)
        with self._staging() as staged:
            held = self._extract(archive, staged=staged, scope=scope)
            lock = Project().out(self.root) / "collection.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(lock, timeout=self.transport.deadline):
                files = {
                    source.relative_to(staged).as_posix(): source
                    for source in staged.rglob("*")
                    if source.is_file()
                }
                self._preflight(files, held, kept=self._kept(scope))
                kept = self.evidence.ingest([staged], staged=staged).indexed if files else 0
                return kept + self.evidence.adopt(held)

    def pull(self, host: str, *, root: str, path: str, python: str) -> int:
        """Collect a remote path and return the number of paths new to the center.

        root: destination workspace, interpreted remotely.
        path: project-relative selection with forward slashes.
        python: the command starting mainboard's own Python on the host (`Dialect.python`),
            never a system one. Path arguments travel through standard input instead of shell
            interpolation.
        """
        relative = relative_path(path)
        kept = self._kept(relative)
        # The digests of a whole evidence tree run to tens of megabytes as a literal.
        known = json.dumps([kept, self._local(relative, kept=kept)]).encode()
        packed = base64.b64encode(zlib.compress(known, 6)).decode()
        script = Path(__file__).with_name("pack.py").read_text(encoding="utf-8") + (
            "\nimport base64, zlib\n"
            f"kept, known = json.loads(zlib.decompress(base64.b64decode({packed!r})))\n"
            f"pack({root!r}, relative={relative.as_posix()!r}, kept=kept, known=known, "
            f"cache={self.digests!r})\n"
        )
        with self._staging() as staged:
            archive = staged / "transfer.zip"
            self.transport.run(
                (*self.transport.command(host), f"{python} -"),
                host,
                operation="collect",
                input_text=script,
                output=archive,
            )
            return self.merge(archive, path=relative.as_posix())

    def _kept(self, relative: PurePosixPath) -> dict[str, str]:
        """The digest the workspace lake indexes for each path under `relative`.

        A host sends none of those paths again, nor any file holding one of those digests.
        """
        return {row.path: row.sha256 for row in self.evidence.indexed(relative.as_posix())}

    def _local(self, relative: PurePosixPath, *, kept: Mapping[str, str]) -> dict[str, str]:
        """Digest each file the tree holds under `relative` unless the lake indexes it as is.

        A byte-identical remote copy of one is then not shipped again.
        """
        digests = KeptDigests(self.root, "collection")
        found = {
            key: digests.of(str(path), key=key)
            for path in (self.root / relative).rglob("*")
            if not path.is_symlink() and path.is_file()
            for key in (path.relative_to(self.root).as_posix(),)
        }
        digests.save()
        return {key: digest for key, digest in found.items() if kept.get(key) != digest}

    @staticmethod
    def _extract(archive: Path, *, staged: Path, scope: PurePosixPath) -> dict[str, str]:
        """Check every path and ZIP checksum in staging before anything is kept.

        Answers what the host left unsent, by path.
        """
        held: dict[str, str] = {}
        with ZipFile(archive) as source:
            for entry in source.infolist():
                if entry.filename == HELD:
                    held = json.loads(source.read(entry))
                    continue
                relative = relative_path(entry.filename)
                if not relative.is_relative_to(scope) or entry.is_dir():
                    raise ValueError(f"unexpected collection entry: {relative}")
                target = staged / relative
                if target.exists():
                    raise ValueError(f"duplicate collection entry: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as incoming, target.open("xb") as output:
                    shutil.copyfileobj(incoming, output)
        for name in held:
            if not relative_path(name).is_relative_to(scope):
                raise ValueError(f"unexpected collection entry: {name}")
        return held

    def _preflight(
        self, files: Mapping[str, Path], held: Mapping[str, str], *, kept: Mapping[str, str]
    ) -> None:
        """Refuse a transfer changing the bytes the center holds at any path.

        The tree's copy is compared first, else the lake's; a byte-identical copy is no conflict.
        """
        for relative in [*files, *held]:
            target = self.root / relative
            if not target.resolve().is_relative_to(self.root):
                raise ValueError(f"collection destination escapes workspace: {target}")
            expected = _sha256(target) if target.is_file() else kept.get(relative)
            if expected is not None and expected != (
                held.get(relative) or _sha256(files[relative])
            ):
                where = "local copy preserved" if target.is_file() else "the lake's copy kept"
                raise ValueError(f"conflicting collected evidence, {where}: {target}")

    @contextmanager
    def _staging(self) -> Generator[Path]:
        """Yield a fresh staging directory under the generated `tmp`, named by this process.

        It sits on the tree's filesystem. One a killed or timed-out pass left, its process gone,
        is cleared first, never one a live pass is using.
        """
        parent = Project().out(self.root) / "tmp"
        parent.mkdir(parents=True, exist_ok=True)
        for stale in parent.glob("collect-*"):
            with suppress(ValueError, IndexError):
                if not psutil.pid_exists(int(stale.name.split("-")[1])):
                    shutil.rmtree(stale, ignore_errors=True)
        with TemporaryDirectory(prefix=f"collect-{os.getpid()}-", dir=parent) as temporary:
            yield Path(temporary)
