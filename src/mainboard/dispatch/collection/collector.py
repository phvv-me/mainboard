"""Portable, staged collection of project-owned evidence through existing OpenSSH."""

import base64
import json
import shutil
import zlib
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from filelock import FileLock
from patos import FrozenModel

from ...core import Disk
from ...core.errors import MissionError
from ...core.project import Project
from ...log import logger
from ...state.evidence import Evidence
from ...state.lake import Lake
from ...trials.artifacts import relative_path
from ...upkeep.disk.transient import Transient
from ..provenance import blob_of
from ..state.digests import KeptDigests
from ..transport.core import SshTransport
from .pack import HELD, MORE

# How many times a pass's bytes land on the center's disk: staged as the archive, again as its
# extracted files, and a third time as the lake's copy. A pass fetches that share of the free disk
# above the floor (`Disk.allowance`), so no pass outgrows the disk however large the collection.
_COPIES = 3

# The most one pass moves, so its stream ends inside the ten-minute stream deadline: a 69 GB survey
# folder once timed out every pass at 35 to 54 GB, never finishing, where 16 GiB leaves twice the
# time at the same speed and the next pass sends the rest.
_PASS = 16 * 2**30


class Eviction(FrozenModel):
    """What one host's `evict` pass did, as it printed it.

    removed: each file removed, with its bytes.
    kept: each file left in place, with why.
    emptied: each directory the removals left empty, removed too.
    """

    removed: dict[str, int] = {}
    kept: dict[str, str] = {}
    emptied: list[str] = []

    @property
    def rows(self) -> list[dict[str, str | int]]:
        """One row per path: what became of it, a removed file's bytes, and why one stayed."""
        rows: list[dict[str, str | int]] = [
            {"path": path, "outcome": "removed", "bytes": size, "why": ""}
            for path, size in self.removed.items()
        ]
        rows += [
            {"path": path, "outcome": "kept", "bytes": "", "why": why}
            for path, why in self.kept.items()
        ]
        rows += [
            {"path": path, "outcome": "emptied", "bytes": "", "why": ""} for path in self.emptied
        ]
        return rows


class Collector:
    """Collect remote evidence into the workspace lake without replacing what the center holds.

    Each transfer is staged before it is kept, so an interrupted download cannot appear as
    complete results and an older node cannot replace newer evidence merely by changing a file's
    timestamp. Collection never deletes the center's evidence and never writes its tree: the lake
    is the evidence's home, readers recall from it, and `mb lake materialize` writes a file back
    for a reader that needs its path. `evict` removes a host's copy of a file only once the lake
    holds it intact at the same path.
    """

    @cached_property
    def evidence(self) -> Evidence:
        return Evidence(Lake.at(self.root))

    @property
    def digests(self) -> str:
        """The workspace-relative digest cache a host keeps for collection and `link` alike."""
        return (
            (Project().out(self.root) / "run" / "collect-digests.json")
            .relative_to(self.root)
            .as_posix()
        )

    @staticmethod
    def _extract(
        archive: Path, *, staged: Path, scope: PurePosixPath
    ) -> tuple[dict[str, str], bool]:
        """Check every path and ZIP checksum in staging before anything is kept.

        Answers what the host left unsent, by path, and whether it stopped short of the rest.
        """
        with ZipFile(archive) as source:
            entries = source.infolist()
            for entry in entries:
                if entry.filename not in {HELD, MORE}:
                    Collector._unpacked(source, entry, staged=staged, scope=scope)
            held: dict[str, str] = next(
                (json.loads(source.read(e)) for e in reversed(entries) if e.filename == HELD), {}
            )
            more = any(entry.filename == MORE for entry in entries)
        for name in held:
            if not relative_path(name).is_relative_to(scope):
                raise ValueError(f"unexpected collection entry: {name}")
        return held, more

    @staticmethod
    def _shipped(*digests: Mapping[str, str]) -> str:
        """The Python expression a host script decodes `digests` from, as a list.

        The digests of a whole evidence tree run to tens of megabytes as a literal, so they travel
        compressed.
        """
        packed = base64.b64encode(zlib.compress(json.dumps(digests).encode(), 6)).decode()
        return f"json.loads(zlib.decompress(base64.b64decode({packed!r})))"

    @staticmethod
    def _unpacked(source: ZipFile, entry: ZipInfo, *, staged: Path, scope: PurePosixPath) -> None:
        """Write one archived file into staging, refusing one outside `scope` or seen twice."""
        relative = relative_path(entry.filename)
        if not relative.is_relative_to(scope) or entry.is_dir():
            raise ValueError(f"unexpected collection entry: {relative}")
        target = staged / relative
        if target.exists():
            raise ValueError(f"duplicate collection entry: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(entry) as incoming, target.open("xb") as output:
            shutil.copyfileobj(incoming, output)

    def __init__(self, root: Path, transport: SshTransport | None = None) -> None:
        self.root = root.resolve()
        self.transport = transport or SshTransport()
        self.transient = Transient(self.root)

    def _diverging(
        self,
        relative: str,
        files: Mapping[str, Path],
        *,
        held: Mapping[str, str],
        kept: Mapping[str, str],
    ) -> str:
        """The blob offered at `relative` when the center holds other bytes there, else empty.

        A destination outside the workspace is refused.
        """
        target = self.root / relative
        if not target.resolve().is_relative_to(self.root):
            raise ValueError(f"collection destination escapes workspace: {target}")
        expected = blob_of(target) if target.is_file() else kept.get(relative)
        offered = held.get(relative) or blob_of(files[relative])
        return "" if expected is None or expected == offered else offered

    def _divert(
        self,
        files: Mapping[str, Path],
        held: Mapping[str, str],
        *,
        scope: PurePosixPath,
        staged: Path,
        kept: Mapping[str, str],
    ) -> dict[str, str]:
        """Keep every path the center already holds as it is, and keep what differs beside it.

        The same path with other bytes (a second registration writing the same file names) used to
        refuse the whole transfer, which left every other file of the run, its receipts included,
        on the host and the job blocked. The center's copy still wins its path, the tree's first,
        else the lake's, and a byte-identical copy is no conflict. A divergent file that was sent
        moves in staging to `<scope>/.conflicts/<sha8>/<its path under scope>` and is kept there; a
        divergent file the host left unsent (the lake holds its bytes elsewhere) is not adopted at
        this path. Answers the paths still to adopt, and warns with how many diverged.
        """
        diverged = 0
        remaining = dict(held)
        for relative in [*files, *held]:
            if not (offered := self._diverging(relative, files, held=held, kept=kept)):
                continue
            diverged += 1
            if relative in files:
                inside = PurePosixPath(relative).relative_to(scope)
                aside = staged / scope / ".conflicts" / offered[:8] / inside
                aside.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(files[relative], aside)
            remaining.pop(relative, None)
        if diverged:
            logger.warning(
                "{} path(s) under {} hold other bytes here than on the host: the center's copy "
                "kept its path, the host's is under {}/.conflicts/<sha8>/",
                diverged,
                scope,
                scope,
            )
        return remaining

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

    def _remote(self, host: str, *, python: str, call: str, operation: str, output: Path) -> None:
        """Run `call` on `host` after the text of `pack.py`, what it prints landing in `output`.

        The script imports what `_shipped` expressions decode with. `output` gets the stream
        deadline, which a pass hashing a whole tree needs.
        """
        script = Path(__file__).with_name("pack.py").read_text(encoding="utf-8") + (
            f"\nimport base64, zlib\n{call}\n"
        )
        self.transport.run(
            (*self.transport.command(host), f"{python} -"),
            host,
            operation=operation,
            input_text=script,
            output=output,
        )

    def evict(self, host: str, *, root: str, path: str, python: str) -> Eviction:
        """Remove the host's copy of each file under `path` the lake holds intact at its path.

        The host hashes through the digest cache collection keeps there, so a file a pass already
        hashed is not read again. root, path, python: as `pull`'s.
        """
        relative = relative_path(path)
        shipped = self._shipped(self.evidence.intact(relative.as_posix()))
        with self.transient.staging(doing=f"evicting {relative} from {host}") as staged:
            report = staged / "eviction.json"
            self._remote(
                host,
                python=python,
                call=(
                    f"evict({root!r}, relative={relative.as_posix()!r}, "
                    f"center=Center(*{shipped}), cache={self.digests!r})"
                ),
                operation="evict",
                output=report,
            )
            return Eviction.model_validate_json(report.read_bytes())

    def merge(self, archive: Path, *, path: str) -> int:
        """Validate the transfer, then keep it in the lake; how many paths are new to the center.

        Local evidence wins any conflict, on disk or in the lake, independent of source-machine
        paths or timestamps. A file the host left unsent because the lake keeps its bytes (named
        in `HELD`) is indexed at its path from that object. Event snapshots retain their original
        offsets so Results can deduplicate overlapping captures. Keeping is idempotent, so a
        retry finishes an interrupted import. A transfer the host ended at its allowance is kept
        whole, then refused, so the job stays unsettled until a later pass has sent the rest.
        """
        scope = relative_path(path)
        with self.transient.staging(doing=f"keeping {scope}") as staged:
            held, more = self._extract(archive, staged=staged, scope=scope)
            lock = Project().out(self.root) / "collection.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(lock, timeout=self.transport.deadline):
                files = {
                    source.relative_to(staged).as_posix(): source
                    for source in staged.rglob("*")
                    if source.is_file()
                }
                held = self._divert(
                    files, held, scope=scope, staged=staged, kept=self._kept(scope)
                )
                left = any(source.is_file() for source in staged.rglob("*"))
                kept = self.evidence.ingest([staged], staged=staged).indexed if left else 0
                kept += self.evidence.adopt(held)
        if more:
            raise MissionError(
                f"kept {kept} path(s) under {scope}; more wait on the host, past what the center "
                "can stage in one pass, and the next pass sends them"
            )
        return kept

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
        shipped = self._shipped(kept, self._local(relative, kept=kept))
        doing = f"collecting {relative} from {host}"
        with self.transient.staging(doing=doing) as staged:
            archive = staged / "transfer.zip"
            allowance = min(Disk(path=staged).allowance(_COPIES, doing=doing), _PASS)
            self._remote(
                host,
                python=python,
                call=(
                    f"pack({root!r}, relative={relative.as_posix()!r}, center=Center(*{shipped}, "
                    f"allowance={allowance}), cache={self.digests!r})"
                ),
                operation="collect",
                output=archive,
            )
            return self.merge(archive, path=relative.as_posix())
