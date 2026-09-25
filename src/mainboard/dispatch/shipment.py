# What one dispatch runs and ships, read once so every path derived from it agrees on the tree.
# A command line (a tool task or hand-written line) ships the mirror's whole allowlist under the
# provenance of the repository owning its code. A job spelled by file ships its closure, exactly
# the files it imports and its node, under a provenance scoped to those files, and runs through
# one runner in the job's environment.

import hashlib
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..jobs.closure import Closure
from ..jobs.pins import stage
from ..jobs.target import Target
from .provenance import Row, Source, SourceTree, Status, blob_of, listing, registered
from .shared import CLOSURE_VAR, COMMIT_VAR, DEFERRED_VAR, DIGEST_VAR, FIRST_PARTY_VAR, SOURCE_VAR

if TYPE_CHECKING:
    from collections.abc import Sequence


def runner() -> str:
    """The module a job's script runs its target through, `python -m` style."""
    return f"{Project().name}.jobs.call"


class Shipment(FrozenModel):
    """What one dispatch runs, and what it ships to run it. Paths are workspace-relative.

    command: the line the job body runs on the host.
    spelling: what the run's records call it, the command itself or the job as spelled.
    imports: the import roots the job's `PYTHONPATH` names inside the tree it runs from, ahead of
        everything the environment adds.
    listing: one `path blob status` row per shipped file, empty for a command.
    needs: data paths the job reads, linked back to the mirror.
    pins: the job's Hub pins, staged under the workspace in the cache layout and shipped as needs.
    fetch: the results path the job declared.
    first_party: top-level names the workspace's import roots define, which the runner imports
        from the closure alone.
    deferred: top-level names whose whole distribution the closure left to the environment, which
        the runner's finder admits unchecked rather than refusing for want of a listing.
    """

    command: str
    spelling: str
    source: Source
    imports: tuple[str, ...] = ()
    listing: str = ""
    needs: tuple[str, ...] = ()
    pins: tuple[str, ...] = ()
    fetch: str = ""
    first_party: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()

    @classmethod
    def of_command(cls, command: str, *, source: Source, imports: Sequence[str]) -> Shipment:
        """A command line, shipping the mirror under the provenance of the tree owning it."""
        return cls(command=command, spelling=command, source=source, imports=tuple(imports))

    @classmethod
    def of_closure(cls, closure: Closure, *, root: Path) -> Shipment:
        """A job, shipping its closure under a provenance scoped to it, run through the runner."""
        source, rows = SourceTree(root).seal(closure.files, built=closure.built)
        target = closure.target
        head = f"{target.file}::{target.name}" if target.name else target.file
        return cls(
            command=shlex.join(["python", "-m", runner(), head, "--", *target.args]),
            spelling=target.spelling,
            source=source,
            imports=closure.roots,
            listing=listing(rows),
            needs=closure.needs,
            pins=stage(closure.pins, root),
            fetch=closure.fetch,
            first_party=closure.first_party,
            deferred=closure.deferred,
        )

    @property
    def sealed(self) -> bool:
        """Whether this dispatch ships a closure rather than the mirror."""
        return bool(self.listing)

    @property
    def files(self) -> tuple[str, ...]:
        """Explicit closure files, including resources excluded by the ordinary mirror."""
        return tuple(row.partition("\t")[0] for row in self.listing.splitlines())

    @property
    def listing_name(self) -> str:
        """The file the listing is staged under, content-addressed like the job script."""
        return f"closure-{self.source.digest[:12]}.tsv"

    def admit(self, root: Path) -> None:
        """Refuse an unregistered research shipment before remote work or allocation.

        Ordinary commands and software targets pass. Legacy project-specific seal checks run in
        their own protocol; this checks the captured adjacent node and the sealed source bytes
        without importing the job.
        """
        file, _, name = next(iter(shlex.split(self.spelling)), "").partition("::")
        if not file.endswith(".py") or not (target := Target(file=file, name=name)).registration:
            return
        if not self.sealed:
            raise MissionError("research submission requires a captured Mainboard source bundle")
        rows = [
            Row(path=p, blob=b, status=Status(s))
            for p, b, s in (line.split("\t") for line in self.listing.splitlines())
        ]
        if hashlib.sha256(self.listing.encode()).hexdigest() != self.source.digest:
            raise MissionError("research source listing does not match its recorded digest")
        registered(root / target.registration, rows, root=root)
        for row in rows:
            if blob_of(root / row.path) != row.blob:
                raise MissionError(f"{row.path} changed after Mainboard prepared the job")

    def exports(self, closure: str = "") -> dict[str, str]:
        """The provenance variables a run reads, only those with a value to carry.

        closure: where the run finds its listing, empty for a command that ships none.
        """
        carried = {
            SOURCE_VAR: self.source.identity,
            COMMIT_VAR: self.source.commit,
            DIGEST_VAR: self.source.digest,
            CLOSURE_VAR: closure,
            FIRST_PARTY_VAR: ":".join(self.first_party),
            DEFERRED_VAR: ":".join(self.deferred),
        }
        return {name: value for name, value in carried.items() if value}

    def locally(self, root: Path, *, closure: str = "") -> list[str]:
        """The POSIX argv for a local shell or container, with imports and provenance.

        The runner and variables a dispatched script gets, so a local run writes the receipts a
        dispatched one would, with import roots resolved against `root` rather than a pinned tree.

        closure: the staged listing, workspace-relative, empty for a command.
        """
        exported = self.local_exports(root, closure=closure)
        if self.imports:
            exported["PYTHONPATH"] = ":".join(str(root / place) for place in self.imports)
        return [
            "env",
            *(f"{name}={value}" for name, value in exported.items()),
            *shlex.split(self.command),
        ]

    def local_exports(self, root: Path, *, closure: str = "") -> dict[str, str]:
        """The native process environment, without assuming a POSIX `env` executable."""
        exported = self.exports(str(root / closure) if closure else "")
        if self.imports:
            exported = {
                "PYTHONPATH": os.pathsep.join(str(root / place) for place in self.imports),
                **exported,
            }
        return exported
