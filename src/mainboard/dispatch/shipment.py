# What one dispatch runs and what it ships to run it, read once so every path derived from the
# dispatch agrees about which tree it is.
#
# Two spellings arrive here. A command line is what a tool task or a hand-written line is, and it
# ships the mirror's whole allowlist under the provenance of the repository owning its code. A
# job spelled by file ships its closure, exactly the files it imports and the node it lives in,
# under a provenance scoped to those files, and runs through one runner in the job's environment.

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from ..jobs.closure import Closure
from .provenance import Repositories, Source, listing
from .shared import CLOSURE_VAR, COMMIT_VAR, DEFERRED_VAR, DIGEST_VAR, FIRST_PARTY_VAR, SOURCE_VAR

if TYPE_CHECKING:
    from collections.abc import Sequence


def runner() -> str:
    """The module a job's script runs its target through, `python -m` style."""
    return f"{Project().name}.jobs.call"


class Shipment(FrozenModel):
    """What one dispatch runs, and what it ships to run it.

    command: the line the job body runs on the host.
    spelling: what the run's records call it, the command itself or the job as spelled.
    source: the dispatching tree's provenance, read once.
    imports: the workspace-relative import roots the job's `PYTHONPATH` names inside the tree it
        runs from, ahead of everything the environment adds.
    listing: the closure listing, one `path blob status` row per shipped file, empty for a
        command that ships the mirror.
    needs: workspace-relative data paths the job reads, linked back to the mirror.
    fetch: the results path the job declared, empty when it declared none.
    first_party: the top-level names the workspace's own import roots define, which the runner
        refuses to import from anywhere but the closure.
    deferred: top-level names whose whole distribution the closure left to the environment,
        which the runner's finder admits unchecked rather than refusing for want of a listing.
    """

    command: str
    spelling: str
    source: Source
    imports: tuple[str, ...] = ()
    listing: str = ""
    needs: tuple[str, ...] = ()
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
        source, rows = Repositories(root).seal(closure.owner, closure.files, built=closure.built)
        target = closure.target
        return cls(
            command=shlex.join(
                ["python", "-m", runner(), f"{target.file}::{target.name}", "--", *target.args]
            ),
            spelling=target.spelling,
            source=source,
            imports=closure.roots,
            listing=listing(rows),
            needs=closure.needs,
            fetch=closure.fetch,
            first_party=closure.first_party,
            deferred=closure.deferred,
        )

    @property
    def sealed(self) -> bool:
        """Whether this dispatch ships a closure rather than the mirror."""
        return bool(self.listing)

    @property
    def listing_name(self) -> str:
        """The file the listing is staged under, content-addressed like the job script."""
        return f"closure-{self.source.digest[:12]}.tsv"

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
        """The argv that runs this shipment here, its import roots and provenance exported.

        The same runner and the same variables a dispatched script gets, so a local run writes
        receipts a dispatched one would, with the import roots resolved against this workspace
        rather than a pinned tree.

        root: the workspace root the import roots resolve against.
        closure: the staged listing, workspace-relative, empty for a command.
        """
        exported = self.exports(str(root / closure) if closure else "")
        if self.imports:
            exported = {
                "PYTHONPATH": ":".join(str(root / place) for place in self.imports),
                **exported,
            }
        return [
            "env",
            *(f"{name}={value}" for name, value in exported.items()),
            *shlex.split(self.command),
        ]
