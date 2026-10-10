"""The roots a host keeps whatever scheduler or center it can still reach.

A launch pins its source tree and environment prefix on the host before anything is queued, and
a sweep reads those pins back. The pins protect the roots where they already stand, the host's
`dispatch/sources` and `prefixes/<environment>`, so adopting them moves and rebuilds nothing. No
scheduler verdict retires a pin: only the submission authority, which knows every evidence
obligation has ended, may.

Retention assumes one release per host. A release without it sweeps the same roots and neither
reads the pins nor honors them, so two releases pruning one host at once could collect what the
newer pinned. The drain-and-switch cutover guarantees it: every host changes release in one step
after its old controllers drain.
"""

import hashlib
import os
from contextlib import suppress
from functools import cached_property, reduce
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from filelock import FileLock, Timeout
from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Callable

# The pin directory the host's standard-library agent writes, which `Snapshot.__retain` spells.
_PINS = ".retention"


class Retention(FrozenModel):
    """What must outlive a sweep: executing roots, historical proofs, and what is not known.

    sources: trees something queued, held or running executes from.
    prefixes: environments a pinned launch activates.
    proofs: roots historical evidence cites, kept apart from executables with live users.
    unknown: every observation that could not be made, which keeps every root.
    """

    version: Literal[1] = 1
    sources: frozenset[str] = frozenset()
    prefixes: frozenset[str] = frozenset()
    proofs: frozenset[str] = frozenset()
    unknown: tuple[str, ...] = ()

    def including(self, other: Retention) -> Retention:
        """Join two observations; neither one's gaps are filled by the other."""
        return Retention(
            sources=self.sources | other.sources,
            prefixes=self.prefixes | other.prefixes,
            proofs=self.proofs | other.proofs,
            unknown=self.unknown + other.unknown,
        )

    def protects(self, path: Path) -> bool:
        """Say whether a sweep must keep `path`, which anything unknown makes certain.

        Kept is a path that is, holds or lies inside a named root, spelled or resolved.
        """
        spellings = {path.absolute(), Path(os.path.realpath(path))}
        return bool(self.unknown) or any(
            spelling in self._named
            or spelling in self._enclosing
            or not self._named.isdisjoint(spelling.parents)
            for spelling in spellings
        )

    @cached_property
    def _enclosing(self) -> frozenset[Path]:
        """Every directory that holds a named root."""
        return frozenset(parent for root in self._named for parent in root.parents)

    @cached_property
    def _named(self) -> frozenset[Path]:
        """Each named root as written and as it resolves, since an alias reaches the same bytes."""
        written = {Path(root) for root in (*self.sources, *self.prefixes, *self.proofs)}
        return frozenset(written | {Path(os.path.realpath(root)) for root in written})


class RetentionPins:
    """One host's pins, and the lock that orders its agent's pinning against sweeps.

    root: the host's workspace root.
    out: the generated-state directory name under `root`.
    """

    # The directory name a snapshot's path carries, which scripts and queues spell it by.
    SOURCES: ClassVar[str] = "sources"

    def __init__(self, root: Path, out: str) -> None:
        # Under the dispatch state `dispatch.shared.state_dir` names, which no transfer prunes.
        self.base = root / out / "dispatch" / self.SOURCES
        self._lock = FileLock(
            self.base / ".pin.lock",
            is_singleton=True,
            preserve_lock_file=True,
            fallback_to_soft=False,
        )

    def swept(self, inventory: Retention, sweep: Callable[[Retention], list[str]]) -> list[str]:
        """Answer what `sweep` removed, under the pin lock and with the host's pins joined in.

        Housekeeping never delays a launch: while the agent holds the lock nothing is swept, and
        nothing is swept while any ownership is unknown. The pins are read under the lock
        because an agent may have pinned since the caller took `inventory`.
        """
        self.base.mkdir(parents=True, exist_ok=True)
        with suppress(Timeout), self._lock.acquire(timeout=0):
            retained = inventory.including(self._read())
            return [] if retained.unknown else sweep(retained)
        return []

    @staticmethod
    def _pin(path: Path) -> Retention:
        """Read the pin at `path`, which must be the immutable record its content names."""
        content = path.read_bytes()
        pin = Retention.model_validate_json(content)
        roots = (*pin.sources, *pin.prefixes, *pin.proofs)
        if path.name != hashlib.sha256(content).hexdigest() + ".json":
            raise ValueError(f"{path} is not named by its content")
        if not all(Path(root).is_absolute() for root in roots):
            raise ValueError(f"{path} names a relative root")
        return pin

    def _read(self) -> Retention:
        """Join every pin published here; one that cannot be read leaves ownership unknown."""
        directory = self.base / _PINS
        if not directory.exists():
            return Retention()
        try:
            pins = [self._pin(path) for path in directory.iterdir() if path.name[0] != "."]
        except OSError, ValueError:
            return Retention(unknown=(f"unreadable pins in {directory}",))
        return reduce(Retention.including, pins, Retention())
