"""A swept study: one collection policy, many input points, one row each.

A measurement taken at a single point cannot say which of its conditions produced it, and a
study that varies nothing cannot separate a property of the system from a property of the input
it happened to use. Both failures are cheap to make and expensive to find, so this exists to
make the conditions part of the artifact rather than part of the setup.

What varies is domain knowledge and stays with the domain. A tokenizer varies document length
and script mix; a matrix kernel varies shape and dtype. Mainboard supplies the sweep, the
policy and the row, and asks only that a point can name itself, which is what `Point` says.

Rows are the artifact rather than a chart. The same run has to be re-readable as new questions
arrive, so a row keeps its point beside its profile and nothing is reduced on the way in. The
document that cost sixteen seconds in the tokenizer study was invisible in every average taken
over it and obvious the moment a distribution survived.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .profiler import Collection, Profiler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .result import Profile


@runtime_checkable
class Point(Protocol):
    """One input configuration a study visits.

    A study needs two things from a point and deliberately not a third. A name, so a row can be
    read and a facet can be titled. And that it be serialisable, so the conditions travel with
    the measurement. What the point actually configures is the domain's business.
    """

    @property
    def label(self) -> str:
        """Return a short name identifying this point among the others."""
        ...


@dataclass(frozen=True, slots=True)
class Row[P: Point]:
    """One point's conditions beside what was observed there.

    Both halves are kept. A throughput number whose input specification is not attached is hard
    to reproduce and easy to misattribute, since the axis that explains it may not be the one
    the caller thought they were varying. Generic over the caller's own point type, so a facet
    reading a domain field back off `row.point` sees that field rather than the bare `Point`
    protocol every study accepts.
    """

    label: str
    point: P
    profile: Profile
    seconds: float = 0.0

    @property
    def failed(self) -> bool:
        """Return whether this point produced no evidence at all."""
        return not (self.profile.summaries or self.profile.kernels or self.profile.python)


@dataclass(frozen=True, slots=True)
class Study[P: Point]:
    """A collection policy and the points to apply it at.

    collection: what to gather at every point, held once rather than restated per point so two
        points cannot silently differ in how they were measured.
    points: the input configurations to visit, usually a product of axes built by the caller.
    """

    collection: Collection = field(default_factory=Collection)
    points: tuple[P, ...] = ()

    @classmethod
    def over(cls, points: Sequence[P], *, collection: Collection | None = None) -> Study[P]:
        """Build a study over `points`, all measured the same way."""
        return cls(collection=collection or Collection(), points=tuple(points))

    def run(self, work: Callable[[P], None], *, warm: bool = True) -> tuple[Row[P], ...]:
        """Measure `work` at every point, returning one row each.

        A point that raises still yields a row, with whatever evidence was gathered before it
        failed. A sweep that abandons its results because one configuration is unsupported
        wastes every point before it, and a failed point is itself a finding.

        `warm` runs the first point once before anything is measured, because whatever a target
        compiles or allocates on its first call is charged to whichever point happens to come
        first. Left off, the first row of a GPU sweep read 4630 ms against its neighbours' 2.5,
        which is a property of the harness masquerading as a property of that point.
        """
        rows = []
        if warm and self.points:
            with suppress(Exception):
                work(self.points[0])
        for point in self.points:
            label = point.label
            started = time.perf_counter()
            # A failed point is a row and not a dead sweep, since abandoning the results would
            # waste every point measured before it and the failure is itself a finding.
            with Profiler.under(self.collection) as profiler, suppress(Exception):
                work(point)
            rows.append(
                Row(
                    label=label,
                    point=point,
                    profile=profiler.result(),
                    seconds=time.perf_counter() - started,
                )
            )
        return tuple(rows)
