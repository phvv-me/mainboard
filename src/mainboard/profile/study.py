# A swept study: one collection policy, many input points, one row each, point beside profile.

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .profiler import Collection, Profiler
from .protocols import DeviceProbe
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
    def has_evidence(self) -> bool:
        """Whether this point captured evidence, independently of any scientific verdict."""
        return bool(
            self.profile.summaries
            or self.profile.windows
            or self.profile.kernels
            or self.profile.memcpys
            or self.profile.activities
        )


@dataclass(frozen=True, slots=True)
class Study[P: Point]:
    """A collection policy and the points to apply it at.

    collection: what to gather at every point, held once rather than restated per point so two
        points cannot silently differ in how they were measured.
    points: the input configurations to visit, usually a product of axes built by the caller.
    gpus: the devices available to every point's `Profiler` session (mainboard.probe is not a
        dependency of profiling, so the caller resolves and passes them).
    """

    collection: Collection = field(default_factory=Collection)
    points: tuple[P, ...] = ()
    gpus: Sequence[DeviceProbe] = ()

    @classmethod
    def over(
        cls,
        points: Sequence[P],
        *,
        collection: Collection | None = None,
        gpus: Sequence[DeviceProbe] = (),
    ) -> Study[P]:
        """Build a study over `points`, all measured the same way."""
        return cls(collection=collection or Collection(), points=tuple(points), gpus=gpus)

    def run(self, work: Callable[[P], None], *, warm: bool = True) -> tuple[Row[P], ...]:
        """Measure `work` at every point, returning one row each.

        Work and warmup exceptions propagate. A collected span does not make failed work
        successful. Use parametrized pytest trials when points need independent failure
        handling and durable receipts; this helper only returns a successful sweep.

        `warm` runs the first point once before anything is measured, because whatever a target
        compiles or allocates on its first call is charged to whichever point happens to come
        first. Left off, the first row of a GPU sweep read 4630 ms against its neighbours' 2.5,
        which is a property of the harness masquerading as a property of that point.
        """
        rows = []
        if warm and self.points:
            work(self.points[0])
        for point in self.points:
            label = point.label
            started = time.perf_counter()
            with Profiler.under(self.collection, gpus=self.gpus) as profiler:
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
