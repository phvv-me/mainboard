# A sweep has to keep each point's conditions, and survive a point that fails.

from collections.abc import Sequence

from hypothesis import given, settings
from hypothesis import strategies as st
from patos import FrozenModel

from mainboard import Collection
from mainboard import ProfileStudy as Study
from mainboard.profile import Feature, Point, Row

from .support import one_process_gpu


class Shape(FrozenModel):
    """A stand-in for whatever a domain varies, which the study never needs to understand."""

    size: int

    @property
    def label(self) -> str:
        return f"size{self.size}"


# Every example opens a profiler session per point, the most expensive body in this slice, so
# the budget is trimmed from the shared default.
@settings(max_examples=10)
@given(sizes=st.lists(st.integers(min_value=0, max_value=99), min_size=1, max_size=3, unique=True))
def test_every_point_keeps_its_conditions_beside_its_measurement(sizes: Sequence[int]) -> None:
    """A number whose input specification is not attached is hard to reproduce.

    The policy and the devices are stated once for the whole sweep, so two points cannot
    silently differ in how they were measured, and a point that produced no evidence at
    all still comes back as a row that says so.
    """
    points = [Shape(size=size) for size in sizes]
    assert all(isinstance(point, Point) for point in points)
    gpu = one_process_gpu()
    study = Study.over(points, collection=Collection(features=Feature.SPANS), gpus=(gpu,))
    assert study.collection.features is Feature.SPANS
    assert study.gpus == (gpu,)
    # Serialisable, so the policy can be stored beside the rows it produced.
    assert "features" in study.collection.model_dump_json()

    rows = study.run(lambda _: None)
    assert [row.label for row in rows] == [point.label for point in points]
    assert [row.point for row in rows] == points
    assert all(isinstance(row, Row) and row.seconds >= 0.0 and row.failed for row in rows)


def test_one_failing_point_does_not_discard_the_others() -> None:
    """Abandoning a sweep because one configuration is unsupported wastes every point before it."""

    def work(point: Shape) -> None:
        if point.size == 2:
            raise RuntimeError("unsupported on this device")

    rows = Study.over([Shape(size=size) for size in (1, 2, 3)]).run(work)
    assert len(rows) == 3
    assert [row.label for row in rows] == ["size1", "size2", "size3"]


def test_the_sweep_warms_before_it_measures() -> None:
    """Whatever a target compiles on its first call must not be charged to the first point.

    Left off, the first row of a GPU sweep read 4630 ms against its neighbours' 2.5, which
    is a property of the harness masquerading as a property of that point. A bare `Study()`
    warms too, since it carries the same default policy.
    """
    calls: list[str] = []
    points = (Shape(size=1), Shape(size=2))
    Study(points=points).run(lambda point: calls.append(point.label))
    assert calls == ["size1", "size1", "size2"], calls

    calls.clear()
    Study.over(points).run(lambda point: calls.append(point.label), warm=False)
    assert calls == ["size1", "size2"], calls
    assert Study(points=points).collection == Collection()
