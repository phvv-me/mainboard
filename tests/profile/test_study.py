# A sweep keeps each point's conditions and never mistakes partial evidence for success.

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from patos import FrozenModel

from mainboard import Collection, span
from mainboard import ProfileStudy as Study
from mainboard.profile import Feature, Point, Row

from .support import one_process_gpu


class Shape(FrozenModel):
    """Whatever a domain varies, which the study never needs to understand."""

    size: int

    @property
    def label(self) -> str:
        return f"size{self.size}"


# Every example opens a profiler session per point, so the example budget stays small.
@settings(max_examples=10)
@given(sizes=st.lists(st.integers(min_value=0, max_value=99), min_size=1, max_size=3, unique=True))
def test_every_point_keeps_its_conditions_beside_its_measurement(sizes: Sequence[int]) -> None:
    """Policy and devices are stated once per sweep, so two points cannot silently differ in
    how they were measured; a point without evidence still comes back as a row saying so."""
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
    assert all(
        isinstance(row, Row) and row.seconds >= 0.0 and not row.has_evidence for row in rows
    )


@pytest.mark.parametrize("warm", [True, False])
def test_work_that_raises_after_collecting_evidence_still_fails(warm: bool) -> None:
    """Neither warmup nor a measured point may turn a swallowed exception into a row."""
    visited: list[int] = []

    def work(point: Shape) -> None:
        visited.append(point.size)
        with span("partial"):
            raise RuntimeError("unsupported on this device")

    study = Study.over(
        [Shape(size=size) for size in (1, 2, 3)], collection=Collection(features=Feature.SPANS)
    )
    with pytest.raises(RuntimeError, match="unsupported on this device"):
        study.run(work, warm=warm)
    assert visited == [1]


def test_evidence_is_not_a_failure_or_scientific_verdict() -> None:
    study = Study.over([Shape(size=1)], collection=Collection(features=Feature.SPANS))

    def work(point: Shape) -> None:
        with span(point.label):
            pass

    assert study.run(work, warm=False)[0].has_evidence


def test_the_sweep_warms_before_it_measures() -> None:
    """First-call compilation must not be charged to the first point: left off, a GPU sweep's
    first row read 4630 ms against its neighbours' 2.5."""
    calls: list[str] = []
    points = (Shape(size=1), Shape(size=2))
    Study(points=points).run(lambda point: calls.append(point.label))
    assert calls == ["size1", "size1", "size2"], calls

    calls.clear()
    Study.over(points).run(lambda point: calls.append(point.label), warm=False)
    assert calls == ["size1", "size2"], calls
    assert Study(points=points).collection == Collection()
