# A sweep has to keep each point's conditions, and survive a point that fails.

from patos import FrozenModel

from mainboard import Collection
from mainboard import ProfileStudy as Study
from mainboard.profile import Feature, Point, Row

from .conftest import one_process_gpu


class Shape(FrozenModel):
    """A stand-in for whatever a domain varies, which the study never needs to understand."""

    size: int

    @property
    def label(self) -> str:
        return f"size{self.size}"


def test_a_domain_point_satisfies_the_protocol_without_importing_it() -> None:
    """`Point` asks only for a name, so a domain model satisfies it structurally."""
    assert isinstance(Shape(size=1), Point)


def test_every_point_keeps_its_conditions_beside_its_measurement() -> None:
    """A number whose input specification is not attached is hard to reproduce."""
    points = [Shape(size=size) for size in (1, 2, 3)]
    rows = Study.over(points, collection=Collection(features=Feature.SPANS)).run(lambda _: None)
    assert [row.label for row in rows] == ["size1", "size2", "size3"]
    assert [row.point for row in rows] == points
    assert all(row.seconds >= 0.0 for row in rows)


def test_one_failing_point_does_not_discard_the_others() -> None:
    """Abandoning a sweep because one configuration is unsupported wastes every point before it."""

    def work(point: Shape) -> None:
        if point.size == 2:
            raise RuntimeError("unsupported on this device")

    rows = Study.over([Shape(size=size) for size in (1, 2, 3)]).run(work)
    assert len(rows) == 3
    assert [row.label for row in rows] == ["size1", "size2", "size3"]


def test_the_policy_and_gpus_are_stated_once_for_the_whole_sweep() -> None:
    """Two points cannot differ in how they were measured if there is only one policy."""
    gpu = one_process_gpu()
    study = Study.over([Shape(size=1)], collection=Collection(features=Feature.SPANS), gpus=(gpu,))
    assert study.collection.features is Feature.SPANS
    assert study.gpus == (gpu,)
    # Serialisable, so the policy can be stored beside the rows it produced.
    assert "features" in study.collection.model_dump_json()


def test_rows_report_a_point_that_produced_nothing() -> None:
    """A point with no evidence is a finding, so it has to be visible rather than absent."""
    rows = Study.over([Shape(size=1)], collection=Collection(features=Feature.SPANS)).run(
        lambda _: None
    )
    assert isinstance(rows[0], Row)
    assert rows[0].failed is True


def test_the_sweep_warms_before_it_measures() -> None:
    """Whatever a target compiles on its first call must not be charged to the first point."""
    calls: list[str] = []
    points = [Shape(size=1), Shape(size=2)]
    Study.over(points).run(lambda point: calls.append(point.label))
    assert calls == ["size1", "size1", "size2"], calls
    Study.over(points).run(lambda point: calls.clear(), warm=False)


def test_study_default_collection_and_gpus_are_empty() -> None:
    """A bare `Study()` collects with the default policy and no GPU."""
    study = Study(points=(Shape(size=1),))
    assert study.collection == Collection()
    assert study.gpus == ()
