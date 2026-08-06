"""A sweep has to keep each point's conditions, and survive a point that fails."""

from __future__ import annotations

import pytest

from mainboard.models.base import FrozenModel
from mainboard.profiling.profiler import Collection, Feature
from mainboard.profiling.study import Point, Row, Study


class Shape(FrozenModel):
    """A stand-in for whatever a domain varies, which mainboard never needs to understand."""

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

    def work(point: object) -> None:
        if getattr(point, "size", 0) == 2:
            raise RuntimeError("unsupported on this device")

    rows = Study.over([Shape(size=size) for size in (1, 2, 3)]).run(work)
    assert len(rows) == 3
    assert [row.label for row in rows] == ["size1", "size2", "size3"]


def test_the_policy_is_stated_once_for_the_whole_sweep() -> None:
    """Two points cannot differ in how they were measured if there is only one policy."""
    study = Study.over([Shape(size=1)], collection=Collection(features=Feature.SPANS))
    assert study.collection.features is Feature.SPANS
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


def test_a_sparkline_shows_shape_and_says_nothing_about_missing_points() -> None:
    """A gap has to read as a gap, since a zero would look like a measured collapse."""
    from mainboard.profiling.surface import sparkline

    assert sparkline([1.0, 2.0, 3.0])[0] < sparkline([1.0, 2.0, 3.0])[-1]
    assert sparkline([5.0, 5.0, 5.0]) == "▁▁▁"
    assert sparkline([1.0, 0.0, 3.0])[1] == " "
    assert sparkline([0.0, 0.0]) == "  "


def test_the_surface_groups_one_row_per_within_value() -> None:
    """Reading down the panel is what says whether the second axis changes the first."""
    from mainboard.profiling.surface import facet

    class Cell(FrozenModel):
        run: int
        script: str

        @property
        def label(self) -> str:
            return f"{self.run}/{self.script}"

    points = [Cell(run=run, script=script) for run in (1, 2) for script in ("a", "b")]
    rows = Study.over(points).run(lambda _: None)
    rendered = facet(
        rows,
        along=lambda point: point.run,
        within=lambda point: point.script,
        measure=lambda row: float(row.point.run * 100),
    )
    assert rendered is not None


def test_show_surface_prints_the_facet(capsys: pytest.CaptureFixture[str]) -> None:
    """`show_surface` is the print-to-terminal shorthand around `facet`."""
    from mainboard.profiling.surface import show_surface

    class Cell(FrozenModel):
        run: int
        script: str

        @property
        def label(self) -> str:
            return f"{self.run}/{self.script}"

    points = [Cell(run=run, script=script) for run in (1, 2) for script in ("a", "b")]
    rows = Study.over(points).run(lambda _: None)
    show_surface(
        rows,
        along=lambda point: point.run,
        within=lambda point: point.script,
        measure=lambda row: float(row.point.run * 100),
        color=False,
    )
    assert "a" in capsys.readouterr().out
