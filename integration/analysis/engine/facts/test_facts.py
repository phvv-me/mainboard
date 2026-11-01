"""A fact base serves the views its families derived and refuses every other view, each time a
query names one. A family's name gives no right to the views that begin with it."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from mainboard.analysis.engine.facts import Budget, Facts, Query

_REFUSAL = "`{}` is no view of a registered fact family"
_TIMES = 3


@pytest.fixture
def facts(tmp_path: Path) -> Iterator[Facts]:
    """A fact base with one family, `ModuleFact`, whose views begin with `module_`."""
    base = Facts(tmp_path, Budget(folder=tmp_path / "facts"))
    base.adopt("ModuleFact", {"facts": "SELECT 'module:m.py' AS fact_id, 'm.py' AS path"})
    yield base
    base.close()


def _answers(base: Facts, view: str) -> list[str]:
    """What each of `_TIMES` queries naming `view` came to: the refusal's text, else `served`."""
    answers = []
    for _ in range(_TIMES):
        try:
            base.ensure(Query(sql=f"SELECT * FROM {view}"))
        except ValueError as refusal:
            answers.append(str(refusal))
        else:
            answers.append("served")
    return answers


@pytest.mark.parametrize(
    ("view", "answer"),
    [
        pytest.param("module_facts", "served", id="a-view-the-family-derived"),
        pytest.param(
            "module_surface_facts",
            _REFUSAL.format("module_surface_facts"),
            id="a-view-that-begins-with-the-prefix-of-a-derived-family",
        ),
        pytest.param(
            "rust_surface_clones",
            _REFUSAL.format("rust_surface_clones"),
            id="a-view-of-a-family-nothing-derived",
        ),
    ],
)
def test_a_view_is_served_or_refused_the_same_way_each_time_a_query_names_it(
    facts: Facts, view: str, answer: str
) -> None:
    assert _answers(facts, view) == [answer] * _TIMES
