import re
import sys
from typing import TYPE_CHECKING

import pytest

from mainboard.render import human

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mainboard.render.values import Row

_WIDE = {f"column_{index}": f"value-{index:02d}-{'x' * 12}" for index in range(10)}


@pytest.mark.parametrize(
    ("rows", "fields", "shown", "hidden"),
    [
        ([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}], None, ["a", "b", "1", "4"], []),
        ([{"a": "1", "b": "2", "c": "3"}], ["c"], ["3"], ["a"]),
        ([{"a": None}], None, [], ["None"]),
        # rich would read a manifest table heading as a style tag and print nothing.
        ([{"where": "[dev.python.deps]"}], None, ["[dev.python.deps]"], []),
    ],
    ids=["every_column", "projected", "empty_cell", "bracketed_cell"],
)
def test_a_table_prints_its_cells_as_data_under_its_title(
    capsys: pytest.CaptureFixture[str],
    rows: Sequence[Row],
    fields: Sequence[str] | None,
    shown: list[str],
    hidden: list[str],
) -> None:
    human.render_table(rows, fields=fields, title="jobs")
    printed = capsys.readouterr().out
    assert all(token in printed for token in ["jobs", *shown])
    assert not any(token in printed for token in hidden)


def test_a_table_with_no_rows_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    human.render_table([])
    assert not capsys.readouterr().out.strip()


def test_a_long_cell_wraps_rather_than_cut_to_an_ellipsis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    handle = "e775" * 40
    human.render_table([{"handle": handle}], title="verdict")
    printed = capsys.readouterr().out
    assert "…" not in printed
    # Folding only breaks lines, so the handle reads back whole once the box is stripped.
    assert handle in "".join(character for character in printed if character.isalnum())


def test_progress_uses_the_live_spinner_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(human.Console, "is_terminal", property(lambda self: True))
    stages: list[str] = []
    with human.progress("working") as stage:
        stage("second stage")
        stages.append("ran")
    assert stages == ["ran"]


def test_progress_prints_each_stage_as_its_own_line_off_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(human.Console, "is_terminal", property(lambda self: False))
    with human.progress("working") as stage:
        stage("first stage")
        stage("second stage")
    assert capsys.readouterr().err.splitlines() == ["working", "first stage", "second stage"]


def test_install_traceback_installs_a_rich_excepthook(monkeypatch: pytest.MonkeyPatch) -> None:
    default = sys.excepthook
    monkeypatch.setattr(sys, "excepthook", default)
    human.install_traceback()
    assert sys.excepthook is not default


def test_a_wide_table_off_a_terminal_keeps_each_row_on_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    human.render_table([_WIDE], title="wide")
    printed = capsys.readouterr().out
    assert all(value in printed for value in _WIDE.values())
    assert sum(line.count("value-") for line in printed.splitlines()) == 10
    assert max(line.count("value-") for line in printed.splitlines()) == 10


def test_a_wide_table_on_a_terminal_folds_to_the_terminals_own_width(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(human.Console, "is_terminal", property(lambda self: True))
    monkeypatch.setenv("COLUMNS", "60")
    human.render_table([_WIDE], title="wide")
    printed = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert max(len(line) for line in printed.splitlines()) <= 60
