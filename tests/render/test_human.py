import sys
from typing import TYPE_CHECKING

from mainboard.render import human

if TYPE_CHECKING:
    import pytest


def test_a_table_prints_every_row_under_its_title_and_projects_to_the_given_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Columns come off the data unless a caller names them, and then only those are shown."""
    human.render_table([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}], title="jobs")
    printed = capsys.readouterr().out
    assert "jobs" in printed
    assert all(token in printed for token in ("a", "b", "1", "4"))
    human.render_table([{"a": "1", "b": "2", "c": "3"}], fields=["c"])
    narrowed = capsys.readouterr().out
    assert "3" in narrowed
    assert "a" not in narrowed


def test_a_table_with_nothing_to_show_prints_nothing_and_an_empty_cell_prints_blank(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No rows is no output at all, and a missing value is a gap rather than the word `None`."""
    human.render_table([])
    assert not capsys.readouterr().out.strip()
    human.render_table([{"a": None}])
    assert "None" not in capsys.readouterr().out


def test_a_cell_in_square_brackets_survives_the_render(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A manifest table heading is data, and rich would otherwise read it as a style tag."""
    human.render_table([{"where": "[dev.python.deps]"}])
    assert "[dev.python.deps]" in capsys.readouterr().out


def test_progress_uses_the_live_spinner_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A block reaching several stages says which one it is on instead of standing still."""
    monkeypatch.setattr(human.Console, "is_terminal", property(lambda self: True))
    stages: list[str] = []
    with human.progress("working") as stage:
        stage("second stage")
        stages.append("ran")
    assert stages == ["ran"]


def test_progress_prints_each_stage_as_its_own_line_off_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A background job piped to a log gets real lines instead of a spinner nobody renders.

    `Console.status` answers every `.update()` with silence off a terminal and prints once at
    the very end, so a long onboarding piped to a log stood indistinguishable from a hang.
    """
    monkeypatch.setattr(human.Console, "is_terminal", property(lambda self: False))
    with human.progress("working") as stage:
        stage("first stage")
        stage("second stage")
    assert capsys.readouterr().err.splitlines() == ["working", "first stage", "second stage"]


def test_install_traceback_installs_a_rich_excepthook(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI error boundary, restored afterwards so the suite keeps its own hook."""
    default = sys.excepthook
    monkeypatch.setattr(sys, "excepthook", default)
    human.install_traceback()
    assert sys.excepthook is not default
