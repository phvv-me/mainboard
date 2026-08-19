import sys
from typing import TYPE_CHECKING

from mainboard.render import human

if TYPE_CHECKING:
    import pytest


def test_render_table_prints_every_row_and_column(capsys: pytest.CaptureFixture[str]) -> None:
    human.render_table([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
    out = capsys.readouterr().out
    assert "a" in out
    assert "b" in out
    assert "1" in out
    assert "4" in out


def test_render_table_projects_to_the_given_fields(capsys: pytest.CaptureFixture[str]) -> None:
    human.render_table([{"a": "1", "b": "2", "c": "3"}], fields=["c"])
    out = capsys.readouterr().out
    assert "c" in out
    assert "3" in out
    assert "a" not in out


def test_render_table_shows_a_title_when_given(capsys: pytest.CaptureFixture[str]) -> None:
    human.render_table([{"a": "1"}], title="jobs")
    assert "jobs" in capsys.readouterr().out


def test_render_table_handles_no_rows_and_no_fields(capsys: pytest.CaptureFixture[str]) -> None:
    human.render_table([])
    assert not capsys.readouterr().out.strip()


def test_render_table_renders_a_none_cell_as_blank(capsys: pytest.CaptureFixture[str]) -> None:
    human.render_table([{"a": None}])
    out = capsys.readouterr().out
    assert "None" not in out


def test_progress_runs_its_block_as_a_context_manager() -> None:
    ran = False
    with human.progress("working"):
        ran = True
    assert ran


def test_install_traceback_installs_a_rich_excepthook() -> None:
    default = sys.excepthook
    human.install_traceback()
    assert sys.excepthook is not default


def test_progress_yields_a_stage_setter_the_block_can_call() -> None:
    with human.progress("working") as stage:
        stage("second stage")


def test_a_cell_in_square_brackets_survives_the_render(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A manifest table heading is data, and rich would otherwise read it as a style tag."""
    human.render_table([{"where": "[dev.python.deps]"}])
    assert "[dev.python.deps]" in capsys.readouterr().out
