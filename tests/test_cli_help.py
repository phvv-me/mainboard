from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.cli import build

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], "One interface for environments"),
        (["query"], "--out"),
        (["batch", "run"], "--only"),
        (["ARTIFACTS"], "query"),
        (["--", "--max-usd"], "submit"),
    ],
)
def test_help_reads_live_commands_without_a_workspace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected: str,
) -> None:
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["help", *arguments])
    assert expected in capsys.readouterr().out


def test_help_reports_no_match(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="no command help matches"):
        build(tmp_path)(["help", "unfindable-search-word"])
