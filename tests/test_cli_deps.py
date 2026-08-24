import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.cli import build
from mainboard.doctor import Doctor, Section, Verdict

if TYPE_CHECKING:
    from pathlib import Path

    from .support import Relayed

_ROWS = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["add", "tqdm", "--json"], ["[dev.python.deps]", "pixi.lock"]),
        (["remove", "tqdm"], ["pixi.lock"]),
        (["upgrade", "--agent"], ["name\twhere\tbefore\tafter"]),
    ],
    ids=["as json", "as the default rich table", "as the compact table"],
)
def test_every_dependency_verb_prints_the_constraint_and_the_pins_its_solve_moved(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected: list[str],
) -> None:
    """An edit and its dragged pins render as one shape.

    Both are the same fact, something moved from one version to another somewhere, so one
    table carries them rather than two.
    """
    with pytest.raises(SystemExit, match="0"):
        build(depot)(argv)
    out = capsys.readouterr().out
    if argv[-1] == "--json":
        assert [row["where"] for row in json.loads(out)] == expected
        return
    assert all(fragment in out for fragment in expected)


@pytest.mark.parametrize(
    "compact", [False, True], ids=["printed above the record", "carried inside the record"]
)
def test_new_prints_the_rows_to_paste_where_the_reader_can_reach_them(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    compact: bool,
) -> None:
    """Rows print pasteable at a terminal and stay whole in the record.

    Repeating them wrapped inside a table cell would only make them harder to copy back out,
    while a machine reading the record has nowhere else to get them.
    """
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["new", "scratch probe", *(["--json"] if compact else [])])
    out = capsys.readouterr().out
    if compact:
        assert json.loads(out)["snippet"] == _ROWS
        return
    assert out.startswith(_ROWS)
    assert "snippet" not in out


def test_new_refuses_an_answer_written_without_its_value(depot: Path) -> None:
    """A bare word is a question nobody answered, and guessing what it meant helps no one."""
    with pytest.raises(MissionError, match="question=value"):
        build(depot)(["new", "p", "--answer", "home"])


@pytest.mark.parametrize(
    ("verdict", "code", "detail"),
    [
        (Verdict.WARN, "0", "gold is asleep"),
        (Verdict.FAIL, "1", "6 breakages"),
    ],
    ids=["a sleeping host is a word, not a nonzero exit", "a broken workspace shows in the exit"],
)
def test_doctor_exits_nonzero_only_when_something_is_actually_broken(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verdict: Verdict,
    code: str,
    detail: str,
) -> None:
    """A sleeping host never bends the exit status.

    The status is what a script branches on, so it must mean the code and not the network.
    """
    found = [Section(section="fleet", verdict=verdict, detail=detail, fix="fix it")]
    monkeypatch.setattr(Doctor, "sections", lambda self: found)
    with pytest.raises(SystemExit, match=code):
        build(depot)(["doctor"])
    assert detail in capsys.readouterr().out
