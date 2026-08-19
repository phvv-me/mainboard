import json
from typing import TYPE_CHECKING

import pytest

from mainboard import Board
from mainboard.cli import build
from mainboard.deps import Change, Dependencies
from mainboard.doctor import Doctor, Section, Verdict
from mainboard.scaffold import Scaffold, Scaffolded

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'

_MOVED = [
    Change(name="tqdm", where="[dev.python.deps]", before="absent", after=">=4.70.0, <5"),
    Change(name="tqdm", where="pixi.lock", before="absent", after="4.70.0"),
]


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str, str, bool, bool]]:
    """Record what each dependency verb asked for, without editing or solving anything."""
    calls: list[tuple[str, str, str, str, bool, bool]] = []

    def add(
        self: Dependencies,
        spec: str,
        *,
        ecosystem: str = "conda",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
    ) -> list[Change]:
        calls.append(("add", spec, ecosystem, env, dev, resolve))
        return _MOVED

    def remove(
        self: Dependencies,
        name: str,
        *,
        ecosystem: str = "",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
    ) -> list[Change]:
        calls.append(("remove", name, ecosystem, env, dev, resolve))
        return _MOVED

    def upgrade(
        self: Dependencies,
        name: str = "",
        *,
        ecosystem: str = "",
        env: str = "",
        dev: bool = False,
    ) -> list[Change]:
        calls.append(("upgrade", name, ecosystem, env, dev, True))
        return _MOVED

    monkeypatch.setattr(Dependencies, "add", add)
    monkeypatch.setattr(Dependencies, "remove", remove)
    monkeypatch.setattr(Dependencies, "upgrade", upgrade)
    return calls


def test_add_passes_its_flags_through_and_prints_what_moved(
    workspace: Path,
    asked: list[tuple[str, str, str, str, bool, bool]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The short ecosystem flag, the dev flag and the resolve toggle all reach the verb."""
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["add", "tqdm", "-l", "python", "--dev", "--no-resolve", "--json"])
    assert asked == [("add", "tqdm", "python", "", True, False)]
    printed = json.loads(capsys.readouterr().out)
    assert [row["where"] for row in printed] == ["[dev.python.deps]", "pixi.lock"]


def test_remove_searches_the_whole_manifest_unless_narrowed(
    workspace: Path,
    asked: list[tuple[str, str, str, str, bool, bool]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ecosystem flag means no narrowing, which is what makes the bare verb usable."""
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["remove", "tqdm"])
    assert asked == [("remove", "tqdm", "", "", False, True)]
    assert "pixi.lock" in capsys.readouterr().out


def test_upgrade_carries_no_name_when_the_whole_lock_is_meant(
    workspace: Path,
    asked: list[tuple[str, str, str, str, bool, bool]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare upgrade is a different request from a named one and reaches the verb as one."""
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["upgrade", "--agent"])
    assert asked == [("upgrade", "", "", "", False, True)]
    assert capsys.readouterr().out.splitlines()[0] == "name\twhere\tbefore\tafter"


def test_new_prints_the_rows_to_paste_above_the_record(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A snippet nobody can read out of a table cell is a snippet nobody pastes."""
    made = Scaffolded(
        project="scratch-probe",
        path=str(workspace / "research/scratch-probe"),
        tasks=str(workspace / "research/scratch-probe/mainboard.tasks.toml"),
        paste=f"{workspace / 'mainboard.toml'} [tasks]",
        snippet=_ROWS,
    )
    monkeypatch.setattr(Scaffold, "render", lambda self, name, **given: made)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["new", "scratch probe"])
    out = capsys.readouterr().out
    assert out.startswith(_ROWS)
    assert "scratch-probe" in out


def test_new_carries_the_snippet_as_a_field_in_the_compact_modes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine reading the record needs the rows in it, not printed above it."""
    made = Scaffolded(project="p", path="/p", snippet=_ROWS)
    monkeypatch.setattr(Scaffold, "render", lambda self, name, **given: made)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["new", "p", "--json"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["snippet"] == _ROWS


def test_doctor_exits_nonzero_only_when_something_is_actually_broken(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sleeping host must never make the exit status mean the network instead of the code."""
    warned = [Section(section="fleet", verdict=Verdict.WARN, detail="gold is asleep")]
    monkeypatch.setattr(Doctor, "sections", lambda self: warned)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["doctor", "--json"])
    assert json.loads(capsys.readouterr().out)[0]["verdict"] == "warn"


def test_doctor_fails_the_shell_when_a_section_fails(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit status is what a script branches on, so a broken workspace has to show there."""
    broken = [Section(section="math", verdict=Verdict.FAIL, detail="6 breakages", fix="fix it")]
    monkeypatch.setattr(Doctor, "sections", lambda self: broken)
    with pytest.raises(SystemExit, match="1"):
        build(workspace)(["doctor"])
    assert "6 breakages" in capsys.readouterr().out


def test_the_board_hands_out_each_new_surface(workspace: Path) -> None:
    """Every verb reaches its subsystem through the one addressable interface."""
    board = Board(workspace)
    assert isinstance(board.deps(), Dependencies)
    assert isinstance(board.doctor(), Doctor)
    assert isinstance(board.scaffold(), Scaffold)
