import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from mainboard import Board
from mainboard.center.migrate import Migration
from mainboard.center.verify import Verification
from mainboard.cli import build
from mainboard.core.section import Section, Verdict
from mainboard.delimiter import Delimiter
from mainboard.dispatch import HostSetup
from mainboard.probe.snapshot import HostFacts
from mainboard.probe.system import Card, System
from mainboard.proc import Processes

# One row of each verdict, the shape every judged report prints.
_ROWS = [
    Section(section="workstation: git", verdict=Verdict.PASS, detail="git 2.51"),
    Section(
        section="path zsh", verdict=Verdict.WARN, detail="missing rg", fix="mainboard install"
    ),
]

# A census that the fixture manifest judges as broken: a platform it does not declare.
_WINDOWS = System(system="Windows", arch="AMD64", gpus=(Card(name="RTX 5080", vram_mb=16303),))


@pytest.mark.parametrize(
    ("rows", "code"),
    [(_ROWS, 0), ([*_ROWS, Section(section="smoke", verdict=Verdict.FAIL, detail="no")], 1)],
    ids=["a warning is a word", "a failure is the exit status"],
)
def test_verify_prints_every_row_and_exits_on_whether_any_failed(
    depot: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rows: list[Section],
    code: int,
) -> None:
    """The suite is one table a script can branch on, the same shape `doctor` prints."""
    monkeypatch.setattr(Verification, "__init__", lambda self, board: None)
    monkeypatch.setattr(Verification, "sections", lambda self: rows)
    with pytest.raises(SystemExit, match=str(code)):
        build(depot)(["center", "verify", "--json"])
    printed = json.loads(capsys.readouterr().out)
    assert [row["section"] for row in printed] == [row.section for row in rows]


def test_migrate_hands_the_destination_and_root_to_the_migration(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verb is a dispatch: which machine, which root, and the rows the move answered."""
    asked: list[tuple[str, str]] = []

    def run(self: Migration) -> list[Section]:
        asked.append((self.destination, self.root))
        self.watch("cloning")
        return _ROWS

    monkeypatch.setattr(Migration, "run", run)
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["center", "migrate", "pedro-home", "--root", "C:/life", "--agent"])
    assert asked == [("pedro-home", "C:/life")]
    assert "path zsh" in capsys.readouterr().out


def test_facts_prints_the_findings_beside_the_facts_and_keeps_json_the_wire_snapshot(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A human reads what the facts mean; a machine reading `--json` gets the facts alone.

    `--json` is what one machine answers another with, so a findings key there would ride into
    every setup record the far side keeps.
    """
    monkeypatch.setattr(Board, "facts", lambda self: HostFacts(hostname="box", system=_WINDOWS))
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["facts", "--agent"])
    out = capsys.readouterr().out
    assert "platform\tfail" in out
    assert "win-64 is not among the declared platforms" in out
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["facts", "--json"])
    assert "findings" not in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("verb", ["setup", "sync"])
def test_setup_and_sync_add_the_findings_their_census_adds_up_to(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    """The record says what the host became and the findings what that means here.

    The JSON mode carries both in one document, since a script has nowhere else to read them.
    """
    onboarded = HostSetup(host="gold", root="/r", hardware=HostFacts(system=_WINDOWS))
    monkeypatch.setattr(Board, "install", lambda self, env, **options: onboarded)
    with pytest.raises(SystemExit, match="0"):
        build(depot)([verb, "gold", "--json"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["host"] == "gold"
    assert "platform" in {row["section"] for row in printed["findings"]}
    with pytest.raises(SystemExit, match="0"):
        build(depot)([verb, "gold"])
    assert "findings: gold" in capsys.readouterr().out


class Chores(Processes):
    """The process chores answering fixed outcomes, each call kept as the line it was."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def kill(self, pids: Sequence[int], *, force: bool = False) -> list[int]:
        self.calls.append(f"kill {list(pids)} force={force}")
        return [7]

    def timeout(self, seconds: float, command: Sequence[str]) -> int:
        self.calls.append(f"timeout {seconds} {list(command)}")
        return 124

    def wait(
        self, *, file: Path | None = None, port: str = "", pid: int = 0, seconds: float = 0.0
    ) -> bool:
        self.calls.append(f"wait {file} {port} {pid} {seconds}")
        return False


def test_proc_verbs_reach_the_portable_process_chores(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each verb is one method, its exit status the answer a script branches on."""
    chores = Chores()
    monkeypatch.setattr("mainboard.cli.Processes", lambda: chores)
    with pytest.raises(SystemExit, match="1"):
        build(depot)(["proc", "kill", "7", "8", "--force"])
    assert "no process 7" in capsys.readouterr().err
    app = build(depot)
    with pytest.raises(SystemExit, match="124"):
        app(Delimiter(app).placed(["proc", "timeout", "1.5", "pytest", "-x"]))
    with pytest.raises(SystemExit, match="1"):
        build(depot)(["proc", "wait", "--port", "localhost:1", "--timeout", "2"])
    assert chores.calls == [
        "kill [7, 8] force=True",
        "timeout 1.5 ['pytest', '-x']",
        "wait None localhost:1 0 2.0",
    ]


def test_proc_kill_exits_clean_when_every_process_was_there(
    depot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing gone is nothing to report."""
    monkeypatch.setattr(Processes, "kill", lambda self, pids, force=False: [])
    monkeypatch.setattr(Processes, "wait", lambda self, **conditions: True)
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["proc", "kill", "7"])
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["proc", "wait", "--pid", "7"])


def test_the_center_group_holds_only_the_monorepo_verbs_and_lint_stays_general(
    depot: Path,
) -> None:
    """Targets never need the center's verbs, so none of them is spelled at the top level too."""
    app = build(depot)
    center = {name for name in app["center"] if not name.startswith("-")}

    assert center == {"git", "paper", "verify", "migrate"}
    assert {"lint", "proc", "center"} <= set(app)
    assert not {"git", "paper"} & set(app)
