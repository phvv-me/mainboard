import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from plumbum import ProcessExecutionError

from mainboard import Board, MissionError, Project
from mainboard.dispatch.schedulers import HostUnreachable
from mainboard.dispatch.state import RunRecord
from mainboard.jobs.beacon import CELL, CELLS
from mainboard.pulse import Probe, Pulse, Pulses, Reading

from .support import Clock, run


def test_a_job_is_only_called_quiet_once_it_has_been_seen_printing_nothing_new(
    tmp_path: Path,
) -> None:
    """The on-disk memory lets a later process know what the first look saw; growth resets the
    silence, and a job that printed nothing yet has no pulse, since a queue looks like a hang."""
    board = SimpleNamespace(root=tmp_path)
    clock = Clock()
    output = {"1": f"{CELLS} 2\n{CELL} passed a.py::t[x]\n", "2": ""}

    def read(records: Sequence[RunRecord]) -> dict[RunRecord, Reading]:
        readings = {
            record: Reading(output=output[record.handle], gpu_pct=97) for record in records
        }
        return {**readings, run("3", target="gone"): Reading(gpu_pct=97)}

    looks = [run("1"), run("2")]
    first = Pulses(board, read=read, clock=clock).taken(looks)
    assert list(first) == [looks[0]]
    seen = first[looks[0]]
    assert seen == Pulse(handle="1", target="gold", progress=seen.progress, gpu_pct=97)
    assert seen.progress.counted == "1/2"

    clock.now += 45
    later = Pulses(board, read=read, clock=clock).taken(looks)
    assert later[looks[0]].quiet_s == 45

    output["1"] += f"{CELL} failed a.py::t[y]\n"
    clock.now += 30
    grown = Pulses(board, read=read, clock=clock).taken(looks)
    assert (grown[looks[0]].quiet_s, grown[looks[0]].progress.failed) == (0, 1)
    assert Pulses(board, read=read, clock=clock).taken([]) == {}


def test_the_memory_forgets_a_job_a_day_after_it_last_grew_and_survives_being_torn(
    tmp_path: Path,
) -> None:
    """Another process may watch a job this look was not asked about, so it is kept a day."""
    board = SimpleNamespace(root=tmp_path)
    memory = tmp_path / Project().out_dir / "pulse.json"
    memory.parent.mkdir(parents=True)
    memory.write_text("torn {", encoding="utf-8")
    clock = Clock()

    def read(records: Sequence[RunRecord]) -> dict[RunRecord, Reading]:
        return {record: Reading(output="epoch 1\n") for record in records}

    pulses = Pulses(board, read=read, clock=clock)
    pulses.taken([run("1")])
    pulses.taken([run("2")])
    assert set(json.loads(memory.read_text(encoding="utf-8"))) == {"gold/1", "gold/2"}
    clock.now += 86_401
    pulses.taken([run("2")])
    assert set(json.loads(memory.read_text(encoding="utf-8"))) == {"gold/2"}

    memory.unlink()
    memory.mkdir()
    assert pulses.taken([run("2")])[run("2")].quiet_s is None


class Remote:
    """An open connection to a host, answering the card query the way `nvidia-smi` would."""

    def __init__(self, cards: tuple[int, str]) -> None:
        self.cards = cards

    def __getitem__(self, program: str | list[str]) -> Remote:
        return self

    def run(self, retcode: None) -> tuple[int, str, str]:
        return (*self.cards, "")


class Logs:
    """A scheduler whose logs are the given texts, one refusing the way a forgotten task does."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def logs(self, remote: Remote, root: str, *, handle: str) -> str:
        if handle not in self.texts:
            raise ProcessExecutionError(["pueue", "log"], 1, "", "no such task")
        return self.texts[handle]


@pytest.mark.parametrize(
    ("kind", "cards", "gpu"),
    [
        pytest.param("ssh", (0, "12\n97\n"), 97, id="a-host-that-runs-the-job-itself"),
        pytest.param("ssh", (127, "nvidia-smi: not found"), None, id="a-host-with-no-driver"),
        pytest.param("pbs", (0, "97\n"), None, id="a-login-node-carries-no-card-of-the-job-s"),
    ],
)
def test_one_connection_reads_every_log_on_a_host_and_its_cards_where_they_are_the_job_s(
    board: Board,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    cards: tuple[int, str],
    gpu: int | None,
) -> None:
    opened: list[str] = []
    remote = Remote(cards)

    @contextmanager
    def connection(target: str) -> Iterator[Remote]:
        opened.append(target)
        yield remote

    monkeypatch.setattr("mainboard.pulse.connection", connection)
    monkeypatch.setattr(Board, "remote_root", lambda self: "/work")
    monkeypatch.setattr(
        "mainboard.pulse.registry.SCHEDULERS.select",
        lambda kind, default: Logs({"1": "epoch 1\n"}),
    )
    monkeypatch.setattr(
        Board, "job", lambda self, handle, host="": SimpleNamespace(transcript=lambda: "rented")
    )
    records = [run("1", kind=kind), run("2", kind=kind), run("9", target="vast", kind="vast")]

    readings = Probe(board)(records)

    assert opened == ["gold"]
    assert readings == {
        records[0]: Reading(output="epoch 1\n", gpu_pct=gpu),
        records[1]: Reading(output="", gpu_pct=gpu),
        records[2]: Reading(output="rented"),
    }


@pytest.mark.parametrize(
    "fault",
    [HostUnreachable("ssh: connect"), MissionError("no root"), OSError("reset")],
    ids=["unreachable", "no-declared-root", "dropped"],
)
def test_a_host_that_will_not_answer_costs_its_own_runs_their_pulse_and_nothing_else(
    board: Board, monkeypatch: pytest.MonkeyPatch, fault: Exception
) -> None:
    def refused(self: Board) -> str:
        raise fault

    monkeypatch.setattr(Board, "remote_root", refused)
    assert Probe(board)([run("1")]) == {}
