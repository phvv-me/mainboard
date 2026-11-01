"""A pueue task that has not started is waiting, never failed, and each state pueue has reads one
way when listed, settled and cancelled.

Everything runs on this machine: `pueue` is a fake first on PATH whose daemon lists the task.
"""

import json
import os
import shlex
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.dispatch.jobs import vocabulary
from mainboard.dispatch.schedulers import Pueue, pueue


class FakePueue:
    """A `pueue` first on this machine's PATH whose daemon lists `tasks` and records every verb it
    is asked but `status`, one line each."""

    @contextmanager
    def first(self) -> Generator[None]:
        """This `pueue` ahead of any other on the local machine's PATH."""
        with local.env(PATH=f"{self.folder}{os.pathsep}{local.env['PATH']}"):
            yield

    @property
    def verbs(self) -> list[list[str]]:
        try:
            said = (self.folder / "verbs").read_text()
        except FileNotFoundError:
            return []
        return [line.split() for line in said.splitlines()]

    def __init__(self, folder: Path, *tasks: Mapping[str, object]) -> None:
        self.folder = folder
        listing = json.dumps({"tasks": {str(task["id"]): task for task in tasks}})
        (folder / "listing.json").write_text(listing)
        program = folder / "pueue"
        program.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = status ]; then cat {shlex.quote(str(folder / "listing.json"))}; '
            f'else echo "$*" >> {shlex.quote(str(folder / "verbs"))}; fi\n'
        )
        program.chmod(0o755)


_DONE = {"enqueued_at": "t", "start": "t", "end": "t"}
_HOLD, _GO = {"enqueued_at": "t", "start": "t"}, ("running", "running", False)
_WAIT, _STAY = ("running", "queued", False), ("running", "", False)
_OVER, _FAIL = ("ok", "", True), ("failed", "", True)

# A status pueue prints, its (verdict, stage, settled), and the verb a cancel sends it (None leaves
# it alone).
_VARIANTS = [
    pytest.param({"Locked": {"previous_status": "Queued"}}, _WAIT, "remove", id="locked"),
    pytest.param("Locked", _WAIT, "remove", id="locked-bare"),
    pytest.param({"Stashed": {"enqueue_at": None}}, _WAIT, "remove", id="stashed"),
    pytest.param({"Queued": {"enqueued_at": "t"}}, _WAIT, "remove", id="queued"),
    pytest.param({"Running": _HOLD}, _GO, "kill", id="running"),
    pytest.param({"Paused": _HOLD}, _STAY, "kill", id="paused"),
    pytest.param({"Done": {**_DONE, "result": "Success"}}, _OVER, None, id="done-success"),
    pytest.param({"Done": {**_DONE, "result": {"Failed": 3}}}, _FAIL, None, id="done-3"),
    pytest.param({"Done": {**_DONE, "result": "Killed"}}, _FAIL, None, id="done-killed"),
    pytest.param(None, ("vanished", "", True), None, id="forgotten"),
]

# A status this tool cannot place, and its (verdict, stage, settled): a cancel refuses it.
_UNPLACED = [
    pytest.param({"Frozen": {}}, _STAY, id="unknown"),
    pytest.param("Frozen", _STAY, id="unknown-bare"),
]


@pytest.fixture
def pueue_host(status: str | Mapping[str, object] | None, tmp_path: Path) -> Generator[FakePueue]:
    """A `pueue` first on PATH whose daemon lists task 7 in `status`, or no task for None."""
    task = [] if status is None else [{"id": 7, "label": "job", "status": status}]
    host = FakePueue(tmp_path, *task)
    with host.first():
        yield host


def listed() -> tuple[str, str, bool]:
    """The verdict, stage and settledness that listing and settling give pueue task 7."""
    found = Pueue().states(local, "/fake-root", ["7"])["7"]
    return found.verdict, found.stage, found.verdict in vocabulary.Word.settled()


def test_the_pueue_table_names_every_state_pueue_has() -> None:
    tags: set[str] = set()
    for case in [*_VARIANTS, *_UNPLACED]:
        match case.values[0]:
            case str() as tag:
                tags.add(tag)
            case Mapping() as tagged:
                tags.update(tagged)
    assert {state.value for state in pueue.PueueState} <= tags


@pytest.mark.parametrize(("status", "meaning", "stop"), _VARIANTS)
def test_every_pueue_state_has_one_meaning_for_listing_settling_and_cancelling(
    pueue_host: FakePueue, meaning: tuple[str, str, bool], stop: str | None
) -> None:
    assert listed() == meaning
    Pueue().cancel(local, "/fake-root", handle="7")
    assert pueue_host.verbs == ([[stop, "7"]] if stop else [])


@pytest.mark.parametrize(("status", "meaning"), _UNPLACED)
def test_a_pueue_state_this_tool_cannot_place_is_never_stopped_by_a_guess(
    pueue_host: FakePueue, meaning: tuple[str, str, bool]
) -> None:
    assert listed() == meaning
    with pytest.raises(MissionError, match="pueue task 7"):
        Pueue().cancel(local, "/fake-root", handle="7")
    assert pueue_host.verbs == [], "a state this tool cannot place is not stopped by a guess"
