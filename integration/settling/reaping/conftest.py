from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mainboard import Board
from mainboard.dispatch.backends import CloudBackend
from mainboard.manifest import HostProfile
from mainboard.settling.reaping.reaper import Reaper
from mainboard.state import Held, Holdings

from ...dispatch.backends.fakes import Recorded
from .builders import machine, record
from .constants import PROVIDER

_LISTED = "/instances"


@pytest.fixture(autouse=True)
def key(monkeypatch) -> None:
    """A test key, so the recorded answers are read without a real account."""
    monkeypatch.setenv("SHADEFORM_API_KEY", "test-key")


@pytest.fixture
def board(workspace: Path) -> Board:
    """A workspace whose registry holds five creations, one of them held as a session's box.

    m-done settled and was reported, m-owed settled with its report still owed, m-live runs,
    m-held is reported too but held, and mainboard-crash is a create that died before its bind.
    """
    found = Board(workspace)
    for each in (
        record("mainboard-done", handle="m-done", verdict="ok", reported="ok"),
        record("mainboard-owed", handle="m-owed", verdict="ok"),
        record("mainboard-live", handle="m-live", verdict="running"),
        record("mainboard-held", handle="m-held", verdict="ok", reported="ok"),
        record("mainboard-crash", handle="mainboard-crash", verdict="submitting"),
    ):
        found.dispatcher.cache.record(each)
    Holdings(workspace, home=found.manifest.workspace.lake_home(workspace)).save(
        Held(
            alias="shadeform-h100",
            provider=PROVIDER,
            handle="m-held",
            deadline=datetime.now(UTC) + timedelta(hours=1),
            profile=HostProfile(kind="ssh"),
        )
    )
    return found


@pytest.fixture
def wire() -> Recorded:
    """Shadeform's answers: those five machines, three it cannot vouch for, and a delete."""
    listing = [
        machine("m-done", label="mainboard-done"),
        machine("m-owed", label="mainboard-owed"),
        machine("m-live", label="mainboard-live"),
        machine("m-held", label="mainboard-held"),
        machine("m-crash", label="mainboard-crash"),
        machine("m-foreign", label="mainboard-someone-else"),
        machine("m-hand", label="my-own-box"),
        machine("m-blank", label=""),
    ]
    return Recorded(
        {("GET", _LISTED): {"instances": listing}, ("POST", "/instances/m-done/delete"): {}}
    )


@pytest.fixture
def reaper(board: Board, wire: Recorded) -> Reaper:
    """The pass over that registry and that listing, asking Shadeform alone."""
    return Reaper(board, providers=[CloudBackend.find(PROVIDER)(transport=wire)])
