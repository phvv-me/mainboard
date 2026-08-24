from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mainboard.dispatch import Handle
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import POLL_SECONDS
from mainboard.experiments import Study

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Unpack

    from mainboard.dispatch import Verdict
    from mainboard.experiments.fleet import ResourceOverrides


def make_run(
    name: str, *, handle: str, submitted_at: str = "t0", verdict: str | None = None
) -> RunRecord:
    """A dispatch `RunRecord` labeled `name`, resolved to `verdict` when one is given.

    name: the free-text dispatch label a study join reads back.
    handle: the scheduler handle the row is keyed by.
    submitted_at: the ISO-8601 dispatch time the registry orders rows by.
    verdict: the terminal outcome dispatch resolved, `None` while it has none.
    """
    return RunRecord(
        handle=handle,
        target="gold",
        kind="pbs",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at=submitted_at,
        name=name,
        verdict=verdict,
    )


@dataclass
class FakeJob:
    """A `Job`-like stub, since `Fleet` only ever reads `.handle` off what `submit` returns."""

    handle: Handle


class FakeBoundBoard:
    """A `Board.on(host)`-like stub recording every `submit` call onto its parent `FakeBoard`."""

    def __init__(self, board: FakeBoard, host: str) -> None:
        self.board = board
        self.host = host

    def submit(
        self, command: str, *, name: str = "", **overrides: Unpack[ResourceOverrides]
    ) -> FakeJob:
        self.board.calls.append((self.host, command, name, overrides))
        self.board.counter += 1
        handle = Handle(id=str(self.board.counter), host=self.host, root="/work/x", kind="pbs")
        return FakeJob(handle)


class FakeDispatcher:
    """A `Dispatcher`-like stub whose `await_many` resolves from a pre-seeded verdict map.

    Carries a real `cache`, exactly `Dispatcher.cache`'s shape, since a study's progress is
    read by joining its ledger against dispatch's own resolved verdicts.
    """

    def __init__(self, cache: Cache) -> None:
        self.cache = cache
        self.verdicts: dict[Handle, Verdict] = {}
        self.awaited: list[Handle] = []

    def await_many(
        self, handles: Sequence[Handle], *, interval: float = POLL_SECONDS
    ) -> dict[Handle, Verdict]:
        self.awaited.extend(handles)
        return {handle: self.verdicts[handle] for handle in handles}


def dispatch_cache() -> Cache:
    """The dispatch run registry in a private in-memory database.

    Every table the file-backed registry creates is created here too, so a join reads and
    writes exactly what production does without paying the WAL journal's fsync per test, which
    is what made every study test in this slice slow.
    """
    return Cache(Path(":memory:"))


class FakeBoard:
    """A `Board`-like stub carrying a real root and a recorded `on`, never an ssh or a process."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, str, str, ResourceOverrides]] = []
        self.counter = 0
        self.dispatcher = FakeDispatcher(dispatch_cache())

    def on(self, host: str) -> FakeBoundBoard:
        return FakeBoundBoard(self, host)


@pytest.fixture
def cache() -> Cache:
    """A dispatch run registry a study's report is joined against."""
    return dispatch_cache()


@pytest.fixture
def board(tmp_path: Path) -> FakeBoard:
    """A board rooted at the test's own tmp path, its dispatch registry held in memory."""
    return FakeBoard(tmp_path)


@pytest.fixture(scope="session")
def study() -> Study:
    """One study identity every fleet and report test dispatches its trials under.

    Session-scoped because a `Study` is frozen and every test that uses it writes its ledger
    under its own tmp path, so nothing is shared but the identity itself.
    """
    return Study.create("joint-search", config_space={"bits": [1, 2]}, git_sha="abc123")
