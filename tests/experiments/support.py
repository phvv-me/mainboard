from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mainboard.dispatch import Handle
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import POLL_SECONDS

if TYPE_CHECKING:
    from typing import Unpack

    from mainboard.dispatch import Verdict
    from mainboard.experiments.fleet import ResourceOverrides


def make_run(
    name: str, *, handle: str, submitted_at: str = "t0", verdict: str | None = None
) -> RunRecord:
    """A dispatch `RunRecord` labeled `name`, resolved to `verdict` when one is given."""
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
    """A `Dispatcher`-like stub with a real `cache`, its `await_many` answered from `verdicts`."""

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
    """The real dispatch run registry in memory, sparing each test the WAL journal's slow fsync."""
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
