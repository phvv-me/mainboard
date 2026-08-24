from pathlib import Path
from typing import TYPE_CHECKING

from mainboard import Board
from mainboard.batch import BatchSpec, Event, Receipts
from mainboard.manifest import HostProfile

if TYPE_CHECKING:
    from mainboard.batch.receipts import Bus


class Recorder:
    """A `Bus` that keeps its events in memory, so a test reads what was published directly."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def replay(self) -> list[Event]:
        return list(self.events)


def declaring(board: Board, alias: str, profile: HostProfile) -> None:
    """Declare one more host on `board`'s manifest, for a target the fixture never carried.

    The manifest and the resolver built from it are the board's own shared caches, so replacing
    the first means dropping the second rather than leaving a plan resolved against the old one.
    """
    hosts = {**board.manifest.hosts, alias: profile}
    board.shared["manifest"] = board.manifest.model_copy(update={"hosts": hosts})
    board.shared.pop("resolver", None)


def spec(*jobs: dict[str, object], name: str = "smoke") -> BatchSpec:
    """A batch spec over already-written job tables."""
    return BatchSpec.of(name, jobs)


def published(bus: Bus, topic: str) -> list[Event]:
    """Every event of `topic` on `bus`, oldest first."""
    return [event for event in bus.replay() if event.topic == topic]


def receipts(path: Path) -> Receipts:
    """A file-backed bus under `path`, the transport a real batch writes through."""
    return Receipts(path / "events.ndjson")
