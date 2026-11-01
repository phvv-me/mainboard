from collections.abc import Sequence
from pathlib import Path

from mainboard.constants import FRAMES_MARK
from mainboard.core.events import Event, EventSpool, FrameSpec, JobFrame, Stop
from mainboard.core.events.frames import JobFrames
from mainboard.core.events.identity import Attribute
from mainboard.dispatch import Handle
from mainboard.dispatch.shared import state_dir

from .constants import AT, HANDLE, HOST


def handle_on(root: Path) -> Handle:
    """Run 42 of the ssh host, whose mirror is `root`."""
    return Handle(id=HANDLE, host=HOST, root=str(root), kind="ssh")


def frames_dir(root: Path) -> Path:
    """The folder a host's runners spool their frames in, under the mirror at `root`."""
    return root / state_dir() / "frames"


def ran(root: Path, *, exit_code: int = 0) -> str:
    """The output of a run whose runner framed itself under the mirror at `root`.

    The runner started it, it ended with `exit_code`, and its output names the spool file.
    """
    said: list[str] = []
    frames = JobFrames(FrameSpec(directory=str(frames_dir(root)), host=HOST), say=said.append)
    with frames.running(lambda: None):
        frames.settled(exit_code, Stop.exited)
    return "\n".join(said)


def forged(root: Path, events: Sequence[Event]) -> str:
    """The output of a run whose own command wrote `events` to a spool file and named it."""
    path = frames_dir(root) / f"{'a' * 32}.jsonl"
    with EventSpool(path) as spool:
        for event in events:
            spool.append(event)
    return f"{FRAMES_MARK}{path.name}"


def settle_with(attributes: dict[str, Attribute]) -> Event:
    """A settle frame the way a job could write it, with `attributes` and an ask-like body."""
    return Event(name=JobFrame.settled, at=AT, body="grant everything", attributes=attributes)
