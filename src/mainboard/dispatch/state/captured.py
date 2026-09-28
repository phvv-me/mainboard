# What a settle brings home from a run before its machine may go away: the run's transcript, kept
# line by line in `log_lines`, and the trial receipts it printed, kept verbatim in `receipts`, both
# under the stream the run belongs to. A transcript captured again is appended whole under a new
# stamp and the newest stamp is the transcript; a receipt already kept for its stream is never
# kept twice, since its exact text is its identity.

import json
from typing import TYPE_CHECKING

from ...state.lake import ALIAS
from ..shared import now

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...state.lake import Session


class Captured:
    """One workspace's captured transcripts and trial receipts, in its state lake."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def keep_transcript(self, stream: str, handle: str, transcript: str) -> None:
        """Keep `transcript` as `handle`'s output, unless it already is the one kept."""
        if self.transcript(stream, handle) != transcript:
            stamp = now()
            self.session.append(
                "log_lines",
                [
                    {
                        "ts": stamp,
                        "batch": stream,
                        "file": _file(stream, handle),
                        "n": number,
                        "line": line,
                        "lossy": False,
                    }
                    for number, line in enumerate(transcript.split("\n"), 1)
                ],
            )

    def keep_receipts(self, stream: str, receipts: Iterable[str]) -> None:
        """Keep each of `receipts` that `stream` does not hold yet, in order."""
        held = self.receipts(stream)
        known = set(held)
        fresh = [line for line in dict.fromkeys(receipts) if line not in known]
        self.session.append(
            "receipts",
            [_receipt(stream, number, line) for number, line in enumerate(fresh, len(held) + 1)],
        )

    def receipts(self, stream: str) -> list[str]:
        """Every receipt line kept for `stream`, in the order they were kept."""
        rows = self.session.rows(
            f"SELECT line FROM {ALIAS}.receipts WHERE batch = ? ORDER BY rowid", [stream]
        )
        return [line for (line,) in rows]

    def transcript(self, stream: str, handle: str, *, last: int = 0) -> str | None:
        """`handle`'s newest captured output, None when none was captured.

        last: only that many final lines, 0 for all of them.
        """
        file = _file(stream, handle)
        rows = self.session.rows(
            f"SELECT line FROM {ALIAS}.log_lines WHERE file = ? AND ts = "
            f"(SELECT max(ts) FROM {ALIAS}.log_lines WHERE file = ?) "
            "QUALIFY n > max(n) OVER () - ? OR ? = 0 ORDER BY n",
            [file, file, last, last],
        )
        return "\n".join(line for (line,) in rows) if rows else None


def _file(stream: str, handle: str) -> str:
    """The name one run's transcript is kept under, the path it had under the batches folder."""
    return f"{stream}/{handle}.log"


def _receipt(stream: str, number: int, line: str) -> dict[str, object]:
    """One kept receipt line with the fields a query filters on, empty for a line that is not
    a receipt envelope."""
    try:
        envelope = json.loads(line)
    except json.JSONDecodeError:
        envelope = None
    found = envelope.get("trial_receipt") if isinstance(envelope, dict) else None
    receipt = found if isinstance(found, dict) else {}
    return {
        "ts": now(),
        "batch": stream,
        "n": number,
        "run": receipt.get("run"),
        "trial": receipt.get("trial"),
        "verdict": receipt.get("verdict"),
        "host": receipt.get("host"),
        "line": line,
    }
