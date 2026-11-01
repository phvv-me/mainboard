"""One trial's event stream: the spool it fills and the logger sink that feeds it."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from structlog.contextvars import bind_contextvars, reset_contextvars

from ...constants import SINK
from ...enums import Kind
from ...log import EventDict, logger, sinks
from ...observe.frames import Frame
from ...observe.spool import Spool
from ..enums import Outcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from ..session import Trial

# What every event carries that is the logger's envelope rather than the caller's metadata.
_ENVELOPE = frozenset(("event", "level", "timestamp", "exc_info", "stack_info"))


class TrialStream:
    """The frames of one trial, shared by every `Log` bound from it.

    Every line logged while the trial runs is kept with it, whichever library logged it; the
    trial's own methods name the sink explicitly, so they are kept from any thread. The stream
    opens its spool beside the node's store, announces the trial, and `close` ends it.

    context: the provenance announced by the opening frame.
    node: the claim that holds this trial's evidence.
    key: the digest of the trial's node id, naming its spool within the run.
    """

    def __init__(
        self, trial: Trial, context: Mapping[str, JsonValue], *, node: str, key: str
    ) -> None:
        session = trial.session
        self.trial = trial
        self.identity = f"{session.run}/{key}"
        directory = session.declared.universe.dataset(node).root.parent / "artifacts"
        self.spool = Spool(directory / session.run / key, "events")
        trial.artifacts["events"] = self.spool.dir.relative_to(session.store(node).root).as_posix()
        sinks[self.identity] = self._message
        self.bound = bind_contextvars(**{SINK: self.identity})
        self.logger = logger.bind(**{SINK: self.identity})
        self.event("started", context, metadata={}, kind=Kind.started)

    def _finish(self, *, passed: bool) -> None:
        """Record the end of the stream, refusing a settled trial whose artifacts moved."""
        self.event(
            "ended",
            {"passed": passed, "verdict": self.trial.settled},
            metadata={},
            kind=Kind.ended,
        )
        if self.trial.settled and self.trial.artifacts != self.trial.recorded_artifacts:
            reason = (
                "Artifacts changed after the last receipt was settled; "
                "settle only after output checks and cleanup evidence are attached"
            )
            self.trial.record("", reason=reason, measured={}, outcome=Outcome.FAILED)
            raise RuntimeError(reason)

    def _message(self, event: EventDict) -> None:
        self.event(
            "message",
            {
                "text": str(event["event"]),
                "level": event["level"],
                "metadata": {key: value for key, value in event.items() if key not in _ENVELOPE},
            },
            metadata={},
        )

    def close(self, *, passed: bool) -> None:
        """Flush the stream and remove only this trial's sink."""
        try:
            self._finish(passed=passed)
        finally:
            sinks.pop(self.identity, None)
            reset_contextvars(**self.bound)
            self.spool.close()

    def event(
        self,
        topic: str,
        payload: Mapping[str, JsonValue],
        *,
        metadata: dict[str, JsonValue],
        kind: Kind = Kind.line,
    ) -> None:
        """Append one frame of this trial, carrying the diagnostic context its caller bound."""
        self.spool.append(
            Frame(
                job=self.identity,
                kind=kind,
                at=datetime.now(UTC),
                payload={
                    "topic": topic,
                    "trial": self.trial.item.nodeid,
                    "metadata": metadata,
                    "data": dict(payload),
                },
            )
        )
