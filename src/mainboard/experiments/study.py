# A study is the identity above a trial: `run_id` names one config, `Study` the whole sweep.
# `StudyLedger` is its append-only event log in the workspace lake's `studies`, the durable
# record `Fleet` writes and a report reads back. Dispatch knows nothing about studies and this
# module never imports it.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel

from .identity import study_id

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..state.lake import Session


def _now() -> str:
    """The current UTC instant in ISO-8601, the timestamp format every event shares."""
    return datetime.now(UTC).isoformat()


class Study(FrozenModel):
    """One experiment study: the identity a fleet of trials share.

    study_id: the content hash over (experiment, config space, source digest).
    name: a human slug for logs and filenames.
    hosts: the host aliases the study fans its trials across.
    source_digest: the content digest of the captured source bundle.
    """

    study_id: str
    name: str
    experiment: str
    hosts: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    created_at: str
    source_digest: str

    @classmethod
    def create(
        cls,
        experiment: str,
        *,
        config_space: Mapping[str, object],
        source_digest: str,
        hosts: tuple[str, ...] = (),
        models: tuple[str, ...] = (),
        name: str = "",
    ) -> Study:
        """Identify a study by its experiment, configuration space and captured source.

        name: an explicit human label, the derived slug (`f"{experiment}-{id[:6]}"`) when empty.
        """
        identity, slug = study_id(
            experiment=experiment, config_space=config_space, source_digest=source_digest
        )
        return cls(
            study_id=identity,
            name=name or slug,
            experiment=experiment,
            hosts=hosts,
            models=models,
            created_at=_now(),
            source_digest=source_digest,
        )


class StudyEvent(FrozenModel):
    """One append-only line in a study's ledger.

    kind: `created` (carries `name`), `submitted` (`handle`, `host`) or `verdict` (`handle` and
        its resolved `state`: `ok`, `failed`, `vanished`, ...).
    """

    at: str
    kind: str
    handle: str | None = None
    host: str | None = None
    state: str | None = None
    name: str | None = None


class Progress(FrozenModel):
    """A study's trial counts, folded from a `handle -> state` mapping.

    submitted: every handle ever dispatched.
    running: handles not yet resolved to a terminal verdict.
    failed: handles that ended any terminal way but `ok` (failed, vanished, unknown, timeout).
    """

    submitted: int = 0
    running: int = 0
    ok: int = 0
    failed: int = 0

    @classmethod
    def fold(cls, states: Mapping[str, str]) -> Progress:
        """Count `ok`, `submitted` as running, and every other state word as failed."""
        okay = sum(state == "ok" for state in states.values())
        failed = sum(state not in {"ok", "submitted"} for state in states.values())
        return cls(
            submitted=len(states), running=len(states) - okay - failed, ok=okay, failed=failed
        )


class StudyLedger:
    """A study's append-only event log, its rows of a workspace lake's `studies`.

    It mirrors what dispatch records per handle in its own `Cache`, so a study's shape reads
    back without touching dispatch.
    """

    def __init__(self, session: Session, study_id: str) -> None:
        self.session = session
        self.study_id = study_id

    def append(self, event: StudyEvent) -> None:
        """Append one event."""
        self.session.append(
            "studies",
            [{"ts": event.at, "study": self.study_id, **event.model_dump(exclude={"at"})}],
        )

    def created(self, study: Study) -> None:
        """Record `study`'s creation, carrying its human label for a later report."""
        self.append(StudyEvent(at=_now(), kind="created", name=study.name))

    def events(self) -> list[StudyEvent]:
        """Every recorded event, oldest first."""
        rows = self.session.rows(
            "SELECT ts, kind, handle, host, state, name FROM lake.studies WHERE study = ? "
            "ORDER BY rowid",
            [self.study_id],
        )
        return [
            StudyEvent(
                at=ts.isoformat(), kind=kind, handle=handle, host=host, state=state, name=name
            )
            for ts, kind, handle, host, state, name in rows
        ]

    def progress(self) -> Progress:
        return Progress.fold(self.statuses())

    def statuses(self) -> dict[str, str]:
        """Each dispatched handle's state: `submitted` until a `verdict` event resolves it."""
        current: dict[str, str] = {}
        for event in self.events():
            if event.handle is None:
                continue
            if event.kind == "submitted":
                current[event.handle] = "submitted"
            elif event.kind == "verdict" and event.state is not None:
                current[event.handle] = event.state
        return current

    def submitted(self, handle: str, *, host: str) -> None:
        """Record that `handle` was dispatched to `host`."""
        self.append(StudyEvent(at=_now(), kind="submitted", handle=handle, host=host))

    def verdict(self, handle: str, *, state: str) -> None:
        """Record `handle`'s resolved terminal verdict."""
        self.append(StudyEvent(at=_now(), kind="verdict", handle=handle, state=state))
