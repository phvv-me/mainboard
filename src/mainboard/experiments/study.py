# A study is the identity above a trial: `run_id` names one config, `Study` the whole sweep.
# `StudyLedger` is its append-only JSON-lines event log, the durable record `Fleet` writes and a
# report reads back. Dispatch knows nothing about studies and this module never imports it.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from .identity import study_id

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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
    """A study's append-only event log at `<root>/.mainboard/studies/<study_id>.jsonl`.

    It mirrors what dispatch records per handle in its own `Cache`, so a study's shape reads
    back without touching dispatch.
    """

    def __init__(self, root: Path, study_id: str) -> None:
        self.path = root / Project().out_dir / "studies" / f"{study_id}.jsonl"

    @classmethod
    def at(cls, path: Path) -> StudyLedger:
        """A ledger bound to an already-resolved `.jsonl` path, for a caller without the root."""
        ledger = cls.__new__(cls)
        ledger.path = path
        return ledger

    def append(self, event: StudyEvent) -> None:
        """Append one event line, creating the ledger's directory on first use."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as opened:
            opened.write(event.model_dump_json() + "\n")

    def created(self, study: Study) -> None:
        """Record `study`'s creation, carrying its human label for a later report."""
        self.append(StudyEvent(at=_now(), kind="created", name=study.name))

    def events(self) -> list[StudyEvent]:
        """Every recorded event, oldest first."""
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [StudyEvent.model_validate_json(line) for line in lines if line]

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
