# A study is the identity missing above a trial: `run_id` names one config, `Study` names the
# whole sweep a fleet of trials belongs to. `StudyLedger` is its append-only event log, one JSON
# line per event, the durable record `Fleet` writes to and a report reads back.
#
# The join key with dispatch: every job a study submits carries `name="study:<study_id>"` on its
# `RunRecord` (dispatch's own free-text label field). Dispatch never parses that string and knows
# nothing about studies; a caller that wants a study's runs out of the dispatch `Cache` filters
# `RunRecord.name` for the `study:` prefix itself. This module never imports or modifies dispatch.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from .identity import study_id

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _now() -> str:
    """The current instant as an ISO-8601 string, the timestamp format every event shares."""
    return datetime.now(UTC).isoformat()


class Study(FrozenModel):
    """One experiment study: the identity a fleet of trials share.

    study_id: the content-hash identity over (experiment, config space, git sha).
    name: a human slug for logs, filenames, and `Fleet`'s dispatch label.
    experiment: the registered experiment name this study runs.
    hosts: the host aliases the study fans its trials across.
    models: the model ids the study sweeps.
    created_at: ISO-8601 creation time.
    git_sha: the short HEAD sha the study was created at.
    dirty: whether the working tree had uncommitted changes at creation.
    """

    study_id: str
    name: str
    experiment: str
    hosts: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    created_at: str
    git_sha: str
    dirty: bool = False

    @classmethod
    def create(
        cls,
        experiment: str,
        *,
        config_space: Mapping[str, object],
        git_sha: str,
        dirty: bool = False,
        hosts: tuple[str, ...] = (),
        models: tuple[str, ...] = (),
        name: str = "",
    ) -> Study:
        """A freshly identified study over `experiment`'s config space at the current git sha.

        name: an explicit human label, the derived slug (`f"{experiment}-{id[:6]}"`) when empty.
        """
        identity, slug = study_id(
            experiment=experiment, config_space=config_space, git_sha=git_sha
        )
        return cls(
            study_id=identity,
            name=name or slug,
            experiment=experiment,
            hosts=hosts,
            models=models,
            created_at=_now(),
            git_sha=git_sha,
            dirty=dirty,
        )


class StudyEvent(FrozenModel):
    """One append-only line in a study's ledger.

    at: ISO-8601 event time.
    kind: `created`, `submitted`, or `verdict`.
    handle: the dispatch handle id, for a `submitted` or `verdict` event.
    host: the host alias the job runs on, for a `submitted` event.
    state: the resolved verdict word (`ok` / `failed` / `vanished` / ...), for a `verdict` event.
    name: the owning study's human label, for a `created` event.
    """

    at: str
    kind: str
    handle: str | None = None
    host: str | None = None
    state: str | None = None
    name: str | None = None


class Progress(FrozenModel):
    """A study's trial counts, folded from a `handle -> state` mapping.

    submitted: total handles the study has ever dispatched.
    running: handles dispatched but not yet resolved to a terminal verdict.
    ok: handles that finished cleanly.
    failed: handles that ended any other terminal way (failed, vanished, unknown, timeout).
    """

    submitted: int = 0
    running: int = 0
    ok: int = 0
    failed: int = 0

    @classmethod
    def fold(cls, states: Mapping[str, str]) -> Progress:
        """Bucket each handle's state into submitted/running/ok/failed counts.

        `ok` and `submitted` (dispatched, not yet resolved) are the two recognized states;
        anything else terminal (`failed`, `vanished`, `unknown`, ...) counts as `failed`, and
        `running` is whatever is left once both are subtracted.

        states: each handle's current state word, however it was resolved.
        """
        okay = sum(1 for state in states.values() if state == "ok")
        failed = sum(1 for state in states.values() if state not in {"ok", "submitted"})
        return cls(
            submitted=len(states), running=len(states) - okay - failed, ok=okay, failed=failed
        )


class StudyLedger:
    """A study's append-only event log, one JSON line per event.

    Persists at `<root>/.mainboard/studies/<study_id>.jsonl`. Every event a caller records here
    is also, independently, whatever dispatch itself recorded for the same handle in its own
    `Cache`; the ledger exists so a study's shape (how many trials, which are still running)
    reads back without touching dispatch at all.
    """

    def __init__(self, root: Path, study_id: str) -> None:
        self.path = root / Project().out_dir / "studies" / f"{study_id}.jsonl"

    @classmethod
    def at(cls, path: Path) -> StudyLedger:
        """A ledger bound directly to an already-resolved `.jsonl` path.

        The reporting layer's `overview` walks a studies directory by file, one `<study_id>`
        per glob match, so it never has the board root `__init__` derives that path from.
        """
        ledger = cls.__new__(cls)
        ledger.path = path
        return ledger

    def append(self, event: StudyEvent) -> None:
        """Append one event line, creating the ledger's directory on first use."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as opened:
            opened.write(event.model_dump_json() + "\n")

    def created(self, study: Study) -> None:
        """Record `study`'s own creation, its human label carried for a later report to read."""
        self.append(StudyEvent(at=_now(), kind="created", name=study.name))

    def events(self) -> list[StudyEvent]:
        """Every recorded event, oldest first, or `[]` when nothing has been appended yet."""
        if not self.path.is_file():
            return []
        lines = self.path.read_text().splitlines()
        return [StudyEvent.model_validate_json(line) for line in lines if line]

    def progress(self) -> Progress:
        """Submitted/running/ok/failed counts, folded from `statuses`."""
        return Progress.fold(self.statuses())

    def statuses(self) -> dict[str, str]:
        """Each dispatched handle's current state, folded from its most recent event.

        A handle with no `verdict` event yet reads `submitted`; one that has resolved reads its
        verdict word instead. A `created` event carries no handle and folds into neither.
        """
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
