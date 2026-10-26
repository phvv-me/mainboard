# The dispatch registries in the workspace's state lake: `runs_log`, whose `runs` view is the
# current record per `(target, handle, submitted_at)` since a scheduler handle alone is not an
# identity (pueue reissues small integer ids after a daemon restart), and `host_facts`, whose
# `hosts` view is one onboarding record per alias. Both logs are append-only: a change appends the
# whole new record, a removal appends a drop, and the views read the last record per key.

from contextlib import suppress
from typing import TYPE_CHECKING

from filelock import FileLock
from patos import FrozenModel, Lifecycle
from pydantic import ValidationError
from sqlalchemy import Select, select

from ...state import schema
from ...state.lake import Lake, Session
from ..hosts.lease import Lease
from ..hosts.onboard import HostSetup
from ..jobs import vocabulary
from ..jobs.vocabulary import Request
from ..shared import now

if TYPE_CHECKING:
    from collections.abc import Mapping


class RunRecord(FrozenModel):
    """One dispatched job's provenance, the `runs_log` record payload.

    handle: the scheduler's job handle, or a provider's own run id (the dispatch-wide run id).
    kind: the target's kind at submit time, a scheduler's (`ssh` / `pbs` / `slurm` / `local`) or
        a provider's (`vast` / `hpc-ai` / `modal`), which a later pass routes on.
    script: the original shipment spelling, or the prepared host script for a direct submission
        without a shipment. Older queued records may contain a generated script.
    args: the script arguments, shell-quoted and space-joined.
    git_sha: the short HEAD sha the workspace was at when dispatched.
    dirty: 1 when the working tree had uncommitted changes, else 0.
    fetch_path: the results path to pull back, when `--fetch` was given.
    name: a human label shown instead of the internal script path; empty falls back to the
        script's basename at render time.
    node: the ledger slug this run serves, carried into its receipts; empty is still valid.
    source: the key of the pinned source tree the run executes from, so a sweep knows which host
        snapshot is in use and which is garbage. Provider creation intents retain it before
        their machine exists.
    commit: the whole commit of the tree owning the dispatched code, not always the one
        `git_sha` names: a monorepo dispatching a submodule's job records the submodule's.
        Empty when git answered nothing.
    digest: the content digest of that tree, which a run on a mirror with no history seals
        itself against. Empty when `commit` is.
    state: the last resolved scheduler outcome, memoized so a finished job is read from the
        cache instead of re-probed over ssh. `None` means never resolved; a terminal verdict
        here is trusted without touching the host.
    verdict: one of the shared `vocabulary` verdicts.
    reported: the verdict a durable monitor last surfaced, the cursor that keeps a periodic sweep
        reporting only jobs newly terminal. `None` means never reported, so the first sweep
        that finds it terminal announces it.
    evidence: delivery status, separate from the computational verdict. `copied` means hashes
        were verified but release is pending; `verified` means settlement finished. An
        `unverified` correction preserves the original verdict while qualifying its evidence.
    creation: the unique provider-request label, retained when its handle replaces the intent.
    request: the original request for held jobs, or the resolved environment/resources for a
        provider creation. Only held requests are retried automatically; a lost create reply
        requires reconciliation with the provider.
    lease: accepted rental quote and release deadline, retained before provider creation.
    project: the workspace project the run was dispatched from (the directory under
        `research/` or `packages/` the command ran in, or `MB_PROJECT`), empty for runs
        recorded before it was kept.
    reason: why the row is in its state when the state cannot say it alone, i.e. the refusal a
        target's quota answered a held dispatch with. Empty for every run that was taken.
    """

    handle: str
    target: str
    kind: str
    script: str
    args: str
    git_sha: str = ""
    dirty: int | None = None
    submitted_at: str
    fetch_path: str | None = None
    name: str = ""
    node: str = ""
    source: str = ""
    commit: str = ""
    digest: str = ""
    state: str | None = None
    exit_code: int | None = None
    verdict: str | None = None
    reported: str | None = None
    evidence: str = ""
    creation: str = ""
    request: Request | None = None
    lease: Lease | None = None
    reason: str = ""
    project: str = ""

    @property
    def label(self) -> str:
        """The name `jobs` prints for this run: its label, else the script it was submitted as."""
        return self.name or self.script


class Log:
    """The registry's append-only logs in one lake session, each write that reads first under
    the registry lock, the way the one SQLite statement it replaces was atomic."""

    @property
    def _registry(self) -> FileLock:
        """The lock every read-then-append holds, so two processes cannot both win a race."""
        return self._lock("registry.lock")

    @staticmethod
    def _newest() -> Select:
        """Every current run's record, the newest dispatch first."""
        return select(schema.runs.c.record).order_by(schema.runs.c.submitted_at.desc())

    @staticmethod
    def _reaches(current: str | None, verdict: str | int | None) -> bool:
        """Whether a run recorded at `current` may be recorded at `verdict`.

        It may when never resolved, for the same verdict again, or for a move `VERDICTS` declares.
        """
        if current is None or verdict == current:
            return True
        with suppress(ValueError):
            moved = Lifecycle(vocabulary.Word.transitions(), vocabulary.Word(current))
            return moved.allowed(vocabulary.Word(str(verdict)))
        return False

    @staticmethod
    def _row(run: RunRecord, *, dropped: bool = False) -> dict[str, object]:
        """The `runs_log` row recording `run`, or its removal when `dropped`."""
        return {
            "ts": now(),
            "target": run.target,
            "handle": run.handle,
            "submitted_at": run.submitted_at,
            "record": run.model_dump_json(),
            "dropped": dropped,
        }

    @staticmethod
    def _unregistered(run: RunRecord) -> LookupError:
        """No record of `run`, for a change that needs one."""
        return LookupError(f"no registered run {run.handle!r} on {run.target!r}")

    def __init__(self, lake: Lake, session: Session) -> None:
        self.lake = lake
        self.session = session

    def _at(self, run: RunRecord) -> RunRecord | None:
        """The current record of `run`'s identity, None when it was never recorded or dropped."""
        identity = (
            schema.runs.c.target == run.target,
            schema.runs.c.handle == run.handle,
            schema.runs.c.submitted_at == run.submitted_at,
        )
        return next(iter(self._records(select(schema.runs.c.record).where(*identity))), None)

    def _lock(self, name: str) -> FileLock:
        """The process-reentrant file lock `name` under the lake's run directory."""
        path = self.lake.out / "run" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(path, is_singleton=True)

    def _on(self, target: str | None) -> list[RunRecord]:
        """Every current run, on `target` when given, newest first."""
        newest = self._newest()
        return self._records(newest.where(schema.runs.c.target == target) if target else newest)

    def _records(self, statement: Select) -> list[RunRecord]:
        """The run records `statement` selects, in its order."""
        rows = self.session.rows(statement)
        return [RunRecord.model_validate_json(record, extra="ignore") for (record,) in rows]

    def _transition(
        self,
        run: RunRecord,
        expected: str,
        update: Mapping[str, object],
        *,
        missing: Exception,
    ) -> RunRecord:
        """Move `run` on only while its current verdict is `expected`, raising `missing` else."""
        with self._registry:
            current = self._at(run)
            if current is None or current.verdict != expected:
                raise missing
            moved = current.model_copy(update=dict(update))
            self.session.append(schema.runs_log, [self._row(moved)])
            return moved

    def close(self) -> None:
        """Detach the lake now rather than whenever the collector reclaims this registry."""
        self.session.close()


class Creations(Log):
    """Provider creation intents: a request recorded before its machine exists, reserved once,
    then bound to the provider's handle, left prepared, or reopened once declined."""

    def bind(self, run: RunRecord, handle: str) -> RunRecord:
        """Replace the intent identity in one commit, preserving its provenance and time."""
        with self._registry:
            current = self._at(run)
            if current is None or current.verdict != vocabulary.Word.SUBMITTING:
                raise LookupError(f"no submitting creation {run.creation!r} on {run.target!r}")
            bound = current.model_copy(
                update={
                    "handle": handle,
                    "state": vocabulary.Word.QUEUED,
                    "verdict": vocabulary.Word.QUEUED,
                }
            )
            self.session.append(
                schema.runs_log, [self._row(current, dropped=True), self._row(bound)]
            )
            return bound

    def named(self, label: str, *, target: str) -> RunRecord:
        """Read the same request before or after its provider handle replaced the intent."""
        for run in self._on(target):
            if run.creation == label:
                return run
        raise LookupError(f"no creation {label!r} on {target!r}")

    def prepared(self, run: RunRecord, verdict: str, *, lease: Lease | None = None) -> RunRecord:
        """Claim preparation once, so cancellation and creation cannot both win."""
        if not self._reaches(vocabulary.Word.PREPARED, verdict):
            raise ValueError(f"invalid prepared transition to {verdict!r}")
        return self._transition(
            run,
            vocabulary.Word.PREPARED,
            {
                "state": verdict,
                "verdict": verdict,
                "reported": verdict if verdict in vocabulary.Word.settled() else None,
                "lease": lease,
            },
            missing=ValueError(
                f"creation {run.creation!r} is no longer prepared; do not resend it"
            ),
        )

    def reopen(self, run: RunRecord) -> RunRecord:
        """Return a submitting creation to prepared, once the provider has declined it.

        The refusal proves nothing was allocated, so the request may be sent again or left to
        `interrupted` to close.
        """
        return self._transition(
            run,
            vocabulary.Word.SUBMITTING,
            {
                "state": vocabulary.Word.PREPARED,
                "verdict": vocabulary.Word.PREPARED,
                "lease": None,
            },
            missing=ValueError(f"creation {run.creation!r} is not submitting; nothing to reopen"),
        )

    def reserve(self, run: RunRecord) -> None:
        """Reserve a creation once; unresolved identical requests cannot allocate twice."""
        with self._registry:
            pending = (vocabulary.Word.PREPARED, vocabulary.Word.SUBMITTING)
            if any(
                other.verdict in pending
                and other.script == run.script
                and other.digest == run.digest
                for other in self._on(run.target)
            ):
                raise ValueError(
                    f"an unresolved creation already exists for {run.name or run.script!r} on "
                    f"{run.target}; inspect its job record and provider label before retrying"
                )
            self.session.append(schema.runs_log, [self._row(run)])


class Hosts(Log):
    """Every host's onboarding record, `host_facts` read through its `hosts` view."""

    @staticmethod
    def _current(facts: str) -> HostSetup | None:
        """A recorded setup, None when an older mainboard wrote it and `setup` must redo it."""
        with suppress(ValidationError):
            return HostSetup.model_validate_json(facts, extra="ignore")
        return None

    @staticmethod
    def _host(setup: HostSetup, *, probed_at: str | None) -> dict[str, object]:
        """The `host_facts` row recording `setup`, ordered among its alias's by `probed_at`."""
        return {
            "ts": now(),
            "alias": setup.host,
            "probed_at": probed_at,
            "facts": setup.model_dump_json(),
            "dropped": False,
        }

    def all(self) -> list[HostSetup]:
        """Every onboarded host, most recently set up first."""
        rows = self.session.rows(
            select(schema.hosts.c.facts).order_by(schema.hosts.c.probed_at.desc())
        )
        return [setup for (facts,) in rows if (setup := Hosts._current(facts)) is not None]

    def drop(self, alias: str) -> None:
        """Forget `alias`'s onboarding, for a machine that no longer exists to be set up."""
        with self._registry:
            self.session.append(
                schema.host_facts, [{"ts": now(), "alias": alias, "facts": "{}", "dropped": True}]
            )

    def recorded(self, alias: str) -> HostSetup:
        """`alias`'s recorded onboarding, raising when the host was never set up."""
        rows = self.session.rows(select(schema.hosts.c.facts).where(schema.hosts.c.alias == alias))
        if not rows:
            raise LookupError(f"host {alias!r} has never been set up; run `mb host setup {alias}`")
        try:
            return HostSetup.model_validate_json(rows[0][0], extra="ignore")
        except ValidationError:
            raise LookupError(
                f"host {alias!r} was set up by an older release; run `mb host setup {alias}`"
            ) from None

    def save(self, setup: HostSetup) -> HostSetup:
        """Stamp `setup` with the current time and record it as `setup.host`'s onboarding.

        The store owns the timestamp, so two records of the same host can always be ordered.
        """
        stamped = setup.model_copy(update={"onboarded_at": now()})
        with self._registry:
            self.session.append(
                schema.host_facts, [Hosts._host(stamped, probed_at=stamped.onboarded_at)]
            )
        return stamped

    def synced(self, alias: str) -> None:
        """Stamp `alias`'s mirror watermark, which a later transfer set measures its delta from.

        A host never onboarded has no mirror this store has seen, so nothing to stamp.
        """
        with self._registry:
            try:
                setup = self.recorded(alias)
            except LookupError:
                return
            synced = setup.model_copy(update={"synced_at": now()})
            self.session.append(
                schema.host_facts, [Hosts._host(synced, probed_at=setup.onboarded_at)]
            )
