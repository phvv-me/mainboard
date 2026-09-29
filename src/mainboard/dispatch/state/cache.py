# The dispatch registries in the workspace's state lake: `runs_log`, whose `runs` view is the
# current record per `(target, handle, submitted_at)` since a scheduler handle alone is not an
# identity (pueue reissues small integer ids after a daemon restart), and `host_facts`, whose
# `hosts` view is one onboarding record per alias. Both logs are append-only: a change appends the
# whole new record, a removal appends a drop, and the views read the last record per key.

import weakref
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

from filelock import FileLock
from patos import FrozenModel
from pydantic import ValidationError

from ...state.lake import ALIAS, Lake, Session
from .. import vocabulary
from ..lease import Lease
from ..onboard import HostSetup
from ..shared import now, workspace
from ..vocabulary import Request

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The terminal verdicts as SQL literals, fixed vocabulary words, for the lake to filter on.
_TERMINAL = ", ".join(f"'{verdict}'" for verdict in sorted(vocabulary.TERMINAL))


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


class Cache:
    """The dispatch registries, `runs` and `hosts`, in one workspace's state lake.

    The lake is opened on the first statement, not at construction, so a board that never touches
    the registry (a host installing an environment) never creates or attaches one, and the one
    session is kept for this cache's life rather than attached per statement. Every write that
    reads first (a reservation, a compare-and-set transition) holds the registry lock across its
    read and append, the way the one SQLite statement it replaces was atomic.
    """

    def __init__(self, lake: Lake | None = None) -> None:
        self.lake = lake or Lake.at(workspace())
        self.session = self.lake.session()

    @classmethod
    def private(cls) -> Cache:
        """A registry in a lake of its own under a temporary directory, removed with the cache."""
        root = Path(mkdtemp(prefix="mb-registry-"))
        cache = cls(Lake.at(root))
        weakref.finalize(cache, _discard, cache.session, root)
        return cache

    @property
    def path(self) -> Path:
        """The catalog this registry is kept in."""
        return self.lake.catalog

    @property
    def settlement(self) -> FileLock:
        """The shared launch/cancel/settle lock; remote lifecycle requires durable state."""
        return self._lock("settlement.lock")

    @property
    def _registry(self) -> FileLock:
        """The lock every read-then-append holds, so two processes cannot both win a race."""
        return self._lock("registry.lock")

    def close(self) -> None:
        """Detach the lake now rather than whenever the collector reclaims this cache."""
        self.session.close()

    def forget(self, run: RunRecord) -> None:
        """Drop one run's record entirely, the only thing that ever leaves `runs`.

        A quota-held dispatch is a record about a request, replaced by the real run once it goes
        through; keeping both would read thirteen jobs as fourteen, so the placeholder goes
        rather than being settled into a verdict it never had.
        """
        with self._registry:
            self._append("runs_log", [_row(run, dropped=True)])

    def delivery(self, run: RunRecord, status: str) -> RunRecord:
        """Advance evidence without replacing another process's computation or report fields."""
        return self._change(run, evidence=status)

    def drop_host(self, alias: str) -> None:
        """Forget `alias`'s onboarding, for a machine that no longer exists to be set up."""
        with self._registry:
            self._append(
                "host_facts", [{"ts": now(), "alias": alias, "facts": "{}", "dropped": True}]
            )

    def host(self, alias: str) -> HostSetup:
        """`alias`'s recorded onboarding, raising when the host was never set up."""
        rows = self._rows(f"SELECT facts FROM {ALIAS}.hosts WHERE alias = ?", (alias,))
        if not rows:
            raise LookupError(f"host {alias!r} has never been set up; run `mb host setup {alias}`")
        try:
            return HostSetup.model_validate_json(rows[0][0], extra="ignore")
        except ValidationError:
            raise LookupError(
                f"host {alias!r} was set up by an older release; run `mb host setup {alias}`"
            ) from None

    def hosts(self) -> list[HostSetup]:
        """Every onboarded host, most recently set up first."""
        rows = self._rows(f"SELECT facts FROM {ALIAS}.hosts ORDER BY probed_at DESC")
        return [setup for (facts,) in rows if (setup := _current(facts)) is not None]

    def mark_synced(self, alias: str) -> None:
        """Stamp `alias`'s mirror watermark, which a later transfer set measures its delta from.

        A host never onboarded has no mirror this store has seen, so nothing to stamp.
        """
        with self._registry:
            try:
                setup = self.host(alias)
            except LookupError:
                return
            synced = setup.model_copy(update={"synced_at": now()})
            self._append("host_facts", [_host(synced, probed_at=setup.onboarded_at)])

    def live(self, project: str = "") -> list[RunRecord]:
        """Every run without a terminal verdict, newest first, never truncated by a limit.

        A listing that hides half a dispatched wave sends an operator to `qstat` by hand.

        project: only the runs dispatched from this project; every run when empty.
        """
        return self._matching(f"coalesce(verdict, '') NOT IN ({_TERMINAL})", project=project)

    def _matching(
        self, where: str, *, project: str = "", limit: int | None = None
    ) -> list[RunRecord]:
        """The current runs `where` holds, newest first, filtered in the lake rather than here.

        A registry of a hundred thousand runs cost 0.7 s a listing when every record was parsed
        to keep a few; the view's columns let the lake answer only those.
        """
        clause = f"({where})" + (" AND project = ?" if project else "")
        bound = "" if limit is None else f" LIMIT {int(limit)}"
        rows = self._rows(
            f"SELECT record FROM {ALIAS}.runs WHERE {clause} ORDER BY submitted_at DESC{bound}",
            (project,) if project else (),
        )
        return [RunRecord.model_validate_json(record, extra="ignore") for (record,) in rows]

    def recent(self, limit: int | None = 20) -> list[RunRecord]:
        """Dispatched runs, newest first; None retains all historical declarations."""
        bound = "" if limit is None else f" LIMIT {int(limit)}"
        rows = self._rows(f"SELECT record FROM {ALIAS}.runs ORDER BY submitted_at DESC{bound}")
        return [RunRecord.model_validate_json(record, extra="ignore") for (record,) in rows]

    def since(self, stamp: str) -> list[RunRecord]:
        """Every run dispatched at or after the ISO instant `stamp`, newest first."""
        rows = self._rows(
            f"SELECT record FROM {ALIAS}.runs WHERE submitted_at >= ? ORDER BY submitted_at DESC",
            (stamp,),
        )
        return [RunRecord.model_validate_json(record, extra="ignore") for (record,) in rows]

    def record(self, run: RunRecord) -> None:
        """Record a dispatched run, replacing whatever its identity recorded before."""
        with self._registry:
            self._append("runs_log", [_row(run)])

    def reserve(self, run: RunRecord) -> None:
        """Reserve a creation once; unresolved identical requests cannot allocate twice."""
        with self._registry:
            pending = (vocabulary.PREPARED, vocabulary.SUBMITTING)
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
            self._append("runs_log", [_row(run)])

    def bind(self, run: RunRecord, handle: str) -> RunRecord:
        """Replace the intent identity in one commit, preserving its provenance and time."""
        with self._registry:
            current = self._at(run)
            if current is None or current.verdict != vocabulary.SUBMITTING:
                raise LookupError(f"no submitting creation {run.creation!r} on {run.target!r}")
            bound = current.model_copy(
                update={"handle": handle, "state": vocabulary.QUEUED, "verdict": vocabulary.QUEUED}
            )
            self._append("runs_log", [_row(current, dropped=True), _row(bound)])
            return bound

    def leave_prepared(
        self, run: RunRecord, verdict: str, *, lease: Lease | None = None
    ) -> RunRecord:
        """Claim preparation once, so cancellation and creation cannot both win."""
        if verdict not in vocabulary.VERDICTS[vocabulary.PREPARED]:
            raise ValueError(f"invalid prepared transition to {verdict!r}")
        return self._transition(
            run,
            vocabulary.PREPARED,
            {
                "state": verdict,
                "verdict": verdict,
                "reported": verdict if verdict in vocabulary.TERMINAL else None,
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
            vocabulary.SUBMITTING,
            {"state": vocabulary.PREPARED, "verdict": vocabulary.PREPARED, "lease": None},
            missing=ValueError(f"creation {run.creation!r} is not submitting; nothing to reopen"),
        )

    def creation(self, label: str, target: str) -> RunRecord:
        """Read the same request before or after its provider handle replaced the intent."""
        for run in self._on(target):
            if run.creation == label:
                return run
        raise LookupError(f"no creation {label!r} on {target!r}")

    def relet(self, run: RunRecord, lease: Lease) -> RunRecord:
        """Replace `run`'s rental lease, the deadline a sweep releases the machine at."""
        with self._registry:
            current = self._at(run)
            if current is None:
                raise _unregistered(run)
            relet = current.model_copy(update={"lease": lease})
            self._append("runs_log", [_row(relet)])
            return relet

    def report(self, run: RunRecord, verdict: str) -> None:
        """Record the verdict a durable monitor last surfaced for `run`."""
        self._change(run, reported=verdict)

    def resolve(
        self, run: RunRecord, state: str | None, exit_code: int | None, verdict: str
    ) -> RunRecord:
        """Memoize a run's resolved scheduler outcome, touching only computation fields.

        A dispatcher may advance evidence while a monitor holds an older record; neither writer
        may replace the other's fields from that stale copy.
        """
        return self._change(run, state=state, exit_code=exit_code, verdict=verdict)

    def run(self, handle: str, target: str | None = None) -> RunRecord:
        """The newest run dispatched as `handle` (older records are history), narrowed to `target`.

        `handle` may also be the name `jobs` prints for a run, its label or else its script, the
        spelling an operator copies off that table; a handle wins where both match. A name
        answers its newest run, since the attempts of one run (`job submit --resume`) share it;
        a handle recorded on several targets raises with the candidates rather than guessing.
        """
        every = self._on(target)
        runs = [run for run in every if run.handle == handle]
        if not runs:
            named = [run for run in every if run.label == handle]
            runs = sorted(named, key=lambda run: run.submitted_at)[-1:]
        if not runs:
            where = f" on {target!r}" if target else ""
            raise LookupError(f"no recorded run {handle!r}{where}")
        candidates = sorted({f"{run.target} {run.handle}" for run in runs})
        if len(candidates) > 1:
            raise LookupError(
                f"{handle!r} names runs {', '.join(candidates)}; pass one handle and its host"
            )
        return runs[0]

    def attempts(self, name: str, target: str) -> int:
        """How many runs named `name` were dispatched to `target`: every attempt of that run."""
        rows = self._rows(
            f"SELECT count(*) FROM {ALIAS}.runs WHERE target = ? AND name = ?", (target, name)
        )
        return int(rows[0][0])

    def save_host(self, setup: HostSetup) -> HostSetup:
        """Stamp `setup` with the current time and record it as `setup.host`'s onboarding.

        The store owns the timestamp, so two records of the same host can always be ordered.
        """
        stamped = setup.model_copy(update={"onboarded_at": now()})
        with self._registry:
            self._append("host_facts", [_host(stamped, probed_at=stamped.onboarded_at)])
        return stamped

    def settled(self, limit: int, project: str = "") -> list[RunRecord]:
        """The `limit` most recently dispatched runs whose verdict is terminal, newest first.

        project: only the runs dispatched from this project; every run when empty.
        """
        return self._matching(f"verdict IN ({_TERMINAL})", project=project, limit=limit)

    def total(self) -> int:
        """How many runs this registry holds, the count a truncated listing measures against."""
        return int(self._rows(f"SELECT count(*) FROM {ALIAS}.runs")[0][0])

    def tracked(self) -> list[RunRecord]:
        """Every run a durable sweep still owes an outcome for, newest first.

        A run leaves only once its terminal verdict has been reported, so a sweep never announces
        a settled run twice nor drops one whose dispatching agent died before recording it.
        """
        return self._matching(
            f"coalesce(verdict, '') NOT IN ({_TERMINAL}) "
            "OR (record->>'reported') IS DISTINCT FROM verdict"
        )

    def _change(self, run: RunRecord, **fields: str | int | None) -> RunRecord:
        """Update one registered identity atomically and answer its complete current record.

        A terminal verdict is never replaced by a different one: a late or stale report leaves the
        whole record as it stands rather than half of it.
        """
        with self._registry:
            current = self._at(run)
            if current is None:
                raise _unregistered(run)
            verdict = fields.get("verdict")
            if (
                "verdict" in fields
                and current.verdict in vocabulary.TERMINAL
                and current.verdict != verdict
            ):
                return current
            changed = current.model_copy(update=fields)
            self._append("runs_log", [_row(changed)])
            return changed

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
            self._append("runs_log", [_row(moved)])
            return moved

    def _at(self, run: RunRecord) -> RunRecord | None:
        """The current record of `run`'s identity, None when it was never recorded or dropped."""
        rows = self._rows(
            f"SELECT record FROM {ALIAS}.runs "
            "WHERE target = ? AND handle = ? AND submitted_at = ?",
            (run.target, run.handle, run.submitted_at),
        )
        return RunRecord.model_validate_json(rows[0][0], extra="ignore") if rows else None

    def _on(self, target: str | None) -> list[RunRecord]:
        """Every current run, on `target` when given, newest first."""
        rows = self._rows(
            f"SELECT record FROM {ALIAS}.runs WHERE target = coalesce(?, target) "
            "ORDER BY submitted_at DESC",
            (target,),
        )
        return [RunRecord.model_validate_json(record, extra="ignore") for (record,) in rows]

    def _lock(self, name: str) -> FileLock:
        """The process-reentrant file lock `name` under the lake's run directory."""
        path = self.lake.out / "run" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(path, is_singleton=True)

    def _rows(self, sql: str, parameters: Sequence[object] = ()) -> list[tuple[Any, ...]]:
        """What `sql` answers in this cache's session."""
        return self.session.rows(sql, parameters)

    def _append(self, table: str, rows: Sequence[Mapping[str, object]]) -> None:
        """Append `rows` to `table` in one commit of this cache's session."""
        self.session.append(table, rows)


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


def _host(setup: HostSetup, *, probed_at: str | None) -> dict[str, object]:
    """The `host_facts` row recording `setup`, ordered among its alias's records by `probed_at`."""
    return {
        "ts": now(),
        "alias": setup.host,
        "probed_at": probed_at,
        "facts": setup.model_dump_json(),
        "dropped": False,
    }


def _discard(session: Session, root: Path) -> None:
    """Detach a private registry's lake and delete the directory it lived in."""
    session.close()
    rmtree(root, ignore_errors=True)


def _unregistered(run: RunRecord) -> LookupError:
    return LookupError(f"no registered run {run.handle!r} on {run.target!r}")


def _current(facts: object) -> HostSetup | None:
    """A recorded setup, None when an older mainboard wrote it and `setup` must record it anew."""
    try:
        return HostSetup.model_validate_json(str(facts), extra="ignore")
    except ValidationError:
        return None
