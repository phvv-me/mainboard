# The dispatch registries: the shared db's `runs` table, keyed by `(target, handle,
# submitted_at)` since a scheduler handle alone is not an identity (pueue reissues small
# integer ids after a daemon restart), and its `hosts` table, one onboarding record per alias.

import weakref
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock
from patos import FrozenModel

from ...core.errors import MissionError
from .. import vocabulary
from ..lease import Lease
from ..onboard import HostSetup
from ..shared import db_file, now
from ..vocabulary import Request
from .storage import connect

if TYPE_CHECKING:
    from collections.abc import Sequence

# The `runs` identity every single-row write matches, bound by `_identity`.
_IDENTITY = "target = ? AND handle = ? AND submitted_at = ?"

# The name `jobs` prints for a run: its label, else the script it was submitted as.
_LABEL = "coalesce(nullif(json_extract(data, '$.name'), ''), json_extract(data, '$.script'))"


class RunRecord(FrozenModel):
    """One dispatched job's provenance, the `runs` table row payload.

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


class Cache:
    """Dispatch state in one SQLite file, with `runs`, `hosts` and `history` tables."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or db_file()
        if self.path != Path(":memory:"):
            self.path = self.path.resolve()
        self.connection = connect(self.path)
        # Closing is the collector's job, not a caller's: otherwise only interpreter exit reclaims
        # it, which every short-lived cache in a suite reports as an unclosed database.
        weakref.finalize(self, self.connection.close)

    @property
    def settlement(self) -> FileLock:
        """The shared launch/cancel/settle lock; remote lifecycle requires durable state."""
        if self.path == Path(":memory:"):
            raise MissionError("durable settlement requires a file-backed dispatch cache")
        return FileLock(self.path.with_suffix(".settlement.lock"), is_singleton=True)

    def forget(self, run: RunRecord) -> None:
        """Drop one run's row entirely, the only thing that ever leaves this table.

        A quota-held dispatch is a row about a request, replaced by the real run once it goes
        through; keeping both would read thirteen jobs as fourteen, so the placeholder goes
        rather than being settled into a verdict it never had.
        """
        self.connection.execute(f"DELETE FROM runs WHERE {_IDENTITY}", _identity(run))

    def delivery(self, run: RunRecord, status: str) -> RunRecord:
        """Advance evidence without replacing another process's computation or report fields."""
        return self._change(run, evidence=status)

    def drop_host(self, alias: str) -> None:
        """Forget `alias`'s onboarding, for a machine that no longer exists to be set up."""
        self.connection.execute("DELETE FROM hosts WHERE alias = ?", (alias,))

    def host(self, alias: str) -> HostSetup:
        """`alias`'s recorded onboarding, raising when the host was never set up."""
        row = self.connection.execute(
            "SELECT facts FROM hosts WHERE alias = ?", (alias,)
        ).fetchone()
        if row is None:
            raise LookupError(f"host {alias!r} has never been set up; run `setup {alias}`")
        return HostSetup.model_validate_json(row["facts"])

    def hosts(self) -> list[HostSetup]:
        """Every onboarded host, most recently set up first."""
        rows = self.connection.execute(
            "SELECT facts FROM hosts ORDER BY probed_at DESC"
        ).fetchall()
        return [HostSetup.model_validate_json(row["facts"]) for row in rows]

    def mark_synced(self, alias: str) -> None:
        """Stamp `alias`'s mirror watermark, which a later transfer set measures its delta from.

        A host never onboarded has no mirror this store has seen, so nothing to stamp.
        """
        try:
            setup = self.host(alias)
        except LookupError:
            return
        self.connection.execute(
            "UPDATE hosts SET facts = ? WHERE alias = ?",
            (setup.model_copy(update={"synced_at": now()}).model_dump_json(), alias),
        )

    def live(self) -> list[RunRecord]:
        """Every run without a terminal verdict, newest first, never truncated by a limit.

        A listing that hides half a dispatched wave sends an operator to `qstat` by hand. The
        verdict lives in each row's payload, not a column, so the filter is here, not in SQL.
        """
        return [run for run in self.recent(None) if run.verdict not in vocabulary.TERMINAL]

    def recent(self, limit: int | None = 20) -> list[RunRecord]:
        """Dispatched runs, newest first; None retains all historical declarations."""
        rows = self.connection.execute(
            "SELECT data FROM runs ORDER BY submitted_at DESC LIMIT ?",
            (-1 if limit is None else limit,),
        ).fetchall()
        return [RunRecord.model_validate_json(row["data"]) for row in rows]

    def record(self, run: RunRecord) -> None:
        """Record a dispatched run (upsert by its `(target, handle, submitted_at)` identity)."""
        self.connection.execute(
            "INSERT INTO runs (target, handle, data, submitted_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(target, handle, submitted_at) DO UPDATE SET data = excluded.data",
            (run.target, run.handle, run.model_dump_json(), run.submitted_at),
        )

    def reserve(self, run: RunRecord) -> None:
        """Reserve a creation once; unresolved identical requests cannot allocate twice."""
        row = self.connection.execute(
            "INSERT INTO runs (target, handle, data, submitted_at) "
            "SELECT ?, ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM runs WHERE target = ? "
            "AND json_extract(data, '$.verdict') IN (?, ?) "
            "AND json_extract(data, '$.script') = ? "
            "AND json_extract(data, '$.digest') = ?) RETURNING handle",
            (
                run.target,
                run.handle,
                run.model_dump_json(),
                run.submitted_at,
                run.target,
                vocabulary.PREPARED,
                vocabulary.SUBMITTING,
                run.script,
                run.digest,
            ),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"an unresolved creation already exists for {run.name or run.script!r} on "
                f"{run.target}; inspect its job record and provider label before retrying"
            )

    def bind(self, run: RunRecord, handle: str) -> RunRecord:
        """Replace the intent identity in one statement, preserving its provenance and time."""
        return self._returned(
            "UPDATE runs SET handle = ?, data = json_set(data, '$.handle', ?, '$.state', ?, "
            f"'$.verdict', ?) WHERE {_IDENTITY} "
            "AND json_extract(data, '$.verdict') = ? RETURNING data",
            (
                handle,
                handle,
                vocabulary.QUEUED,
                vocabulary.QUEUED,
                *_identity(run),
                vocabulary.SUBMITTING,
            ),
            missing=LookupError(f"no submitting creation {run.creation!r} on {run.target!r}"),
        )

    def leave_prepared(
        self, run: RunRecord, verdict: str, *, lease: Lease | None = None
    ) -> RunRecord:
        """Claim preparation once, so cancellation and creation cannot both win."""
        if verdict not in vocabulary.VERDICTS[vocabulary.PREPARED]:
            raise ValueError(f"invalid prepared transition to {verdict!r}")
        return self._returned(
            "UPDATE runs SET data = json_set(data, '$.state', ?, '$.verdict', ?, '$.reported', ?, "
            f"'$.lease', json(?)) WHERE {_IDENTITY} "
            "AND json_extract(data, '$.verdict') = ? RETURNING data",
            (
                verdict,
                verdict,
                verdict if verdict in vocabulary.TERMINAL else None,
                lease.model_dump_json() if lease else "null",
                *_identity(run),
                vocabulary.PREPARED,
            ),
            missing=ValueError(
                f"creation {run.creation!r} is no longer prepared; do not resend it"
            ),
        )

    def reopen(self, run: RunRecord) -> RunRecord:
        """Return a submitting creation to prepared, once the provider has declined it.

        The refusal proves nothing was allocated, so the request may be sent again or left to
        `interrupted` to close.
        """
        return self._returned(
            "UPDATE runs SET data = json_set(data, '$.state', ?, '$.verdict', ?, '$.lease', "
            f"json('null')) WHERE {_IDENTITY} "
            "AND json_extract(data, '$.verdict') = ? RETURNING data",
            (vocabulary.PREPARED, vocabulary.PREPARED, *_identity(run), vocabulary.SUBMITTING),
            missing=ValueError(f"creation {run.creation!r} is not submitting; nothing to reopen"),
        )

    def creation(self, label: str, target: str) -> RunRecord:
        """Read the same request before or after its provider handle replaced the intent."""
        return self._returned(
            "SELECT data FROM runs WHERE target = ? AND json_extract(data, '$.creation') = ?",
            (target, label),
            missing=LookupError(f"no creation {label!r} on {target!r}"),
        )

    def relet(self, run: RunRecord, lease: Lease) -> RunRecord:
        """Replace `run`'s rental lease, the deadline a sweep releases the machine at."""
        return self._returned(
            f"UPDATE runs SET data = json_set(data, '$.lease', json(?)) WHERE {_IDENTITY} "
            "RETURNING data",
            (lease.model_dump_json(), *_identity(run)),
            missing=_unregistered(run),
        )

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

    def _change(self, run: RunRecord, **fields: str | int | None) -> RunRecord:
        """Update one registered identity atomically and return its complete current record."""
        arguments = tuple(item for key, value in fields.items() for item in (f"$.{key}", value))
        placeholders = ", ".join("?, ?" for _ in fields)
        expression = f"json_set(data, {placeholders})"
        if "verdict" in fields:
            terminal = tuple(sorted(vocabulary.TERMINAL))
            held = ", ".join("?" for _ in terminal)
            expression = (
                f"CASE WHEN json_extract(data, '$.verdict') IN ({held}) "
                "AND json_extract(data, '$.verdict') != ? "
                f"THEN data ELSE {expression} END"
            )
            arguments = (*terminal, fields["verdict"], *arguments)
        return self._returned(
            f"UPDATE runs SET data = {expression} WHERE {_IDENTITY} RETURNING data",
            (*arguments, *_identity(run)),
            missing=_unregistered(run),
        )

    def _returned(
        self, sql: str, parameters: Sequence[str | int | None], *, missing: Exception
    ) -> RunRecord:
        """The one row `sql` returns as a record, raising `missing` when it matched none."""
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            raise missing
        return RunRecord.model_validate_json(row["data"])

    def run(self, handle: str, target: str | None = None) -> RunRecord:
        """The newest run dispatched as `handle` (older rows are history), narrowed to `target`.

        `handle` may also be the name `jobs` prints for a run, its label or else its script, the
        spelling an operator copies off that table; a handle wins where both match. A handle
        recorded on several targets, or a name on several runs, raises with the candidates
        rather than guessing one.
        """
        runs = self._runs("handle", handle, target) or self._runs(_LABEL, handle, target)
        if not runs:
            where = f" on {target!r}" if target else ""
            raise LookupError(f"no recorded run {handle!r}{where}")
        candidates = sorted({f"{run.target} {run.handle}" for run in runs})
        if len(candidates) > 1:
            raise LookupError(
                f"{handle!r} names runs {', '.join(candidates)}; pass one handle and its host"
            )
        return runs[0]

    def _runs(self, column: str, value: str, target: str | None) -> list[RunRecord]:
        """Every row whose `column` expression is `value`, on `target` if given, newest first."""
        rows = self.connection.execute(
            f"SELECT data FROM runs WHERE {column} = ? AND target = coalesce(?, target) "
            "ORDER BY submitted_at DESC",
            (value, target),
        ).fetchall()
        return [RunRecord.model_validate_json(row["data"]) for row in rows]

    def save_host(self, setup: HostSetup) -> HostSetup:
        """Stamp `setup` with the current time and upsert it by alias.

        The store owns the timestamp, so two records of the same host can always be ordered.
        """
        stamped = setup.model_copy(update={"onboarded_at": now()})
        self.connection.execute(
            "INSERT INTO hosts (alias, facts, probed_at) VALUES (?, ?, ?) "
            "ON CONFLICT(alias) DO UPDATE SET facts = excluded.facts, "
            "probed_at = excluded.probed_at",
            (stamped.host, stamped.model_dump_json(), stamped.onboarded_at),
        )
        return stamped

    def settled(self, limit: int) -> list[RunRecord]:
        """The `limit` most recently dispatched runs whose verdict is terminal, newest first."""
        landed = (run for run in self.recent(None) if run.verdict in vocabulary.TERMINAL)
        return list(islice(landed, limit))

    def total(self) -> int:
        """How many runs this cache holds, the count a truncated listing measures against."""
        return int(self.connection.execute("SELECT count(*) AS runs FROM runs").fetchone()["runs"])

    def tracked(self) -> list[RunRecord]:
        """Every run a durable sweep still owes an outcome for, newest first.

        A run leaves only once its terminal verdict has been reported, so a sweep never announces
        a settled run twice nor drops one whose dispatching agent died before recording it.
        """
        return [
            run
            for run in self.recent(None)
            if run.verdict not in vocabulary.TERMINAL or run.reported != run.verdict
        ]


def _identity(run: RunRecord) -> tuple[str, str, str]:
    """The parameters `_IDENTITY` binds for `run`."""
    return run.target, run.handle, run.submitted_at


def _unregistered(run: RunRecord) -> LookupError:
    return LookupError(f"no registered run {run.handle!r} on {run.target!r}")
