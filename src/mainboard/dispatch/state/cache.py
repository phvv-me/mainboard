# The dispatch registries: the shared db's `runs` table, keyed by `(target, handle,
# submitted_at)` since a scheduler handle alone is not an identity (pueue reissues small
# integer ids after a daemon restart), and its `hosts` table, one onboarding record per alias.

import weakref
from itertools import islice
from pathlib import Path

from filelock import FileLock
from patos import FrozenModel

from ...core.errors import MissionError
from .. import vocabulary
from ..lease import Lease
from ..onboard import HostSetup
from ..shared import db_file, now
from ..vocabulary import Request
from .storage import connect


class RunRecord(FrozenModel):
    """One dispatched job's provenance, the `runs` table row payload.

    handle: the scheduler's job handle, or a provider's own run id (the dispatch-wide run id).
    target: the alias the job was dispatched to.
    kind: the target's kind at submit time, a scheduler's (`ssh` / `pbs` / `slurm` / `local`) or
        a provider's (`vast` / `hpc-ai` / `modal`), which is what a later pass routes on.
    script: the original shipment spelling, or the prepared host script for a direct
        submission without a shipment. Older queued records may contain a generated script.
    args: the script arguments, shell-quoted and space-joined.
    git_sha: the short HEAD sha the workspace was at when dispatched.
    dirty: 1 when the working tree had uncommitted changes, else 0.
    submitted_at: ISO-8601 dispatch time.
    fetch_path: the results path to pull back, when `--fetch` was given.
    name: a human label for the run, shown instead of the internal script path; empty falls
        back to the script's basename at render time.
    node: the ledger slug this run serves, carried into its receipts; empty when the dispatch
        declared none, which stays a valid run.
    source: the key of the pinned source tree the run executes from, so a sweep knows which
        snapshot on the host is still in use and which is garbage. Provider creation intents
        retain this key before their machine exists.
    commit: the whole commit of the tree that owns the dispatched code, which is not always the
        workspace `git_sha` names: a monorepo dispatching a submodule's job records the
        submodule's. Empty when git answered nothing.
    digest: the content digest of that same tree, the number a run on a mirror with no history
        seals itself against. Empty for the same reason `commit` is.
    state: the last resolved scheduler outcome, memoized so a finished job (whose verdict can
        never change) is read straight from the cache instead of re-probed over ssh. `None`
        means never resolved; a terminal verdict here is trusted without touching the host.
    exit_code: the process exit status, when the scheduler reported one.
    verdict: one of the shared `vocabulary` verdicts.
    reported: the verdict a durable monitor last surfaced for this run, the change cursor that
        keeps a periodic sweep reporting only jobs newly terminal since the last check. `None`
        means never reported, so the first sweep that finds it terminal announces it.
    evidence: delivery status, separate from the computational verdict. `copied` means hashes
        were verified but release is pending; `verified` means settlement finished. An
        `unverified` correction preserves the original verdict while qualifying its evidence.
    creation: the unique provider-request label, retained when its handle replaces the intent.
    request: the original request for held jobs, or the resolved environment/resources for a
        provider creation. Only held requests are automatically retried; a lost create reply
        requires reconciliation with the provider.
    lease: accepted rental quote and release deadline, retained before provider creation.
    reason: why the row is in the state it is in, when the state cannot say it alone: the
        refusal a target's quota answered a held dispatch with. Empty for every run that was
        taken, whose outcome is read off its verdict and its exit code instead.
    """

    handle: str
    target: str
    kind: str
    script: str
    args: str
    git_sha: str
    dirty: int
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
        # A cache outlives no process it was built in, so closing is the collector's job rather
        # than a caller's. Without this the connection is only reclaimed by interpreter exit,
        # which every short-lived cache in a suite reports as an unclosed database.
        weakref.finalize(self, self.connection.close)

    @property
    def settlement(self) -> FileLock:
        """The shared launch/cancel/settle lock; remote lifecycle requires durable state."""
        if self.path == Path(":memory:"):
            raise MissionError("durable settlement requires a file-backed dispatch cache")
        return FileLock(self.path.with_suffix(".settlement.lock"), is_singleton=True)

    def forget(self, run: RunRecord) -> None:
        """Drop one run's row entirely, the only thing that ever leaves this table.

        A dispatch a quota held is a row about a request rather than about a job, and the moment
        the request goes through it is replaced by the real run under the handle the scheduler
        gave it. Keeping both would tell every later reader that thirteen jobs were dispatched as
        fourteen, so the placeholder goes rather than being settled into a verdict it never had.
        """
        self.connection.execute(
            "DELETE FROM runs WHERE target = ? AND handle = ? AND submitted_at = ?",
            (run.target, run.handle, run.submitted_at),
        )

    def delivery(self, run: RunRecord, status: str) -> RunRecord:
        """Advance evidence without replacing another process's computation or report fields."""
        return self._change(run, evidence=status)

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
        """Record that the workspace has just been mirrored to `alias`.

        The watermark a later transfer set measures its delta against. A host no onboarding
        ever recorded has nothing to stamp, and stays that way, since a host whose mirror this
        store has never seen has no delta to compute either.
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
        """Every run that has not reached a terminal verdict, newest first.

        Never truncated by a caller's limit, since a listing that hides half a dispatched wave is
        what sends an operator to `qstat` by hand. A verdict lives inside each row's payload
        rather than in a column, so the filter is here rather than in the query.
        """
        return [run for run in self.__records() if run.verdict not in vocabulary.TERMINAL]

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
        row = self.connection.execute(
            "UPDATE runs SET handle = ?, data = json_set(data, '$.handle', ?, '$.state', ?, "
            "'$.verdict', ?) WHERE target = ? AND handle = ? AND submitted_at = ? "
            "AND json_extract(data, '$.verdict') = ? RETURNING data",
            (
                handle,
                handle,
                vocabulary.QUEUED,
                vocabulary.QUEUED,
                run.target,
                run.handle,
                run.submitted_at,
                vocabulary.SUBMITTING,
            ),
        ).fetchone()
        if row is None:
            raise LookupError(f"no submitting creation {run.creation!r} on {run.target!r}")
        return RunRecord.model_validate_json(row["data"])

    def leave_prepared(
        self, run: RunRecord, verdict: str, *, lease: Lease | None = None
    ) -> RunRecord:
        """Claim preparation once, so cancellation and creation cannot both win."""
        if verdict not in vocabulary.VERDICTS[vocabulary.PREPARED]:
            raise ValueError(f"invalid prepared transition to {verdict!r}")
        row = self.connection.execute(
            "UPDATE runs SET data = json_set(data, '$.state', ?, '$.verdict', ?, '$.reported', ?, "
            "'$.lease', json(?)) "
            "WHERE target = ? AND handle = ? AND submitted_at = ? "
            "AND json_extract(data, '$.verdict') = ? RETURNING data",
            (
                verdict,
                verdict,
                verdict if verdict in vocabulary.TERMINAL else None,
                lease.model_dump_json() if lease else "null",
                run.target,
                run.handle,
                run.submitted_at,
                vocabulary.PREPARED,
            ),
        ).fetchone()
        if row is None:
            raise ValueError(f"creation {run.creation!r} is no longer prepared; do not resend it")
        return RunRecord.model_validate_json(row["data"])

    def creation(self, label: str, target: str) -> RunRecord:
        """Read the same request before or after its provider handle replaced the intent."""
        row = self.connection.execute(
            "SELECT data FROM runs WHERE target = ? AND json_extract(data, '$.creation') = ?",
            (target, label),
        ).fetchone()
        if row is None:
            raise LookupError(f"no creation {label!r} on {target!r}")
        return RunRecord.model_validate_json(row["data"])

    def report(self, run: RunRecord, verdict: str) -> None:
        """Record the verdict a durable monitor last surfaced for `run`."""
        self._change(run, reported=verdict)

    def resolve(
        self, run: RunRecord, state: str | None, exit_code: int | None, verdict: str
    ) -> RunRecord:
        """Memoize a run's resolved scheduler outcome and return the stored record.

        Update only computation fields. A dispatcher may advance evidence while a monitor holds
        an older record; neither writer may replace the other's state from that stale copy.
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
        row = self.connection.execute(
            f"UPDATE runs SET data = {expression} "
            "WHERE target = ? AND handle = ? AND submitted_at = ? RETURNING data",
            (*arguments, run.target, run.handle, run.submitted_at),
        ).fetchone()
        if row is None:
            raise LookupError(f"no registered run {run.handle!r} on {run.target!r}")
        return RunRecord.model_validate_json(row["data"])

    def run(self, handle: str, target: str | None = None) -> RunRecord:
        """The most recent run dispatched as `handle`, optionally narrowed to `target`.

        A reused handle keeps one row per run, so the newest row is the run a live command
        means; the older rows stay as history. A handle recorded on several targets is
        ambiguous without `target` and raises rather than guessing a host.
        """
        rows = self.connection.execute(
            "SELECT data FROM runs WHERE handle = ? ORDER BY submitted_at DESC", (handle,)
        ).fetchall()
        runs = [RunRecord.model_validate_json(row["data"]) for row in rows]
        if target is not None:
            runs = [run for run in runs if run.target == target]
        if not runs:
            where = f" on {target!r}" if target else ""
            raise LookupError(f"no recorded run {handle!r}{where}")
        targets = sorted({run.target for run in runs})
        if len(targets) > 1:
            raise LookupError(
                f"handle {handle!r} is recorded on {', '.join(targets)}; pass the target"
            )
        return runs[0]

    def save_host(self, setup: HostSetup) -> HostSetup:
        """Stamp `setup` with the current time and record it (upsert by alias).

        The store owns the timestamp, so a recorded onboarding always says when it happened
        and two records of the same host can be ordered against each other.
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
        landed = (run for run in self.__records() if run.verdict in vocabulary.TERMINAL)
        return list(islice(landed, limit))

    def total(self) -> int:
        """How many runs this cache holds, the count a truncated listing measures against."""
        return int(self.connection.execute("SELECT count(*) AS runs FROM runs").fetchone()["runs"])

    def tracked(self) -> list[RunRecord]:
        """Every run a durable sweep still owes an outcome for, newest first.

        A run leaves this set only once its verdict is terminal and that same verdict has been
        reported, so a sweep never announces a settled run twice and never drops the one whose
        outcome no process ever recorded, the job whose dispatching agent died before it ended.
        """
        return [
            run
            for run in self.__records()
            if run.verdict not in vocabulary.TERMINAL or run.reported != run.verdict
        ]

    def __records(self) -> list[RunRecord]:
        """Every recorded run, newest first, the whole table a verdict filter reads from."""
        rows = self.connection.execute(
            "SELECT data FROM runs ORDER BY submitted_at DESC"
        ).fetchall()
        return [RunRecord.model_validate_json(row["data"]) for row in rows]
