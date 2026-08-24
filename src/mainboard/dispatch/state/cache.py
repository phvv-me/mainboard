# The dispatch registries: the shared db's `runs` table, keyed by `(target, handle,
# submitted_at)` since a scheduler handle alone is not an identity (pueue reissues small
# integer ids after a daemon restart), and its `hosts` table, one onboarding record per alias.

import weakref
from typing import TYPE_CHECKING

from patos import FrozenModel

from .. import vocabulary
from ..onboard import HostSetup
from ..shared import db_file, now
from .storage import connect

if TYPE_CHECKING:
    from pathlib import Path


class RunRecord(FrozenModel):
    """One dispatched job's provenance, the `runs` table row payload.

    handle: the scheduler's job handle, or a provider's own run id (the dispatch-wide run id).
    target: the alias the job was dispatched to.
    kind: the target's kind at submit time, a scheduler's (`ssh` / `pbs` / `slurm` / `local`) or
        a provider's (`vast` / `hpc-ai` / `modal`), which is what a later pass routes on.
    script: the job script path on the host, or the command itself for a provider run, which
        has no script because the provider took the command directly.
    args: the script arguments, shell-quoted and space-joined.
    git_sha: the short HEAD sha the workspace was at when dispatched.
    dirty: 1 when the working tree had uncommitted changes, else 0.
    submitted_at: ISO-8601 dispatch time.
    fetch_path: the results path to pull back, when `--fetch` was given.
    name: a human label for the run, shown instead of the internal script path; empty falls
        back to the script's basename at render time.
    state: the last resolved scheduler outcome, memoized so a finished job (whose verdict can
        never change) is read straight from the cache instead of re-probed over ssh. `None`
        means never resolved; a terminal verdict here is trusted without touching the host.
    exit_code: the process exit status, when the scheduler reported one.
    verdict: one of the shared `vocabulary` verdicts.
    reported: the verdict a durable monitor last surfaced for this run, the change cursor that
        keeps a periodic sweep reporting only jobs newly terminal since the last check. `None`
        means never reported, so the first sweep that finds it terminal announces it.
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
    state: str | None = None
    exit_code: int | None = None
    verdict: str | None = None
    reported: str | None = None


class Cache:
    """Dispatch state in one SQLite file, with `runs`, `hosts` and `history` tables."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or db_file()
        self.connection = connect(self.path)
        # A cache outlives no process it was built in, so closing is the collector's job rather
        # than a caller's. Without this the connection is only reclaimed by interpreter exit,
        # which every short-lived cache in a suite reports as an unclosed database.
        weakref.finalize(self, self.connection.close)

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

    def recent(self, limit: int = 20) -> list[RunRecord]:
        """The most recent dispatched runs, newest first."""
        rows = self.connection.execute(
            "SELECT data FROM runs ORDER BY submitted_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [RunRecord.model_validate_json(row["data"]) for row in rows]

    def record(self, run: RunRecord) -> None:
        """Record a dispatched run (upsert by its `(target, handle, submitted_at)` identity)."""
        self.connection.execute(
            "INSERT INTO runs (target, handle, data, submitted_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(target, handle, submitted_at) DO UPDATE SET data = excluded.data",
            (run.target, run.handle, run.model_dump_json(), run.submitted_at),
        )

    def report(self, run: RunRecord, verdict: str) -> None:
        """Record the verdict a durable monitor last surfaced for `run`."""
        self.record(run.model_copy(update={"reported": verdict}))

    def resolve(
        self, run: RunRecord, state: str | None, exit_code: int | None, verdict: str
    ) -> RunRecord:
        """Memoize a run's resolved scheduler outcome and return the stored record.

        A terminal verdict is never re-probed once memoized. The stored record comes back so a
        caller writing to the same row again (a durable sweep advancing its `reported` cursor)
        builds on what this call just wrote instead of clobbering it with a stale copy.
        """
        fields = {"state": state, "exit_code": exit_code, "verdict": verdict}
        stored = run.model_copy(update=fields)
        self.record(stored)
        return stored

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

    def tracked(self) -> list[RunRecord]:
        """Every run a durable sweep still owes an outcome for, newest first.

        A run leaves this set only once its verdict is terminal and that same verdict has been
        reported, so a sweep never announces a settled run twice and never drops the one whose
        outcome no process ever recorded, the job whose dispatching agent died before it ended.
        """
        rows = self.connection.execute(
            "SELECT data FROM runs ORDER BY submitted_at DESC"
        ).fetchall()
        runs = [RunRecord.model_validate_json(row["data"]) for row in rows]
        return [
            run
            for run in runs
            if run.verdict not in vocabulary.TERMINAL or run.reported != run.verdict
        ]
