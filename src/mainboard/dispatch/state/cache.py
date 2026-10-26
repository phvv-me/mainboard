"""The run registry: dispatched runs recorded, moved along their verdicts, and listed."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy import ColumnElement, func, or_, select

from ...core.project import Project
from ...state import schema
from ...state.lake import Lake
from ..jobs import vocabulary
from .monitor import SettlementLock
from .registry import Creations, Hosts, Log, RunRecord

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..hosts.lease import Lease

# The terminal verdicts, for the lake to filter on.
_TERMINAL = sorted(vocabulary.Word.settled())

type Only = Literal["all", "live", "settled", "tracked"]


class Cache(Log):
    """The dispatch registries in one workspace's state lake: the runs, with `hosts` (onboarding)
    and `creations` (provider requests before their machine exists) beside them.

    The lake is opened on the first statement, not at construction, so a board that never touches
    the registry (a host installing an environment) never creates or attaches one, and the one
    session is kept for this cache's life rather than attached per statement.
    """

    @property
    def path(self) -> Path:
        """The catalog this registry is kept in."""
        return self.lake.catalog

    @property
    def settlement(self) -> SettlementLock:
        """The shared launch/cancel/settle lock; remote lifecycle requires durable state.

        Whoever holds it records itself beside it, so a pass that finds it held names the holder.
        """
        path = self.lake.out / "run" / "settlement.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        return SettlementLock(path, is_singleton=True)

    def __init__(self, lake: Lake | None = None) -> None:
        lake = lake or Lake.at(Project().workspace())
        super().__init__(lake, lake.session())
        self.hosts = Hosts(lake, self.session)
        self.creations = Creations(lake, self.session)

    def _change(self, run: RunRecord, **fields: str | int | None) -> RunRecord:
        """Update one registered identity atomically and answer its complete current record.

        A verdict moves only along `VERDICTS`, so a terminal one is never replaced: a late or
        stale report leaves the whole record as it stands rather than half of it.
        """
        with self._registry:
            current = self._at(run)
            if current is None:
                raise self._unregistered(run)
            if "verdict" in fields and not self._reaches(current.verdict, fields["verdict"]):
                return current
            changed = current.model_copy(update=fields)
            self.session.append(schema.runs_log, [self._row(changed)])
            return changed

    def count(self, *, name: str = "", target: str = "") -> int:
        """How many runs this registry holds, the count a truncated listing measures against.

        name / target: only the runs named `name` dispatched to `target`, every attempt of that
            run, when given.
        """
        named = [
            *([schema.runs.c.name == name] if name else []),
            *([schema.runs.c.target == target] if target else []),
        ]
        return int(
            self.session.rows(select(func.count()).select_from(schema.runs).where(*named))[0][0]
        )

    def delivery(self, run: RunRecord, status: str) -> RunRecord:
        """Advance evidence without replacing another process's computation or report fields."""
        return self._change(run, evidence=status)

    def forget(self, run: RunRecord) -> None:
        """Drop one run's record entirely, the only thing that ever leaves `runs`.

        A quota-held dispatch is a record about a request, replaced by the real run once it goes
        through; keeping both would read thirteen jobs as fourteen, so the placeholder goes
        rather than being settled into a verdict it never had.
        """
        with self._registry:
            self.session.append(schema.runs_log, [self._row(run, dropped=True)])

    def listed(
        self,
        only: Only = "all",
        *,
        project: str = "",
        limit: int | None = None,
        since: str = "",
    ) -> list[RunRecord]:
        """Recorded runs, newest first, filtered in the lake rather than here.

        A registry of a hundred thousand runs cost 0.7 s a listing when every record was parsed
        to keep a few; the view's columns let the lake answer only those.

        only: `live` keeps every run without a terminal verdict, which a limit never truncates (a
            listing hiding half a dispatched wave sends an operator to `qstat` by hand), `settled`
            the runs with one, `tracked` every run a durable sweep still owes an outcome for: a
            run leaves only once its terminal verdict has been reported, so a sweep never
            announces a settled run twice nor drops one whose dispatching agent died before
            recording it.
        project: only the runs dispatched from this project; every run when empty.
        since: only the runs dispatched at or after this ISO instant.
        """
        chosen = [
            *([_ONLY[only]()] if only != "all" else []),
            *([schema.runs.c.project == project] if project else []),
            *([schema.runs.c.submitted_at >= since] if since else []),
        ]
        return self._records(self._newest().where(*chosen).limit(limit))

    def record(self, run: RunRecord) -> None:
        """Record a dispatched run, replacing whatever its identity recorded before."""
        with self._registry:
            self.session.append(schema.runs_log, [self._row(run)])

    def relet(self, run: RunRecord, lease: Lease) -> RunRecord:
        """Replace `run`'s rental lease, the deadline a sweep releases the machine at."""
        with self._registry:
            current = self._at(run)
            if current is None:
                raise self._unregistered(run)
            relet = current.model_copy(update={"lease": lease})
            self.session.append(schema.runs_log, [self._row(relet)])
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


def _tracked() -> ColumnElement[bool]:
    """Whether a durable sweep still owes a run its outcome: unsettled, or settled unreported."""
    return or_(
        _unsettled(),
        schema.runs.c.record["reported"].as_string().is_distinct_from(schema.runs.c.verdict),
    )


def _unsettled() -> ColumnElement[bool]:
    """Whether a run has no terminal verdict yet."""
    return func.coalesce(schema.runs.c.verdict, "").not_in(_TERMINAL)


# The condition of each listing `Cache.listed` names but `all`.
_ONLY: dict[str, Callable[[], ColumnElement[bool]]] = {
    "live": _unsettled,
    "settled": lambda: schema.runs.c.verdict.in_(_TERMINAL),
    "tracked": _tracked,
}
