# The many-jobs surface over `Board`. A single trial is already `Board.on(host).submit(...)`;
# a study fans that out over many `(host, command)` pairs at once, labels each with its study,
# and remembers enough to resubmit a failure without the caller re-deriving anything.

from contextlib import suppress
from typing import TYPE_CHECKING, TypedDict

from patos import FrozenModel

from ..core.project import Project
from . import reporting
from .identity import labelled_study, study_label
from .study import StudyLedger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Unpack

    from ..board import Board, Job
    from ..dispatch.dispatcher import Handle, Verdict
    from ..dispatch.state.cache import Cache
    from .reporting import StudySummary
    from .study import Progress, Study


class ResourceOverrides(TypedDict, total=False):
    """The `Board.submit` resource keywords `Fleet.submit_all` may override per study.

    Everything `Board.submit` accepts besides `name` (which `Fleet` itself fixes to the
    study's label) and the command (which `submit_all` takes per trial, not per study).
    """

    queue: str
    walltime: str
    mem_gb: int
    gpus: int
    nodes: int
    attempt: int
    fetch: str | None
    env: str
    container: str


class Dispatched(FrozenModel):
    """One fleet-tracked trial: the command that produced it and the study that owns it.

    host: the alias it was dispatched to.
    command: the command it ran, re-issuable verbatim on resubmit.
    study_id: the owning study's id, the ledger a resolved verdict is recorded into.
    """

    host: str
    command: str
    study_id: str


class Fleet:
    """Submit, track, and resubmit a study's many trials over one `Board`.

    Retains an in-memory map from each dispatched `Handle` back to its `Dispatched` origin, the
    bookkeeping `resubmit` needs since a bare scheduler handle carries neither a host nor a
    command to re-issue. The durable record lives in each study's `StudyLedger`; this map only
    spans one `Fleet` instance's lifetime, so a caller drives one fleet for a study's whole life
    (submit, wait, resubmit) rather than rebuilding one per call.
    """

    def __init__(self, board: Board) -> None:
        self.board = board
        self._origins: dict[Handle, Dispatched] = {}

    @classmethod
    def overview(cls, board_root: Path, cache: Cache) -> list[StudySummary]:
        """Every study found under `board_root`, each summary joined against `cache`.

        A classmethod rather than an instance method, since listing every study needs no bound
        host, only the same `(board_root, cache)` shape `reporting.overview` reads directly;
        `Fleet` carries it so a `Board`-level surface needs no import beyond `Fleet` itself.

        board_root: the workspace root a study's `StudyLedger` files live under.
        cache: the dispatch run registry each summary's counts are joined against.
        """
        ledgers_root = board_root / Project().out_dir / "studies"
        return reporting.overview(cache, ledgers_root)

    def progress(self, study: Study) -> Progress:
        """`study`'s live trial counts, dispatch's resolved verdicts merged over the ledger.

        Delegates to `reporting.study_progress`, joining `study`'s own `StudyLedger` against
        this board's shared dispatch `Cache`, so a caller reads one study's up-to-date shape
        without assembling the join itself.
        """
        ledger = StudyLedger(self.board.root, study.study_id)
        return reporting.study_progress(self.board.dispatcher.cache, ledger, study)

    def resubmit(
        self, study: Study, failed_handles: Sequence[Handle], *, attempt: int
    ) -> list[Job]:
        """Re-dispatch each of `failed_handles`'s original commands at `attempt`.

        `Board.submit` evaluates an expression-valued resource default against `attempt`, so a
        retry escalates (a bigger memory ceiling, say) instead of failing the same ceiling
        twice.

        failed_handles: handles this same `Fleet` instance submitted, each re-dispatched to its
            original host with its original command.
        """
        ledger = StudyLedger(self.board.root, study.study_id)
        jobs: list[Job] = []
        for handle in failed_handles:
            origin = self._origins.pop(handle)
            job = self.board.on(origin.host).submit(
                origin.command, name=study_label(study.study_id), attempt=attempt
            )
            self._origins[job.handle] = Dispatched(
                host=origin.host, command=origin.command, study_id=study.study_id
            )
            ledger.submitted(job.handle.id, host=origin.host)
            jobs.append(job)
        return jobs

    def statuses(self, study: Study) -> dict[str, str]:
        """Every handle `study` has dispatched, folded to its current ledger status."""
        return StudyLedger(self.board.root, study.study_id).statuses()

    def submit_all(
        self,
        commands: Sequence[tuple[str, str]],
        *,
        study: Study,
        **resource_overrides: Unpack[ResourceOverrides],
    ) -> list[Job]:
        """Dispatch every `(host, command)` pair as one of `study`'s trials.

        Each job carries `study_label(study.study_id)` as its name, the label a later reader
        joins against dispatch's own run cache, and its dispatch is recorded as a `submitted` event
        in the study's `StudyLedger`. The ledger's first touch also records a `created` event
        carrying `study.name`, so a later `overview` can read a human label back without the
        caller having to keep the original `Study` around.

        commands: the trials to launch, each a `(host alias, shell command)` pair.
        resource_overrides: forwarded to `Board.submit` for every trial (`queue`, `mem_gb`, ...).
        """
        ledger = StudyLedger(self.board.root, study.study_id)
        if not ledger.path.is_file():
            ledger.created(study)
        jobs: list[Job] = []
        for host, command in commands:
            job = self.board.on(host).submit(
                command, name=study_label(study.study_id), **resource_overrides
            )
            self._origins[job.handle] = Dispatched(
                host=host, command=command, study_id=study.study_id
            )
            ledger.submitted(job.handle.id, host=host)
            jobs.append(job)
        return jobs

    def settle(self, verdicts: Mapping[Handle, Verdict]) -> None:
        """Record each resolved verdict in the ledger of the study that owns its handle.

        A handle this fleet never submitted settles too, since the owning study is recoverable
        from the durable dispatch label the trial carries. That is what lets a fresh process
        close out a study it did not start, rebuilding each job with `Board.job` and settling
        it here, instead of the verdicts living only in the process that submitted them.

        verdicts: the terminal outcomes to record, keyed by handle.
        """
        for handle, verdict in verdicts.items():
            study_id = self.owner(handle)
            if study_id:
                StudyLedger(self.board.root, study_id).verdict(handle.id, state=verdict.verdict)

    def owner(self, handle: Handle) -> str:
        """The study id owning `handle`, empty when the handle belongs to no study.

        Prefers this fleet's own record of what it dispatched and falls back to the dispatch
        label the run registry kept, which survives the process that submitted it.
        """
        origin = self._origins.get(handle)
        if origin is not None:
            return origin.study_id
        with suppress(LookupError):
            record = self.board.dispatcher.cache.run(handle.id, handle.host)
            return labelled_study(record.name)
        return ""

    def wait_all(self, jobs: Sequence[Job]) -> dict[Handle, Verdict]:
        """Block until every job in `jobs` is terminal, recording each verdict in its ledger.

        Delegates the actual polling to `Dispatcher.await_many`; no loop of its own, so the
        cadence a durable monitor sweep owns stays entirely outside this surface.
        """
        verdicts = self.board.dispatcher.await_many([job.handle for job in jobs])
        self.settle(verdicts)
        return verdicts
