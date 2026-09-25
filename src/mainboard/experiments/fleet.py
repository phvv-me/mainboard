# The many-jobs surface over `Board`: a study fans `Board.on(host).submit(...)` out over many
# `(host, command)` pairs, labels each with its study, and remembers enough to resubmit a failure.

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

    from ..board import Board, Run
    from ..dispatch.dispatcher import Handle, Verdict
    from ..dispatch.state.cache import Cache
    from .reporting import StudySummary
    from .study import Progress, Study


class ResourceOverrides(TypedDict, total=False):
    """The `Board.submit` keywords a study may override, all but `name` (`Fleet` fixes it to
    the study's label) and the per-trial command."""

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
    """One fleet-tracked trial: its host alias, its command (re-issued verbatim on resubmit),
    and its owning study."""

    host: str
    command: str
    study_id: str


class Fleet:
    """Submit, track, and resubmit a study's many trials over one `Board`.

    Maps each dispatched `Handle` back to its `Dispatched` origin, since a bare scheduler handle
    carries neither a host nor a command to re-issue. The map lives only as long as this
    instance (the durable record is each study's `StudyLedger`), so drive one fleet for a
    study's whole life: submit, wait, resubmit.
    """

    def __init__(self, board: Board) -> None:
        self.board = board
        self._origins: dict[Handle, Dispatched] = {}

    @classmethod
    def overview(cls, board_root: Path, cache: Cache) -> list[StudySummary]:
        """Every study under `board_root`, each summary joined against the dispatch `cache`.

        A classmethod since listing studies needs no bound host.
        """
        return reporting.overview(cache, board_root / Project().out_dir / "studies")

    def owner(self, handle: Handle) -> str:
        """The study id owning `handle`, empty when it belongs to no study.

        Prefers this fleet's own record, then the dispatch label the run registry kept, which
        survives the submitting process.
        """
        origin = self._origins.get(handle)
        if origin is not None:
            return origin.study_id
        with suppress(LookupError):
            record = self.board.dispatcher.cache.run(handle.id, handle.host)
            return labelled_study(record.name)
        return ""

    def progress(self, study: Study) -> Progress:
        """`study`'s live trial counts, dispatch's resolved verdicts merged over its ledger."""
        ledger = StudyLedger(self.board.root, study.study_id)
        return reporting.study_progress(self.board.dispatcher.cache, ledger, study)

    def resubmit(
        self, study: Study, failed_handles: Sequence[Handle], *, attempt: int
    ) -> list[Run]:
        """Re-dispatch each failed handle's original command to its original host at `attempt`.

        `Board.submit` evaluates an expression-valued resource default against `attempt`, so a
        retry escalates (a bigger memory ceiling, say) instead of failing the same one twice.

        failed_handles: handles this same `Fleet` instance submitted.
        """
        ledger = StudyLedger(self.board.root, study.study_id)
        origins = (self._origins.pop(handle) for handle in failed_handles)
        return [
            self._dispatch(ledger, study, origin.host, origin.command, attempt=attempt)
            for origin in origins
        ]

    def settle(self, verdicts: Mapping[Handle, Verdict]) -> None:
        """Record each resolved verdict in the ledger of the study that owns its handle.

        A handle this fleet never submitted settles too, its study recovered from the durable
        dispatch label, so a fresh process can close out a study it did not start (rebuilding
        each job with `Board.job`).
        """
        for handle, verdict in verdicts.items():
            study_id = self.owner(handle)
            if study_id:
                StudyLedger(self.board.root, study_id).verdict(handle.id, state=verdict.verdict)

    def statuses(self, study: Study) -> dict[str, str]:
        """Every handle `study` has dispatched, folded to its current ledger status."""
        return StudyLedger(self.board.root, study.study_id).statuses()

    def submit_all(
        self,
        commands: Sequence[tuple[str, str]],
        *,
        study: Study,
        **resource_overrides: Unpack[ResourceOverrides],
    ) -> list[Run]:
        """Dispatch every `(host alias, shell command)` pair as one of `study`'s trials.

        Each job is named `study_label(study.study_id)`, the key a report joins against
        dispatch's run cache, and ledgered as `submitted`. The ledger's first touch records a
        `created` event carrying `study.name`, so `overview` can show a human label later.

        resource_overrides: forwarded to `Board.submit` for every trial.
        """
        ledger = StudyLedger(self.board.root, study.study_id)
        if not ledger.path.is_file():
            ledger.created(study)
        return [
            self._dispatch(ledger, study, host, command, **resource_overrides)
            for host, command in commands
        ]

    def wait_all(self, jobs: Sequence[Run]) -> dict[Handle, Verdict]:
        """Block until every job is terminal, recording each verdict in its ledger.

        Polling is `Dispatcher.await_many`'s, so the cadence stays outside this surface.
        """
        verdicts = self.board.dispatcher.await_many([job.handle for job in jobs])
        self.settle(verdicts)
        return verdicts

    def _dispatch(
        self,
        ledger: StudyLedger,
        study: Study,
        host: str,
        command: str,
        **overrides: Unpack[ResourceOverrides],
    ) -> Run:
        job = self.board.on(host).submit(command, name=study_label(study.study_id), **overrides)
        self._origins[job.handle] = Dispatched(host=host, command=command, study_id=study.study_id)
        ledger.submitted(job.handle.id, host=host)
        return job
