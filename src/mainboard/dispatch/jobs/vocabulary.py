# What a dispatched job is described with: a resource request, a post-mortem state, and the
# one-word verdict lifecycle. It names no scheduler and no provider on purpose, so a provider
# backend and a queue backend speak it without importing each other's family.

from enum import StrEnum, auto

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, Field

from ..shared import HandleId
from .arch import Arch, arch


class Word(StrEnum):
    """The one word a dispatched job's verdict is, moving only along `transitions`."""

    QUEUED = auto()
    RUNNING = auto()
    PREPARED = auto()
    SUBMITTING = auto()
    # The third live word, which no backend reports itself: the queue is done with the job and
    # the sweep has not brought it home yet (see `JobState.phase`).
    FINISHED = auto()
    OK = auto()
    FAILED = auto()
    VANISHED = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
    # A job somebody stopped on purpose: a decision, not something that happened to the run like
    # `failed` or `vanished`, since a doomed job killed at minute three is a success of judgment
    # and a table filing it beside a crash teaches distrust of the column. Reachable from both
    # live states, since cancelling does not wait for the job to start.
    CANCELLED = auto()
    # A dispatch a target refused for its job quota, kept here until the quota has room. Not a
    # verdict about the unstarted work, so it is never settled and each sweep asks again.
    HELD = auto()
    # A job a plan declared and a run was told to leave out: a decision like `cancelled`, settled
    # from the start since nothing was dispatched and no watch may wait on it.
    SKIPPED = auto()

    @classmethod
    def settled(cls) -> frozenset[Word]:
        """The words settled for good, so a durable sweep trusts the cache instead of a queue
        that may have forgotten."""
        return frozenset(word for word, moves in _MOVES.items() if not moves)

    @classmethod
    def transitions(cls) -> dict[Word, set[Word]]:
        """The moves a run's recorded verdict may make.

        Each is one the code makes: a provider declining a submission reopens it, a job short
        enough to finish between two polls is never seen running, an allocation fails or a
        walltime runs out in the queue, a preempted job is requeued, and a status no backend
        vocabulary names reads as unknown. A settled word moves nowhere, so a further move (a
        stale `running` after `ok`) is refused rather than recorded.
        """
        return {word: set(moves) for word, moves in _MOVES.items()}


_MOVES: dict[Word, frozenset[Word]] = {
    Word.PREPARED: frozenset({Word.SUBMITTING, Word.FAILED, Word.CANCELLED}),
    Word.SUBMITTING: frozenset({Word.PREPARED, Word.QUEUED, Word.RUNNING}),
    Word.HELD: frozenset({Word.QUEUED, Word.RUNNING, Word.FAILED, Word.VANISHED, Word.CANCELLED}),
    Word.QUEUED: frozenset(
        {Word.RUNNING, Word.OK, Word.FAILED, Word.TIMEOUT, Word.VANISHED, Word.UNKNOWN}
        | {Word.CANCELLED}
    ),
    Word.RUNNING: frozenset(
        {Word.QUEUED, Word.OK, Word.FAILED, Word.VANISHED, Word.UNKNOWN, Word.TIMEOUT}
        | {Word.CANCELLED}
    ),
    Word.OK: frozenset(),
    Word.FAILED: frozenset(),
    Word.VANISHED: frozenset(),
    Word.UNKNOWN: frozenset(),
    Word.TIMEOUT: frozenset(),
    Word.CANCELLED: frozenset(),
    Word.SKIPPED: frozenset(),
}


class Pricing(StrEnum):
    """Which offers a search takes: either kind, only interruptible ones, or only on-demand."""

    ANY = auto()
    ON_DEMAND = auto()
    SPOT = auto()

    def admits(self, *, spot: bool) -> bool:
        """Whether an offer that is interruptible (`spot`) or not answers this choice."""
        return self is Pricing.ANY or (self is Pricing.SPOT) == spot


@Parameter(name="*")
class Allocation(FrozenModel):
    """The interactive session a queued host is asked for, the profile's queue and walltime by
    default."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    queue: str = ""
    """on a queued host, the queue the allocation targets, the profile's when omitted."""
    walltime: str = ""
    """on a queued host, the session's wall-clock limit, the profile's when omitted."""
    nodes: int = 1
    """on a queued host, how many compute nodes the allocation spans; a held session of several
    is its own, beside the one-node session."""
    keep: bool = False
    """on a host, hold the session in tmux on the far side so a dropped terminal leaves the
    allocation up, and reattach to one already held."""


class Resources(FrozenModel):
    """A backend-agnostic resource request for one job.

    Each backend maps these onto its own flags (`-l select=` for PBS, `--gpus`/`--mem` for
    SLURM, an instance type for a rental provider) and ignores what it can't express.

    gpus: 0 means none, so CPU-only scripts run on clusters without GPU GRES.
    gpu_name: the GPU type (`H100`, `A10G`) a provider backend needs; ssh-family schedulers
        ignore it.
    nodes: nodes/chunks the request spans.
    walltime: `HH:MM:SS`, when capped.
    queue: scheduler queue/partition name.
    account: charging account / group list.
    container: the container the job runs under, when the profile is containerized.
    mem_gb: system (not GPU) memory.
    attempt: the 1-based try number of this run name, exported to the job as `MB_ATTEMPT`.
    max_usd: the spend cap a provider backend requires before submitting (0.0 means unset);
        ssh-family schedulers, on owned hardware, ignore it.
    spot: rent interruptible capacity, cheaper and taken back at the provider's will.
    arch: the compute capabilities a rented card must have (`sm_120`, `hopper`, `sm_90+`), the
        cheapest card of them rented when `gpu_name` is empty.
    """

    gpus: int = 0
    gpu_name: str = ""
    nodes: int = 1
    walltime: str | None = None
    queue: str | None = None
    account: str = ""
    container: str = ""
    mem_gb: int | None = None
    max_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    attempt: int = 1
    spot: bool = False
    arch: str = ""


class OfferFilter(FrozenModel):
    """What a rented card must be: the card or its capabilities, how many, how priced, how dear.

    gpu_name: the card, as any catalog spells it; any card of `arch` when empty.
    gpus: at least this many cards per machine, 0 for any.
    arch: the compute capabilities the card must have, any when None.
    max_usd_hr: the hourly ceiling, 0 for none.
    """

    gpu_name: str = ""
    gpus: int = 0
    arch: Arch | None = None
    pricing: Pricing = Pricing.ANY
    max_usd_hr: float = 0.0

    @classmethod
    def rented(cls, resources: Resources, *, ceiling: float) -> OfferFilter:
        """What renting `resources` takes, under `ceiling` an hour.

        On-demand unless spot was asked: an interruptible machine can be taken back mid-run,
        which a job survives only through its checkpoint (`mb job submit --resume`).
        """
        return cls(
            gpu_name=resources.gpu_name,
            gpus=resources.gpus,
            arch=arch(resources.arch) if resources.arch else None,
            pricing=Pricing.SPOT if resources.spot else Pricing.ON_DEMAND,
            max_usd_hr=ceiling,
        )

    @property
    def spoken(self) -> str:
        """These filters as a person reads them, `2x H100 spot under $9.00/h`."""
        parts = (
            f"{self.gpus}x" if self.gpus else "",
            self.gpu_name,
            self.arch.spelled if self.arch else "",
            "spot" if self.pricing is Pricing.SPOT else "",
            f"under ${self.max_usd_hr:.2f}/h" if self.max_usd_hr else "",
        )
        return " ".join(part for part in parts if part)


class Request(FrozenModel):
    """One dispatch as it was asked for, enough to ask for it again unchanged.

    A quota refusal means "not now", so a workstation keeps the request and the durable sweep
    resubmits it when the quota has room. Every field is therefore the caller's own ask, never a
    resolved value: profile defaults, queue policy and market price are read again at the retry,
    so a request held overnight lands under the morning's manifest. Provider creation intents
    also retain this model, with resolved values, for recovery rather than resubmission.

    name / node / fetch: the run's label, the ledger slug it serves, and the results path to
        pull back.
    needs: the data paths a job spelled by file was asked to reach on the host.
    queue / walltime / mem_gb / gpus / gpu_name / max_usd / nodes / spot / arch: unset fields fall
        back to the host profile's defaults at retry time.
    attempt: the 1-based try number the profile's expression defaults are evaluated against.
    """

    target: str
    command: str
    name: str = ""
    node: str = ""
    fetch: str | None = None
    needs: tuple[str, ...] = ()
    env: str = ""
    container: str = ""
    queue: str = ""
    walltime: str = ""
    mem_gb: int = 0
    gpus: int = 0
    gpu_name: str = ""
    max_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    nodes: int = 1
    attempt: int = 1
    spot: bool = False
    arch: str = ""

    def resources(self) -> Resources:
        """What this request asks of its host, every unset field left for the profile to fill."""
        return Resources(
            queue=self.queue or None,
            walltime=self.walltime or None,
            mem_gb=self.mem_gb or None,
            gpus=self.gpus,
            gpu_name=self.gpu_name,
            max_usd=self.max_usd,
            nodes=self.nodes,
            attempt=self.attempt,
            spot=self.spot,
            arch=self.arch,
        )


class JobState(FrozenModel):
    """A job's post-mortem state, the unit reconcile compares against the cache.

    A verdict says `running` for everything in flight, right for a lifecycle and wrong for a
    table: an operator watching a wave of thirty five needs two running and thirty three waiting,
    the same fact on PBS, SLURM and pueue. So `stage`, `since`, `estimated_start` and `note` name
    it once here rather than each reader parsing each backend's spelling.

    handle: PBS/SLURM job id, pueue task id or provider run id, always text.
    state: the backend's current state string, None when the job vanished.
    verdict: one word, `ok` / `failed` / `running` / `vanished` / `unknown` / `timeout`.
    stage: `queued` or `running` for a job in flight, mapped from the backend's own word; empty
        where a backend reports neither.
    since: ISO-8601 instant the job entered that stage, empty when unknown.
    estimated_start: ISO-8601 instant the backend expects a queued job to start, usually empty.
    note: why a queued job has not started, in the backend's words (on PBS the resource its
        queue is short of); empty once it runs and on backends that say nothing.
    """

    handle: HandleId
    label: str | None = None
    state: str | None = None
    exit_code: int | None = None
    verdict: str
    stage: str = ""
    since: str = ""
    estimated_start: str = ""
    note: str = ""

    @property
    def phase(self) -> str:
        """The one word a listing shows for a job it has just asked its backend about.

        `stage` while in flight. A terminal verdict not yet brought home is `finished`, rather
        than the backend's letter (`F`/`E` on PBS, `CD`/`TO`/`CA` on SLURM, `Done` in pueue):
        printing `F` beside `queued` and `running` got a clean PBS job read as failed (handle
        3294174, 2026-09-04). The verdict lands on the row one sweep later. A live state with no
        stage keeps the backend's own word.
        """
        if self.stage:
            return self.stage
        if self.verdict in Word.settled():
            return Word.FINISHED
        return self.state or ""
