# The table a batch prints before anything runs: one row per job, what it ships, what hardware it
# lands on, how long that target takes to start work, and what the meter will say. Nothing here
# executes, connects, or rents; every number comes from what this workspace already recorded.

from typing import TYPE_CHECKING

from patos import FrozenModel

from ..compute import summary
from ..core.project import Project
from ..costs import Catalog, Ledger, Quote, SetupFit
from ..dispatch.backends.base import ProviderBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..board import Board
    from .spec import BatchJob
    from .transfer import TransferSet

# What an unmeasured target is assumed to spend getting ready, the same pessimism the catalog
# applies to a provider nobody has timed yet: better to over-quote than to promise a number no
# observation supports.
_UNFITTED_SETUP_S = 300.0

# Where the workspace keeps the two files this reads, both written by the tool itself.
_CATALOG = "catalog.ndjson"
_COSTS = "costs"


def platform(*, alias: str, kind: str) -> str:
    """The key a target's setup time is recorded and read under.

    A rented instance's provisioning belongs to the provider, since every rental of that kind
    waits the same way, while a queue's belongs to the machine, since two ssh boxes behind the
    same scheduler wait nothing alike.

    alias: the declared host alias.
    kind: that host's kind, a scheduler's or a provider's.
    """
    return kind if kind in ProviderBackend.names() else alias


class JobEstimate(FrozenModel):
    """One job priced before it runs.

    job: the job's name inside the batch.
    target: the alias it is dispatched to.
    kind: how that target is reached, a scheduler kind or a provider's.
    hardware: what it lands on, read from onboarding for a machine and from the request for a
        rental, empty when this workspace has never been told.
    wire_bytes: what the job ships, compressed, from its transfer set.
    runtime_s: the command's expected wall seconds, as the spec declared it.
    setup_p50_s / setup_p90_s: the fitted time from dispatch to the command starting, the
        unfitted assumption when fewer than three observations exist.
    setup_samples: how many observations the fit stands on, zero meaning assumed.
    rate_usd_hr: what the target charges per hour, zero for hardware this workspace owns.
    expected_usd / p90_usd: the median and tail cost of this job under that rate.
    """

    job: str
    target: str
    kind: str
    hardware: str = ""
    wire_bytes: int = 0
    runtime_s: float = 0.0
    setup_p50_s: float = 0.0
    setup_p90_s: float = 0.0
    setup_samples: int = 0
    rate_usd_hr: float = 0.0
    expected_usd: float = 0.0
    p90_usd: float = 0.0


class BatchEstimate(FrozenModel):
    """Every job's row and what the batch adds up to.

    batch: the batch id these rows belong to.
    jobs: one row per declared job, in spec order.
    """

    batch: str
    jobs: tuple[JobEstimate, ...]

    @property
    def wire_bytes(self) -> int:
        """What the whole batch ships, compressed."""
        return sum(job.wire_bytes for job in self.jobs)

    @property
    def expected_usd(self) -> float:
        """What the whole batch is expected to cost."""
        return sum(job.expected_usd for job in self.jobs)

    @property
    def p90_usd(self) -> float:
        """What the whole batch costs when every setup lands in its own tail."""
        return sum(job.p90_usd for job in self.jobs)


class Estimator:
    """Prices a batch from what this workspace already knows, without touching a target.

    Two files behind it, both the tool's own: the offer catalog says what hardware costs, and the
    cost ledger says how long each platform actually takes to start work. A target with neither
    is still a row, priced at nothing per hour with an assumed setup, which is exactly right for
    a machine we own and honest for one nobody has measured.
    """

    def __init__(
        self, board: Board, *, catalog: Catalog | None = None, ledger: Ledger | None = None
    ) -> None:
        """board: the workspace whose catalog, ledger and onboarding records are read.

        catalog: the offer roster, the workspace's own file when None.
        ledger: the observation store the setup fits come from, the workspace's own when None.
        """
        generated = board.root / Project().out_dir
        self.board = board
        self.catalog = catalog if catalog is not None else Catalog.load(generated / _CATALOG)
        self.ledger = ledger if ledger is not None else Ledger(generated / _COSTS)

    def row(self, job: BatchJob, transfer: TransferSet) -> JobEstimate:
        """Price one job against its target's fitted behavior and whatever offer covers it.

        The seconds and the dollars are rounded to what an estimate can actually claim. A
        fitted median carried to fifteen digits is false precision, and a reader budgeting
        against this table is deciding in cents.
        """
        profile = self.board.on(job.target).plan().profile
        key = platform(alias=job.target, kind=profile.kind)
        fit = SetupFit.from_ledger(self.ledger, provider=key, gpu=job.gpu_name)
        quote = self.quote(job, kind=profile.kind)
        return JobEstimate(
            job=job.name,
            target=job.target,
            kind=profile.kind,
            hardware=self.hardware(job),
            wire_bytes=transfer.wire_bytes,
            runtime_s=job.runtime_s,
            setup_p50_s=round(fit.p50_s if fit else _UNFITTED_SETUP_S, 2),
            setup_p90_s=round(fit.p90_s if fit else _UNFITTED_SETUP_S, 2),
            setup_samples=fit.samples if fit else 0,
            rate_usd_hr=quote.offer.rate_usd_hr if quote else 0.0,
            expected_usd=round(quote.expected_usd, 4) if quote else 0.0,
            p90_usd=round(quote.p90_usd, 4) if quote else 0.0,
        )

    def quote(self, job: BatchJob, *, kind: str) -> Quote | None:
        """The cheapest offer this provider makes for what `job` asks for, None when it is ours.

        Owned hardware has no offer and needs none: the machine is already paid for, so the row
        prices at zero rather than at a number invented for the column's sake.
        """
        priced = self.catalog.quotes(
            gpu=job.gpu_name,
            run_s=job.runtime_s,
            ledger=self.ledger,
            default_setup_s=_UNFITTED_SETUP_S,
        )
        return next((quote for quote in priced if quote.offer.provider == kind), None)

    def hardware(self, job: BatchJob) -> str:
        """What `job` lands on: what onboarding recorded, else the hardware it asked to rent."""
        try:
            setup = self.board.dispatcher.cache.host(job.target)
        except LookupError:
            return _requested(job)
        return summary(setup.hardware) if setup.hardware else _requested(job)

    def table(
        self, batch: str, jobs: Sequence[BatchJob], transfers: Sequence[TransferSet]
    ) -> BatchEstimate:
        """Every job priced against its own transfer set, in declaration order.

        batch: the batch id the rows belong to.
        jobs: the batch's declared jobs.
        transfers: each job's measured transfer set, paired by position.
        """
        return BatchEstimate(
            batch=batch,
            jobs=tuple(
                self.row(job, transfer) for job, transfer in zip(jobs, transfers, strict=True)
            ),
        )


def _requested(job: BatchJob) -> str:
    """The hardware `job` asked for, as one line, empty when it asked for nothing in particular."""
    if not job.gpus and not job.gpu_name:
        return ""
    return f"{job.gpus or 1}x {job.gpu_name}".strip()
