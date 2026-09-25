# The table a batch prints before anything runs: one row per job, what it ships, what hardware it
# lands on, how long that target takes to start work, and what the meter will say. Nothing here
# executes, dispatches or rents.
#
# It does read a provider's market, and has to. Prices once came from a stored roster nothing ever
# wrote, so every rate read $0.00 while `mainboard compute` priced the same card live, and since
# no paid dispatch happens before this table is read, that silently closed the paid lane (found
# 2026-08-25, a campaign unable to price Volta, Turing, A100 or Blackwell against a $40 cap). A
# target with no stored price is quoted from the provider's live market (the survey's read, which
# rents nothing) and the answer is kept for the next estimate.
#
# Every price is keyed on the card the dispatch will really rent (the profile default for a row
# naming none) matched under `costs.catalog.card`. Keying on the literal `gpu_name` priced a
# card-less row against the roster's cheapest offer and missed `RTX_4090` against the market's
# `RTX 4090` (found 2026-08-26, a Vast batch quoted at zero). An unpriceable row names the card
# rather than printing $0.00.

from typing import TYPE_CHECKING

from patos import FrozenModel

from ..compute import summary
from ..core.errors import MissionError
from ..core.project import Project
from ..costs import Catalog, Ledger, Quote, SetupFit
from ..dispatch.backends.base import Market, ProviderBackend, route

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..board import Board
    from .spec import BatchJob
    from .transfer import TransferSet

# An unmeasured target's assumed setup, the catalog's pessimism: over-quote rather than guess.
_UNFITTED_SETUP_S = 300.0

# Where the workspace keeps the two files this reads, both written by the tool itself.
_CATALOG = "catalog.ndjson"
_COSTS = "costs"


def platform(*, alias: str, kind: str) -> str:
    """The key a target's setup time is recorded under: the provider `kind` for a rental (every
    rental of a kind waits alike), the host `alias` for a queue (two boxes wait nothing alike)."""
    return kind if kind in ProviderBackend.names() else alias


class JobEstimate(FrozenModel):
    """One job priced before it runs.

    kind: a scheduler kind or a provider's.
    hardware: from onboarding for a machine, from the request for a rental, else empty.
    setup_p50_s / setup_p90_s: the fitted time from dispatch to the command starting, the
        unfitted assumption below three observations (`setup_samples` 0).
    expected_usd / p90_usd: the median and tail cost under `rate_usd_hr`.
    rate_source: `owned` (already paid for, a zero that is a fact), `live` (an offer rentable at
        pricing time), `catalog` (the stored roster, not re-checked), or the reason there is no
        price, whose zero means unknown rather than free.
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
    rate_source: str = ""


class BatchEstimate(FrozenModel):
    """Every job's row, in spec order, and what the batch adds up to."""

    batch: str
    jobs: tuple[JobEstimate, ...]

    @property
    def expected_usd(self) -> float:
        return sum(job.expected_usd for job in self.jobs)

    @property
    def p90_usd(self) -> float:
        """What the whole batch costs when every setup lands in its own tail."""
        return sum(job.p90_usd for job in self.jobs)

    @property
    def wire_bytes(self) -> int:
        return sum(job.wire_bytes for job in self.jobs)


class Estimator:
    """Prices a batch from what this workspace knows, asking a market for what it does not.

    The offer roster says what hardware costs and the cost ledger how long each platform takes to
    start work (both the workspace's own files by default). Owned hardware needs neither.
    """

    def __init__(
        self, board: Board, *, catalog: Catalog | None = None, ledger: Ledger | None = None
    ) -> None:
        generated = board.root / Project().out_dir
        self.board = board
        self.catalog = catalog if catalog is not None else Catalog.load(generated / _CATALOG)
        self.ledger = ledger if ledger is not None else Ledger(generated / _COSTS)

    def hardware(self, job: BatchJob, *, card: str) -> str:
        """What `job` lands on: what onboarding recorded, else the hardware it asked to rent."""
        try:
            setup = self.board.dispatcher.cache.host(job.target)
        except LookupError:
            return _requested(job, card)
        return summary(setup.hardware) if setup.hardware else _requested(job, card)

    def priced(self, job: BatchJob, *, kind: str, card: str) -> Quote | None:
        """The cheapest quote the stored roster already makes for `card` on `kind`, else None."""
        priced = self.catalog.quotes(
            gpu=card,
            run_s=job.runtime_s,
            ledger=self.ledger,
            default_setup_s=_UNFITTED_SETUP_S,
        )
        return next((quote for quote in priced if quote.offer.provider == kind), None)

    def quote(self, job: BatchJob, *, kind: str, card: str) -> tuple[Quote | None, str]:
        """The cheapest offer this provider makes for `card`, and its `rate_source`.

        A provider with no key, no route out or nothing matching is not a failure: the row carries
        the refusal, naming the card, in place of a price.
        """
        backend = route(kind)
        if backend == "ssh-family":
            return None, "owned"
        stored = self.priced(job, kind=kind, card=card)
        if stored is not None:
            return stored, "catalog"
        try:
            self.refresh(job, backend=backend, card=card)
        except (MissionError, OSError, ValueError, KeyError) as unpriced:
            return None, f"unpriced: {unpriced}"
        live = self.priced(job, kind=kind, card=card)
        if live is not None:
            return live, "live"
        return None, f"unpriced: {kind} quotes no {card or 'matching'} offer right now"

    def refresh(self, job: BatchJob, *, backend: type[ProviderBackend], card: str) -> None:
        """Ask `backend`'s market what `card` rents for, as `mainboard compute` does, and keep it.

        A backend with no market (hpc-ai, modal) leaves the roster alone and the row unpriced.
        """
        market = backend()
        if not isinstance(market, Market):
            return
        offers = market.catalog(gpu_name=card, gpus=job.gpus)
        if not offers:
            return
        self.catalog.add(*offers)
        self.catalog.save(self.board.root / Project().out_dir / _CATALOG)

    def row(self, job: BatchJob, transfer: TransferSet) -> JobEstimate:
        """Price one job against its target's fitted behavior and whatever offer covers it.

        The card is resolved as `Board.submit` resolves it, and seconds and dollars are rounded to
        what an estimate can claim, since a reader budgeting against it decides in cents.
        """
        profile = self.board.on(job.target).plan().profile
        key = platform(alias=job.target, kind=profile.kind)
        card = job.gpu_name or profile.defaults.gpu_name
        fit = SetupFit.from_ledger(self.ledger, provider=key, gpu=card)
        quote, source = self.quote(job, kind=profile.kind, card=card)
        return JobEstimate(
            job=job.name,
            target=job.target,
            kind=profile.kind,
            hardware=self.hardware(job, card=card),
            wire_bytes=transfer.wire_bytes,
            runtime_s=job.runtime_s,
            setup_p50_s=round(fit.p50_s if fit else _UNFITTED_SETUP_S, 2),
            setup_p90_s=round(fit.p90_s if fit else _UNFITTED_SETUP_S, 2),
            setup_samples=fit.samples if fit else 0,
            rate_usd_hr=quote.offer.rate_usd_hr if quote else 0.0,
            expected_usd=round(quote.expected_usd, 4) if quote else 0.0,
            p90_usd=round(quote.p90_usd, 4) if quote else 0.0,
            rate_source=source,
        )

    def table(
        self, batch: str, jobs: Sequence[BatchJob], transfers: Sequence[TransferSet]
    ) -> BatchEstimate:
        """Every job priced against its transfer set (paired by position), in declaration order."""
        return BatchEstimate(
            batch=batch,
            jobs=tuple(
                self.row(job, transfer) for job, transfer in zip(jobs, transfers, strict=True)
            ),
        )


def _requested(job: BatchJob, card: str) -> str:
    """The hardware `job` will rent (`card` resolved), empty when it asked for nothing."""
    if not job.gpus and not card:
        return ""
    return f"{job.gpus or 1}x {card}".strip()
