# The table a batch prints before anything runs: one row per job, what it ships, what hardware it
# lands on, how long that target takes to start work, and what the meter will say. Nothing here
# executes, dispatches or rents.
#
# It does read a provider's market, and it has to. The prices used to come from a stored offer
# roster alone, and nothing in this tool ever wrote that file, so `Catalog.load` always came back
# empty, every quote came back None, and every rate in the table read $0.00 for every card on
# every provider, while `mainboard compute` priced the same card live off the same key. Since the
# house rule is that no paid dispatch happens until this table has been read, a zero-rate estimate
# did not merely mislead, it silently closed the paid lane (found 2026-08-25 by a campaign that
# could not price Volta, Turing, A100 or Blackwell against a $40 cap).
#
# So a target this workspace has no stored price for is quoted from the provider's own live
# market, the same read the survey makes, and what comes back is written into the roster so the
# next estimate is free. A live offer is one that is rentable right now, which is also what turns
# a catalog price into a real one, and `rate_source` says on every row which of the two a reader
# is looking at. Reading a market rents nothing.
#
# That fix left one silent zero behind, and it is fixed here too. Every lookup was keyed on the
# row's literal `gpu_name`, which is the wrong key twice over: a row naming no card was priced
# against the cheapest offer in the whole roster rather than against the profile default it will
# actually rent, and a row naming one asked the roster under the request's own spelling
# (`RTX_4090`) while the market had answered in the provider's (`RTX 4090`), so it matched
# nothing and came back free. Both keys are now the card the dispatch will really rent, matched
# under `costs.catalog.card`, and a row that still cannot be priced says so with the card it
# could not price rather than printing $0.00 (found 2026-08-26 by a Vast batch quoted at zero
# for every card but the profile's own).

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
    rate_source: where the price came from, which is the difference between a number a reader
        can budget against and one they cannot. `owned` is hardware this workspace already paid
        for, so its zero is a fact. `live` was quoted from an offer that was rentable at pricing
        time. `catalog` came from the stored roster and was not re-checked, so it may name a
        machine somebody else has since taken. Anything else is the reason there is no price,
        and the row's zero means unknown rather than free.
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
    """Every job's row and what the batch adds up to.

    batch: the batch id these rows belong to.
    jobs: one row per declared job, in spec order.
    """

    batch: str
    jobs: tuple[JobEstimate, ...]

    @property
    def expected_usd(self) -> float:
        """What the whole batch is expected to cost."""
        return sum(job.expected_usd for job in self.jobs)

    @property
    def p90_usd(self) -> float:
        """What the whole batch costs when every setup lands in its own tail."""
        return sum(job.p90_usd for job in self.jobs)

    @property
    def wire_bytes(self) -> int:
        """What the whole batch ships, compressed."""
        return sum(job.wire_bytes for job in self.jobs)


class Estimator:
    """Prices a batch from what this workspace knows, asking a market for what it does not.

    Two files behind it, both the tool's own: the offer roster says what hardware costs and the
    cost ledger says how long each platform actually takes to start work. Owned hardware needs
    neither and prices at zero, which is a fact about a machine already paid for rather than a
    gap. A rental does need them, and a fresh workspace has an empty roster, so the provider's
    own market fills it on the first ask and the answer is kept for every ask after.

    Nothing here dispatches or rents. Reading a market is the same read the survey makes, which
    is precisely why they now agree.
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
        """The cheapest offer this provider makes for `card`, and where the price came from.

        Owned hardware has no offer and needs none: the machine is already paid for, so the row
        prices at zero and says `owned` rather than inventing a number for the column's sake.

        A provider this workspace holds no stored price for is asked for one, because the stored
        roster is empty on every fresh workspace and a table of zeroes is worse than no table at
        all. What the market answers is kept, so only the first estimate of a given card pays for
        the round trip. A provider with no key, no route out or nothing matching is not a failure
        here: the row carries the refusal in place of a price, and it names the card it could not
        price, which is what tells a reader the zero means unknown rather than free.
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
        """Ask `backend`'s own market what `card` rents for, and keep what it answers.

        The same read `mainboard compute` makes, which is the whole point: the survey and the
        estimate disagreed because one asked the provider and the other asked a file nothing
        wrote. A backend that quotes no market leaves the roster alone and the row unpriced,
        which is the honest answer for hpc-ai and modal, neither of which publishes a market.
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

        Every price here is keyed on the card the dispatch will really rent, which for a row that
        names none is the host profile's default, exactly as `Board.submit` resolves it. A row
        naming no card is not a row that rents no card, and pricing one against the whole roster
        quoted whatever was cheapest in the file rather than the machine the meter would read.

        The seconds and the dollars are rounded to what an estimate can actually claim. A
        fitted median carried to fifteen digits is false precision, and a reader budgeting
        against this table is deciding in cents.
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


def _requested(job: BatchJob, card: str) -> str:
    """The hardware `job` will rent, as one line, empty when it asked for nothing in particular.

    job: the declared job, for the GPU count it asks for.
    card: the resolved card, the job's own or the target profile's default.
    """
    if not job.gpus and not card:
        return ""
    return f"{job.gpus or 1}x {card}".strip()
