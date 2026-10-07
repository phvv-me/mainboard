from datetime import UTC, datetime
from typing import TYPE_CHECKING

from patos import FrozenModel
from sqlalchemy import func, literal_column, select

from ..state import schema
from .ledger import Ledger, SetupFit
from .model import BillingModel

if TYPE_CHECKING:
    from ..state.lake import Session


def card(name: str) -> str:
    """`name` as the one key a card is matched under, whoever wrote it.

    Offers carry the market's `RTX 4090` while a request spells the provider's search form
    (`RTX_4090` on Vast). Plain string matching priced every row naming a card at $0.00, worse
    than an error, so underscores read as spaces, whitespace collapses and case is ignored:
    exactly the difference between the spellings and nothing more.
    """
    return " ".join(name.replace("_", " ").split()).casefold()


class Offer(FrozenModel):
    """One provider's terms for one piece of hardware in one region.

    Hardware is the query axis (`every GB200 offer`), provider an attribute, and the billing
    mechanics ride along so any offer prices any job. `available` is None when never probed, so
    absence of evidence never reads as presence.
    """

    provider: str
    gpu: str
    gpu_count: int = 1
    spot: bool = False
    region: str = ""
    rate_usd_hr: float
    granularity_s: int = 1
    minimum_s: int = 0
    fees_usd: float = 0.0
    available: bool | None = None
    source: str = "declared"

    @property
    def billing(self) -> BillingModel:
        return BillingModel(**self.model_dump(include=set(BillingModel.model_fields)))


class Quote(FrozenModel):
    """One offer priced for one job, expected and tail cost side by side."""

    offer: Offer
    expected_usd: float
    p90_usd: float
    setup_samples: int = 0


class Catalog:
    """Every known offer, queryable by hardware first.

    Offers arrive declared (manifest or code), imported (a gpuhunt-style price feed) or probed (a
    provider API), and the ledger's fitted setup distributions turn static rates into expected
    and tail costs per job, the comparison static price tables cannot make.
    """

    def __init__(self, offers: tuple[Offer, ...] = ()) -> None:
        self.roster: list[Offer] = list(offers)

    @classmethod
    def load(cls, session: Session) -> Catalog:
        """The roster the lake's newest `save` kept, empty when none was ever saved."""
        fields = tuple(Offer.model_fields)
        quotes = schema.quotes
        newest = select(func.max(quotes.c.ts)).scalar_subquery()
        rows = session.rows(
            select(*(quotes.c[name] for name in fields))
            .where(quotes.c.ts == newest)
            .order_by(literal_column("rowid"))
        )
        return cls(
            tuple(Offer.model_validate(dict(zip(fields, row, strict=True))) for row in rows)
        )

    def add(self, *offers: Offer) -> None:
        self.roster.extend(offers)

    def offers(self, *, gpu: str = "", provider: str = "") -> list[Offer]:
        """Every offer matching the filters, an empty one matching all; `gpu` matches on `card`."""
        return [
            offer
            for offer in self.roster
            if (not gpu or card(offer.gpu) == card(gpu))
            and (not provider or offer.provider == provider)
        ]

    def quotes(
        self,
        *,
        gpu: str,
        run_s: float,
        ledger: Ledger | None = None,
        default_setup_s: float = 300.0,
    ) -> list[Quote]:
        """Every offer for `gpu` priced for a job of `run_s` command wall seconds, cheapest first.

        Fitted setup distributions (none without a `ledger`) price the expected and p90 cases; a
        provider with no fit yet falls back to `default_setup_s` for both, deliberately
        penalizing the unmeasured.
        """
        priced: list[Quote] = []
        for offer in self.offers(gpu=gpu):
            fit = (
                SetupFit.from_ledger(ledger, provider=offer.provider, gpu=offer.gpu)
                if ledger
                else None
            )
            expected, tail = (fit.p50_s, fit.p90_s) if fit else (default_setup_s, default_setup_s)
            priced.append(
                Quote(
                    offer=offer,
                    expected_usd=offer.billing.cost_usd(setup_s=expected, run_s=run_s),
                    p90_usd=offer.billing.cost_usd(setup_s=tail, run_s=run_s),
                    setup_samples=fit.samples if fit else 0,
                )
            )
        return sorted(priced, key=lambda quote: quote.expected_usd)

    def save(self, session: Session) -> None:
        """Keep the whole roster in the lake's `quotes` under one stamp, the newest roster."""
        stamp = datetime.now(UTC)
        session.append(
            schema.quotes, [{"ts": stamp, **offer.model_dump()} for offer in self.roster]
        )
