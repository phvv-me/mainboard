from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

from .ledger import Ledger, SetupFit
from .model import BillingModel


def card(name: str) -> str:
    """`name` as the one key a card is matched under, whoever wrote it.

    The same GPU reaches this catalog under two spellings and both are correct where they are
    written. A provider's own market answers `RTX 4090`, and that is what an imported or probed
    offer carries, while a request spells the card the way that provider's search wants it, which
    for Vast is `RTX_4090`. Matching those as plain strings priced every row naming a card at
    $0.00 while a row that named none was quoted happily, and a zero for hardware the provider
    will rent is worse than an error. So underscores read as spaces, runs of whitespace collapse,
    and case is ignored, which is exactly the difference between the two spellings and nothing
    more.

    name: a card as a request, a manifest or an offer spells it.
    """
    return " ".join(name.replace("_", " ").split()).casefold()


class Offer(FrozenModel):
    """One provider's terms for one piece of hardware in one region.

    The separation the router queries: hardware is the axis (`every GB200
    offer`), provider is an attribute of the offer, and the billing mechanics
    ride along so any offer prices any job. `available` is tri-state, None
    meaning never probed, so absence of evidence never reads as presence.
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
        """This offer's billing mechanics as the cost model's input."""
        return BillingModel(
            **self.model_dump(
                include={
                    "provider",
                    "gpu",
                    "rate_usd_hr",
                    "granularity_s",
                    "minimum_s",
                    "fees_usd",
                }
            ),
        )


class Quote(FrozenModel):
    """One offer priced for one job, expected and tail cost side by side."""

    offer: Offer
    expected_usd: float
    p90_usd: float
    setup_samples: int = 0


class Catalog:
    """Every known offer, queryable by hardware first.

    Offers arrive declared (manifest or code), imported (a gpuhunt-style
    price feed), or probed (a provider API), and the fitted setup
    distributions from the ledger turn static rates into expected and tail
    costs per job, which is the comparison static price tables cannot make.
    """

    def __init__(self, offers: tuple[Offer, ...] = ()) -> None:
        """offers: the initial roster, extended by `add`."""
        self.roster: list[Offer] = list(offers)

    @classmethod
    def load(cls, path: Path) -> Catalog:
        """A catalog read from NDJSON at `path`, empty when the file is absent.

        path: one `Offer` JSON per line, the shape `save` writes.
        """
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return cls()
        return cls(tuple(Offer.model_validate_json(line) for line in lines if line.strip()))

    def add(self, *offers: Offer) -> None:
        """Extend the roster."""
        self.roster.extend(offers)

    def offers(self, *, gpu: str = "", provider: str = "") -> list[Offer]:
        """Every offer matching the filters, hardware being the primary axis.

        gpu: the hardware name, matched on `card`, empty for all.
        provider: narrow to one provider kind, empty for all.
        """
        found = self.roster
        if gpu:
            wanted = card(gpu)
            found = [offer for offer in found if card(offer.gpu) == wanted]
        if provider:
            found = [offer for offer in found if offer.provider == provider]
        return list(found)

    def quotes(
        self,
        *,
        gpu: str,
        run_s: float,
        ledger: Ledger | None = None,
        default_setup_s: float = 300.0,
    ) -> list[Quote]:
        """Every offer for `gpu` priced for a `run_s`-second job, cheapest first.

        Fitted setup distributions price the expected and p90 cases; a
        provider with no observations yet falls back to `default_setup_s`
        for both, which deliberately penalizes the unmeasured.

        gpu: the hardware the job needs.
        run_s: the job's estimated command wall seconds.
        ledger: the observations store feeding the fits, skipped when None.
        default_setup_s: the pessimistic setup guess for unfitted providers.
        """
        priced: list[Quote] = []
        for offer in self.offers(gpu=gpu):
            fit = (
                SetupFit.from_ledger(ledger, provider=offer.provider, gpu=offer.gpu)
                if ledger is not None
                else None
            )
            expected_setup = fit.p50_s if fit else default_setup_s
            tail_setup = fit.p90_s if fit else default_setup_s
            priced.append(
                Quote(
                    offer=offer,
                    expected_usd=offer.billing.cost_usd(setup_s=expected_setup, run_s=run_s),
                    p90_usd=offer.billing.cost_usd(setup_s=tail_setup, run_s=run_s),
                    setup_samples=fit.samples if fit else 0,
                )
            )
        return sorted(priced, key=lambda quote: quote.expected_usd)

    def save(self, path: Path) -> None:
        """Write the roster as NDJSON at `path`, one offer per line, atomically enough."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(offer.model_dump_json() for offer in self.roster)
        path.write_text(text + "\n" if text else "", encoding="utf-8")
