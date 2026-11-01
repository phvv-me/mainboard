from math import inf, isfinite
from operator import itemgetter
from typing import TYPE_CHECKING

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, field_validator

from ...core.errors import MissionError
from ...costs.catalog import Offer
from ...costs.ledger import Ledger
from ...costs.score import Rank, Scorecard
from ...dispatch.backends import HyperstackBackend, ShadeformBackend
from ...dispatch.backends.base import ProviderBackend
from ...dispatch.backends.capabilities import Account
from ...dispatch.backends.cloud.offers import hunt
from ...dispatch.backends.hpcai import HpcAiBackend
from ...dispatch.jobs.arch import arch, capability, sm
from ...dispatch.jobs.vocabulary import OfferFilter
from ...enums import Pricing
from ...fleet.reliability import by_kind, reliability, window
from ...holding.holds import Cards, Holding, Holds
from ...holding.line import Line
from ...log import logger
from ...policy.grants import Grants, GrantsFile
from ...render import Output, progress
from ..session import Verbs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ...render.values import Node
    from ...state.storage.holds import Held

# The catalogs mb rents on, by the `host hold` provider each rents through.
_RENTABLE = {
    "vastai": "vast",
    "runpod": "runpod",
    "lambdalabs": "lambda",
    "hpc-ai": "hpc-ai",
    "digitalocean": "digitalocean",
    "verda": "verda",
    "vultr": "vultr",
    "hotaisle": "hotaisle",
    "hyperstack": "hyperstack",
    "shadeform": "shadeform",
}
_HPCAI = "hpc-ai"
# The clouds that list their own offers, gpuhunt publishing no catalog of them.
_SELF_LISTING = (HyperstackBackend, ShadeformBackend)
_APART = frozenset({_HPCAI, *(cloud.name for cloud in _SELF_LISTING)})


@Parameter(name="*")
class Shopping(FrozenModel):
    """Which offers `host offers` lists."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    arch: str = ""
    """only cards of these compute capabilities: `sm_120`, `sm_100`, `sm_90+` (that or newer),
    `8.9`, or a family (`ampere`, `ada`, `hopper`, `blackwell`, `blackwell-dc` for sm_100/103,
    `blackwell-rtx` for sm_120/121)."""
    count: int = 1
    """cards per machine."""
    max_usd_hr: float = 0.0
    """the most an hour may cost, 0 for any."""
    pricing: Pricing = Pricing.ANY
    """`any` offer, only `spot` (interruptible) or only `on-demand` ones."""
    provider: tuple[str, ...] = ()
    """only these catalogs (`vastai`, `runpod`, `lambdalabs`, `aws`...)."""

    @field_validator("pricing", mode="before")
    @classmethod
    def _spelled(cls, value: str | Pricing) -> str | Pricing:
        """`on-demand`, the spelling the help prints, for the member `on_demand`."""
        return value.replace("-", "_") if isinstance(value, str) else value

    def wanted(self, gpu: str) -> OfferFilter:
        """The offers these flags and the card `gpu` ask for."""
        return OfferFilter(
            gpu_name=gpu,
            gpus=self.count,
            arch=arch(self.arch) if self.arch else None,
            pricing=self.pricing,
            max_usd_hr=self.max_usd_hr,
        )


class _Listed(FrozenModel):
    """One offer as `host offers` prints it, or a catalog that would not list, saying why.

    delivered: the share of the last 30 days' dispatches that ran, from the run registry.
    useful_usd_hr: dollars per useful GPU hour at the landing share the lake measured, which is
        what the rows rank by (`Scorecard.RULE`); empty for a provider never measured, which still
        ranks at its price.
    landed: that landing share (Laplace smoothed), ready_s the median seconds to ssh answering.
    cap: what bounds the provider's bill, the bound every lease of it records.
    """

    provider: str
    gpu: str = ""
    sm: str = ""
    count: int = 0
    usd_hr: float | None = None
    spot: bool = False
    region: str = ""
    instance: str = ""
    stock: str = ""
    rent: str = ""
    delivered: float | None = None
    useful_usd_hr: float | None = None
    landed: float | None = None
    ready_s: float | None = None
    cap: str = ""

    @property
    def offer(self) -> Offer | None:
        """The priced offer this row lists, None for a catalog that would not list."""
        if self.usd_hr is None:
            return None
        return Offer(
            provider=self.provider,
            gpu=self.gpu,
            gpu_count=self.count,
            spot=self.spot,
            region=self.region,
            rate_usd_hr=self.usd_hr,
        )


# The verbs' defaults: compact output, every card in every catalog, the cards a profile names.
_COMPACT = Output()
_SHOPPING = Shopping()
_CARDS = Cards()
# The rank of a catalog that would not list: after every priced offer.
_UNPRICED: Rank = (inf, 0.0, 0.0)


class RentalVerbs(Verbs):
    """Machines rented and kept: what the clouds offer, holding one, releasing it."""

    @staticmethod
    def _hpcai(wanted: OfferFilter, providers: Sequence[str]) -> list[_Listed]:
        """HPC-AI's GPU types as offer rows, out-of-stock ones marked; none without its key.

        gpuhunt carries no HPC-AI catalog, and theirs is small enough to list whole (whole 8-card
        nodes), so a type out of stock still shows what it would cost, and a catalog that would
        not list is a row saying why.
        """
        if (providers and _HPCAI not in providers) or not RentalVerbs._keyed(_HPCAI):
            return []
        try:
            types = HpcAiBackend().api.types()
        except (MissionError, OSError) as unlisted:
            return [_Listed(provider=_HPCAI, stock=f"unlisted: {unlisted}")]
        return [
            _Listed(
                provider=_HPCAI,
                gpu=kind.card,
                sm=sm(capability(kind.card)),
                count=kind.gpus,
                usd_hr=kind.usd_hr,
                spot=kind.spot,
                region=kind.region,
                instance=kind.instance_type_id,
                stock="" if kind.in_stock else "out",
            )
            for kind in types
            if kind.gpu and kind.fits(wanted)
        ]

    @staticmethod
    def _hunted(wanted: OfferFilter, providers: Sequence[str]) -> list[_Listed]:
        """gpuhunt's offers as rows, unless only clouds it does not carry were asked."""
        others = [name for name in providers if name not in _APART]
        if providers and not others:
            return []
        return [
            _Listed(
                provider=item.provider,
                gpu=item.gpu_name or "cpu",
                sm=sm(capability(item.gpu_name or "cpu")),
                count=item.gpu_count,
                usd_hr=round(item.price, 4),
                spot=bool(item.spot),
                region=item.location,
                instance=item.instance_name,
            )
            for item in hunt(wanted, providers=others)
            if item.provider not in _APART
        ]

    @staticmethod
    def _keyed(kind: str) -> bool:
        """Whether the provider backend registered as `kind` has its credentials here."""
        backend = ProviderBackend.find(kind)()
        return isinstance(backend, Account) and backend.standing().keyed

    @staticmethod
    def _own(wanted: OfferFilter, providers: Sequence[str]) -> list[_Listed]:
        """The offers of the clouds that list themselves, gpuhunt carrying none: those keyed here.

        A cloud whose listing fails is a row saying why, so one outage hides no other price.
        """
        rows: list[_Listed] = []
        for cloud in _SELF_LISTING:
            if (providers and cloud.name not in providers) or not RentalVerbs._keyed(cloud.name):
                continue
            try:
                offers = cloud().catalog(gpu_name=wanted.gpu_name, gpus=wanted.gpus)
            except (MissionError, OSError) as unlisted:
                rows.append(_Listed(provider=cloud.name, stock=f"unlisted: {unlisted}"))
                continue
            rows += [
                _Listed(
                    provider=offer.provider,
                    gpu=offer.gpu,
                    sm=sm(capability(offer.gpu)),
                    count=offer.gpu_count,
                    usd_hr=round(offer.rate_usd_hr, 4),
                    spot=offer.spot,
                    region=offer.region,
                    instance=offer.source.rsplit(":", 1)[-1],
                )
                for offer in offers
                if wanted.admits(offer)
            ]
        return rows

    @staticmethod
    def _printed(held: Held, output: Output, *, title: str) -> None:
        """Print one held machine: its alias, where it came from, what it costs, when it ends."""
        payload: dict[str, Node] = {
            "alias": held.alias,
            "provider": held.provider,
            "handle": held.handle,
            "gpu": held.gpu,
            "usd_hr": held.usd_hr,
            "deadline": held.deadline.isoformat(),
            "root": held.profile.root,
        }
        output.print_record(payload, title=title)

    @staticmethod
    def _ranked(found: Sequence[_Listed], *, scorecard: Scorecard) -> list[_Listed]:
        """`found` scored and best first, equal ranks keeping the order the catalogs listed."""
        logger.info("ranked by {}", Scorecard.RULE)
        scored = [RentalVerbs._scored(listed, scorecard=scorecard) for listed in found]
        return [listed for _, listed in sorted(scored, key=itemgetter(0))]

    @staticmethod
    def _rentable(
        listed: _Listed,
        *,
        keys: Mapping[str, bool],
        delivered: Mapping[str, float],
        grants: Grants,
    ) -> _Listed:
        """`listed` with the provider `host hold` rents it through, the share that ran and its cap.

        The provider reads `keyed` when its key is here; the share is of the dispatches to that
        provider that started and ran to an exit.
        """
        kind = _RENTABLE.get(listed.provider, "")
        rent = f"{kind} keyed" if kind and keys.get(kind) else kind
        return listed.model_copy(
            update={
                "rent": rent,
                "delivered": delivered.get(kind),
                "cap": RentalVerbs.capped(kind, grants=grants),
            }
        )

    @staticmethod
    def _scored(listed: _Listed, *, scorecard: Scorecard) -> tuple[Rank, _Listed]:
        """`listed` with its useful price and landing record, and the key the rows rank by.

        A catalog that would not list carries no price and ranks last. An offer of a provider
        never measured is priced at the prior (`Scorecard.RULE`) but shows no measurement.
        """
        offer = listed.offer
        if offer is None:
            return _UNPRICED, listed
        kind = _RENTABLE.get(listed.provider, "")
        if kind:
            offer = ProviderBackend.find(kind).billed(offer)
        score = scorecard.score(offer, kind=kind)
        track = score.track
        if not track.landings:
            return score.order, listed
        ready = track.ready_p50_s
        measured = {
            "useful_usd_hr": round(score.useful_usd_hr, 4),
            "landed": round(track.delivered, 3),
            "ready_s": None if ready is None else round(ready, 1),
        }
        return score.order, listed.model_copy(update=measured)

    @staticmethod
    def capped(kind: str, *, grants: Grants, balance_usd: float | None = None) -> str:
        """What bounds provider `kind`'s bill, as every lease of it records, and what is left.

        A card shows its declared limit, or says none is declared and that a rent is then
        refused; a prepaid provider shows its balance once it was read; a threshold provider is
        bounded rental by rental, so it shows the bound alone. Empty for a name that is no
        provider (a host, a rental).

        grants: the owner's grants, which hold the card limits.
        balance_usd: the balance the provider reports (`Standing.credit_usd`), None unread.
        """
        if kind not in ProviderBackend.names():
            return ""
        backend = ProviderBackend.find(kind)
        try:
            cap = backend.cap(card_usd=grants.card(kind))
        except MissionError:
            return f"{backend.bound}, no limit declared"
        left = cap.remaining(balance_usd=balance_usd)
        return (
            str(cap.bound) if left is None or not isfinite(left) else f"{cap.bound} ${left:,.2f}"
        )

    @staticmethod
    def granted(root: Path) -> Grants:
        """The owner's grants of the workspace at `root`, none while its file is refused."""
        try:
            return GrantsFile(root).read()
        except MissionError as refused:
            logger.warning("no card limit is shown, the grants file was refused: {}", refused)
            return Grants()

    def hold(
        self,
        provider: str,
        *,
        holding: Holding,
        cards: Cards = _CARDS,
        env: str = "",
        output: Output = _COMPACT,
    ) -> None:
        """Rent a machine and keep it as an ssh host until a deadline, set up and ready for jobs.

        A rental per job rebuilds the environment every time; a held machine is set up once and
        then takes `submit --on <alias>` in seconds. The machine gets an alias in the ssh config,
        onboards like `setup`, and is recorded with its deadline, which `monitor` and `compute`
        both enforce by releasing it, so a forgotten hold stops billing on time.

        On a PBS host (`miyabi-g`) it instead keeps the user's one interactive allocation for the
        time asked, renewing it as each walltime ends, and runs submitted jobs in it one at a
        time: `job submit --on miyabi-g --queue held -- <command>`. The line takes the one
        interactive slot, so `shell --on miyabi-g` is refused while it is held. A line already
        held is reported, not doubled; a first allocation that does not come closes it again.

        Args:
            provider: the provider host to rent through, `vast` say, or a PBS host to keep.
            env: the environment to set up, the provider profile's own when omitted.
        """
        local = self.session.board("local")
        if (kept := Line.of(local, provider)) is not None:
            with progress(f"keeping the interactive slot on {provider}") as stage:
                opened = kept.open(holding.duration, walltime=holding.walltime, watch=stage)
            output.print_record(opened.model_dump(), title="hold")
            return
        with progress(f"holding a {cards.gpu_name or cards.arch or provider} machine") as stage:
            held = Holds(local).hold(provider, holding, cards, env=env, watch=stage)
        RentalVerbs._printed(held, output, title="hold")

    def offers(
        self,
        gpu: str = "",
        *,
        shopping: Shopping = _SHOPPING,
        limit: int = 25,
        output: Output = _COMPACT,
    ) -> None:
        """Every cloud's GPU offers, best useful hour first, and whether this tool can rent each.

        Read from gpuhunt, the open catalog dstack publishes: public price lists and live
        marketplaces (AWS, GCP, Azure, Lambda, RunPod, Vast, Verda, Nebius and more), most
        readable without an account, plus the own catalogs of HPC-AI, Hyperstack and Shadeform
        when their keys are here. `rent`
        names the `host hold` provider for a cloud mb can rent on, with `keyed` when its API key
        is present here. `sm` is the card's compute capability: `--arch sm_120` finds the
        cheapest card whose kernels match an RTX 5090's, which is not a B200's (sm_100).
        `delivered` is the share of the last 30 days' dispatches to that provider whose job
        started and ran to an exit (`host list --reliability` breaks it down).

        Rows rank by dollars per useful GPU hour (`useful_usd_hr`) for a one-hour job: the price
        per GPU, plus the setup, the release time and the provider's billed minimum, plus the
        failed landings the lake recorded for that provider, each paying its setup, spread over
        the landings that delivered (`landed`, smoothed so two landings cannot outrank forty;
        `ready_s` is the median time to ssh). Equal costs rank by `landed`, then `ready_s`. A
        provider never measured ranks at its price and shows neither. `cap` is what bounds the
        provider's bill, which a rent through it must stay under.

        Args:
            gpu: the card, as any catalog spells it (`B200`, `RTX 4090`, `H100`).
            limit: how many rows, best first.
        """
        wanted = shopping.wanted(gpu)
        with progress(f"asking every cloud for {wanted.spoken or 'any'} GPUs"):
            found = [
                *RentalVerbs._hunted(wanted, shopping.provider),
                *RentalVerbs._hpcai(wanted, shopping.provider),
                *RentalVerbs._own(wanted, shopping.provider),
            ]
        cache = self.session.board("local").dispatcher.cache
        delivered = by_kind(reliability(cache.listed(since=window(30))))
        grants = RentalVerbs.granted(self.session.workspace_root())
        keys = {kind: RentalVerbs._keyed(kind) for kind in _RENTABLE.values()}
        best = RentalVerbs._ranked(found, scorecard=Scorecard.of(Ledger(cache.session)))
        rows = [
            RentalVerbs._rentable(listed, keys=keys, delivered=delivered, grants=grants)
            for listed in best[:limit]
        ]
        output.print_rows([row.model_dump() for row in rows], title="offers")

    def release(self, alias: str, *, output: Output = _COMPACT) -> None:
        """End a held machine now: stop its billing, settle its record and drop its alias.

        On a PBS host it ends the held line instead: nothing more is claimed, jobs not yet taken
        are cancelled, a job running now finishes and keeps its results retrievable, and the idle
        allocation is freed so the interactive slot is the user's again.

        Args:
            alias: the held machine's alias, as `hold` printed it, or the PBS host.
        """
        local = self.session.board("local")
        if (kept := Line.of(local, alias)) is not None:
            with progress(f"releasing the line on {alias}"):
                output.print_record(kept.release().model_dump(), title="release")
            return
        with progress(f"releasing {alias}"):
            held = Holds(local).release(alias)
        RentalVerbs._printed(held, output, title="release")
