from typing import TYPE_CHECKING

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, field_validator

from ...core.errors import MissionError
from ...dispatch.backends import HyperstackBackend, ShadeformBackend
from ...dispatch.backends.base import ProviderBackend
from ...dispatch.backends.capabilities import Account
from ...dispatch.backends.cloud.offers import hunt
from ...dispatch.backends.hpcai import HpcAiBackend
from ...dispatch.jobs.arch import arch, capability, sm
from ...dispatch.jobs.vocabulary import OfferFilter, Pricing
from ...fleet.reliability import by_kind, reliability, window
from ...holding.holds import Cards, Holding, Holds
from ...holding.line import Line
from ...render import progress
from ..base import Verbs
from ..options import Output

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    """One offer as `host offers` prints it, or a catalog that would not list, saying why."""

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


# The verbs' defaults: compact output, every card in every catalog, the cards a profile names.
_COMPACT = Output()
_SHOPPING = Shopping()
_CARDS = Cards()


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
    def _rentable(
        listed: _Listed, keys: Mapping[str, bool], delivered: Mapping[str, float]
    ) -> _Listed:
        """`listed` with the provider `host hold` rents it through and the share that ran.

        The provider reads `keyed` when its key is here; the share is of the dispatches to that
        provider that started and ran to an exit.
        """
        kind = _RENTABLE.get(listed.provider, "")
        rent = f"{kind} keyed" if kind and keys.get(kind) else kind
        return listed.model_copy(update={"rent": rent, "delivered": delivered.get(kind)})

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
        """Every cloud's GPU offers, cheapest first, and whether this tool can rent each here.

        Read from gpuhunt, the open catalog dstack publishes: public price lists and live
        marketplaces (AWS, GCP, Azure, Lambda, RunPod, Vast, Verda, Nebius and more), most
        readable without an account, plus the own catalogs of HPC-AI, Hyperstack and Shadeform
        when their keys are here. `rent`
        names the `host hold` provider for a cloud mb can rent on, with `keyed` when its API key
        is present here. `sm` is the card's compute capability: `--arch sm_120` finds the
        cheapest card whose kernels match an RTX 5090's, which is not a B200's (sm_100).
        `delivered` is the share of the last 30 days' dispatches to that provider whose job
        started and ran to an exit (`host list --reliability` breaks it down).

        Args:
            gpu: the card, as any catalog spells it (`B200`, `RTX 4090`, `H100`).
            limit: how many rows, cheapest first.
        """
        wanted = shopping.wanted(gpu)
        with progress(f"asking every cloud for {wanted.spoken or 'any'} GPUs"):
            found = [
                *RentalVerbs._hunted(wanted, shopping.provider),
                *RentalVerbs._hpcai(wanted, shopping.provider),
                *RentalVerbs._own(wanted, shopping.provider),
            ]
        runs = self.session.board("local").dispatcher.cache.listed(since=window(30))
        delivered = by_kind(reliability(runs))
        keys = {kind: RentalVerbs._keyed(kind) for kind in _RENTABLE.values()}
        # A catalog that would not list carries no price and goes last.
        cheapest = sorted(found, key=lambda listed: (listed.usd_hr is None, listed.usd_hr or 0.0))
        rows = [RentalVerbs._rentable(listed, keys, delivered) for listed in cheapest[:limit]]
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
