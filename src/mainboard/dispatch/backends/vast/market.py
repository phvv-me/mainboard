"""Vast's market: offer searches under the house floors, the pick, and why nothing was rentable."""

from typing import TYPE_CHECKING, NoReturn

from patos import FrozenModel

from ....core.errors import MissionError
from ...jobs.arch import Arch
from ..base import ProviderBackend, image_cuda

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .api import VastApi

# The images an uncontainerized plan rents under, Vast's base image at each CUDA minor the house
# toolchain spans, oldest first. A rental takes the newest its offer's driver loads, so THE OFFER
# FILTER'S FLOOR IS THE OLDEST REFERENCE HERE and written nowhere else. One pinned image failed
# both ways: 13.3.1 with the floor at 13.0 rented hosts whose driver tops out at 13.0, where the
# container never started and each rental was destroyed after the whole address wait (RTX 5090
# 51865823, RTX 5080 51869485 and 51891087, 2026-09-21); the floor raised to 13.3 then shut every
# L40S, A100 and H100 host (13.0 to 13.2 drivers) out of the market (2026-09-25). The workspace
# itself declares CUDA 13.0 (`[system]`).
_BASE_IMAGES = (
    "vastai/base-image:cuda-13.0.3-auto",
    "vastai/base-image:cuda-13.1.2-auto",
    "vastai/base-image:cuda-13.2.1-auto",
    "vastai/base-image:cuda-13.3.1-auto",
)
# Bundle-row fields every offer publishes, so the floors filter before renting: the highest CUDA
# the driver loads, the compute capability (`750` for `sm_75`), and the measured download in Mbps.
_CUDA_FIELD = "cuda_max_good"
_CAPABILITY_FIELD = "compute_cap"
_DOWNLOAD_FIELD = "inet_down"
# How far above the floor an interruptible rental bids, `bid` says why.
_BID_MARGIN = 1.05
# Offers one search asks for; the query already orders by price, so this bounds the reply size.
_SEARCH_LIMIT = 32


def _cuda_max_good(offer: Mapping) -> float:
    """The highest CUDA version `offer`'s driver loads, 0.0 when unpublished.

    Silence never satisfies a floor, here or in `capability` and `download`: an unknown driver is
    exactly the offer that hands back a contract id and no instance.
    """
    try:
        return float(offer[_CUDA_FIELD])
    except KeyError, TypeError, ValueError:
        return 0.0


def _download(offer: Mapping) -> float:
    """The host's download rate in Mbps, 0 when unpublished."""
    try:
        return float(offer.get(_DOWNLOAD_FIELD) or 0.0)
    except TypeError, ValueError:
        return 0.0


def _capability(offer: Mapping) -> int:
    """`offer`'s compute capability, `750` for `sm_75`, 0 when unpublished."""
    try:
        return int(offer[_CAPABILITY_FIELD])
    except KeyError, TypeError, ValueError:
        return 0


def _describe(offer: Mapping, gpu_name: str) -> str:
    """`offer` as the phrase a refusal names it by, id first so it can be looked up.

    gpu_name: the card asked for, standing in when the row names none.
    """
    where = str(offer.get("geolocation") or "").strip(" ,")
    card = offer.get("gpu_name") or gpu_name or "unknown card"
    return f"offer {offer.get('id')} ({card}, {where or 'unknown location'})"


def bid(offer: Mapping) -> float:
    """What an interruptible rental of `offer` bids per hour: just above its floor.

    A bid of exactly `min_bid` is refused as `no_such_ask`, the ask "no longer available", while
    one a little above it rents the same machine (RTX 5060 Ti offer 53231094, $0.067 refused and
    $0.08 accepted, 2026-09-29), so every spot rent paid for six refusals before this.
    """
    return round(float(offer["min_bid"]) * _BID_MARGIN + 0.001, 4)


def base_image(offer: Mapping) -> str:
    """The newest base image `offer`'s driver loads, or the oldest when it names no version."""
    loadable = [
        image for image in _BASE_IMAGES if (image_cuda(image) or 0) <= _cuda_max_good(offer)
    ]
    return loadable[-1] if loadable else _BASE_IMAGES[0]


class Ask(FrozenModel):
    """What one search of the market asks for.

    gpu_name: the Vast GPU name (`RTX 4090`), underscores read as spaces, empty for any.
    gpus: the cards per machine, 0 for any.
    max_usd_hr: an hourly total-price ceiling, 0 for none.
    cuda: the CUDA the image about to be rented names, raising the driver floor above the base
        images'; 0 for the base images'.
    arch: the compute capabilities the card must have, asked of Vast's own `compute_cap`.
    """

    gpu_name: str = ""
    gpus: int = 0
    max_usd_hr: float = 0.0
    cuda: float = 0.0
    arch: Arch | None = None


class VastMarket:
    """Rentable offers under the house floors, ranked by what this market's pricing mode pays.

    spot: whether rentals bid for interruptible capacity instead of renting on demand.
    disk_gb: local disk per rental, also the storage an offer search is priced at.
    """

    # Derived from the oldest base image, never below the house floor, so the filter and the
    # images cannot disagree again.
    CUDA_FLOOR = max(
        ProviderBackend.CUDA_FLOOR, min(image_cuda(image) or 0 for image in _BASE_IMAGES)
    )

    # A cold rental pulls a multi-gigabyte image before its container exists: hosts below this
    # spent the whole address wait pulling and ended with no container (three rentals,
    # 2026-09-12), while hosts at a few Gbps were running inside two minutes.
    DOWNLOAD_FLOOR_MBPS = 500.0

    def __init__(self, api: VastApi, *, spot: bool, disk_gb: float) -> None:
        self.api = api
        self.spot = spot
        self.disk_gb = disk_gb

    def __query(self, ask: Ask, *, limit: int, floored: bool) -> dict:
        """The `{field: {operator: value}}` table one search sends.

        The four constant filters are the ones their console applies, keeping unverified hosts,
        resold capacity and rented machines out.
        """
        query: dict = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "type": "bid" if self.spot else "on-demand",
            "order": [["dph_total", "asc"]],
            "allocated_storage": self.disk_gb,
            "limit": limit,
            **({"gpu_name": {"eq": ask.gpu_name.replace("_", " ")}} if ask.gpu_name else {}),
            **({"num_gpus": {"eq": ask.gpus}} if ask.gpus else {}),
            **({"dph_total": {"lte": ask.max_usd_hr}} if ask.max_usd_hr else {}),
        }
        low, high = ask.arch.vast() if ask.arch else (0, 0)
        if floored:
            query[_CUDA_FIELD] = {"gte": max(self.CUDA_FLOOR, ask.cuda)}
            query[_DOWNLOAD_FIELD] = {"gte": self.DOWNLOAD_FLOOR_MBPS}
        if floored or ask.arch:
            floor = ProviderBackend.CAPABILITY_FLOOR if floored else 0
            query[_CAPABILITY_FIELD] = {
                "gte": max(floor, low),
                **({"lte": high} if ask.arch else {}),
            }
        return query

    def __unrentable(self, raw: Sequence[Mapping], ask: Ask, *, card: str, ceiling: str) -> str:
        """Why nothing in `raw`, the unfloored market, was rentable.

        The answer names the first floor that turned every offer away.

        card: the machine asked for, spelled for the message (`1x RTX 4090 spot`).
        ceiling: its price ceiling, spelled the same way, empty for none.
        """
        floor = max(self.CUDA_FLOOR, ask.cuda)
        loadable = [row for row in raw if _cuda_max_good(row) >= floor]
        buildable = [
            row for row in loadable if _capability(row) >= ProviderBackend.CAPABILITY_FLOOR
        ]
        if not loadable:
            best = max(raw, key=_cuda_max_good)
            return (
                f"vast has {len(raw)} rentable {card} offer(s){ceiling} and not one driver "
                f"reaches CUDA {floor}, "
                f"the floor this rental needs. The best is {_describe(best, ask.gpu_name)} at "
                f"CUDA {_cuda_max_good(best)}. Renting it would hand back a contract id and no "
                "instance, because vast destroys a container its driver cannot start. Ask for a "
                "card whose hosts run a newer driver."
            )
        if not buildable:
            best = max(loadable, key=_capability)
            return (
                f"vast has {len(loadable)} rentable {card} offer(s){ceiling} whose driver reaches "
                f"CUDA {floor}, "
                "and not one is an architecture that CUDA still builds for. The best is "
                f"{_describe(best, ask.gpu_name)} at compute capability {_capability(best)}, "
                f"below the sm_{ProviderBackend.CAPABILITY_FLOOR // 10} floor. Renting it would "
                "boot, bill, and die at the first kernel launch with no kernel image for its own "
                "card. Maxwell, Pascal and Volta went with it; ask for Turing or newer."
            )
        best = max(buildable, key=_download)
        return (
            f"vast has {len(buildable)} rentable {card} offer(s){ceiling} this house could run "
            "on, and not one "
            f"host downloads at {self.DOWNLOAD_FLOOR_MBPS:.0f} Mbps or more. The best is "
            f"{_describe(best, ask.gpu_name)} at {_download(best):.0f} Mbps. Renting it would "
            "spend the whole landing window pulling the image and end with no container, which "
            "is how three rentals went on 2026-09-12; raise the ceiling to reach a faster host."
        )

    def best(self, offers: Sequence[Mapping]) -> dict:
        """The most reliable offer of a page, ties going to the cheaper machine."""
        chosen = max(offers, key=lambda offer: (float(offer["reliability2"]), -self.rate(offer)))
        return dict(chosen)

    def pick(self, ask: Ask) -> dict:
        """The offer to rent: the most reliable machine the budget and the CUDA floors allow.

        Renting the cheapest listing put earlier rentals at the bottom of the market, where the
        container is billed for and never starts, so price only admits: the pick is the most
        reliable machine on the cheapest page under the cap, landing mid-market.
        """
        offers = self.search(ask)
        if not offers:
            self.refuse(ask)
        return self.best(offers)

    def rate(self, offer: Mapping) -> float:
        """What one hour of `offer` costs under this market's pricing mode."""
        return bid(offer) if self.spot else float(offer["dph_total"])

    def refuse(self, ask: Ask) -> NoReturn:
        """Say why nothing was rentable, naming whichever floor turned the market away.

        An empty page reads the same for an empty market and one this house aged out of (the
        latter cost five dispatches), so one more search, unfloored and paid on this path alone,
        tells them apart. The driver is asked first, being the earlier failure.
        """
        ceiling = f" under ${ask.max_usd_hr:.2f}/hr" if ask.max_usd_hr else ""
        named = ask.gpu_name or (ask.arch.spelled if ask.arch else "any")
        card = f"{ask.gpus}x {named}{' spot' if self.spot else ''}"
        raw = self.search(ask, floored=False)
        if not raw:
            raise MissionError(f"vast has no rentable {card} offer{ceiling} right now")
        raise MissionError(self.__unrentable(raw, ask, card=card, ceiling=ceiling))

    def search(self, ask: Ask, *, limit: int = 0, floored: bool = True) -> list[dict]:
        """Rentable offers matching `ask`, cheapest first under this market's pricing mode.

        Vast ranks by on-demand total in either mode, so a spot search is re-ranked by the bid
        floor it will pay. The floors ride on the wire, since the reply is paged by price and
        filtering afterwards would judge the market on its cheapest 32 rows.

        floored: whether the house floors ride on the query; only `refuse` drops them, to ask the
            raw market what the floors turned away.
        """
        query = self.__query(ask, limit=limit or _SEARCH_LIMIT, floored=floored)
        offers = self.api.request("POST", path="/bundles/", body=query).get("offers") or []
        # The service has returned rows above the requested ceiling, and the quoted rate, not a
        # filter the request carried, decides whether spending is authorized.
        if ask.max_usd_hr:
            offers = [offer for offer in offers if self.rate(offer) <= ask.max_usd_hr]
        return sorted(offers, key=self.rate)
