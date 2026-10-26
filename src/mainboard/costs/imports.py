from typing import TYPE_CHECKING

from .catalog import Offer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gpuhunt import CatalogItem

# gpuhunt's spelling of a provider this workspace names otherwise. Vast is the only collision
# today (`runpod`, `lambdalabs` and the rest agree), so an explicit map beats an alias mechanism.
_FEED_NAMES = {"vastai": "vast"}


def from_vast(rows: Iterable[dict], *, spot: bool = False) -> list[Offer]:
    """Vast.ai offer search rows as `Offer`s, tagged probed.

    A live `/bundles` search knows whether each machine is rentable right now, the one thing a
    scraped catalog can never tell the router.

    spot: the search asked for interruptible capacity, pricing every row at its bid floor rather
        than its on-demand total.
    """
    return [
        Offer(
            provider="vast",
            gpu=str(row["gpu_name"]),
            gpu_count=int(row["num_gpus"]),
            spot=spot,
            region=str(row.get("geolocation") or ""),
            rate_usd_hr=float(row["min_bid"] if spot else row["dph_total"]),
            available=row.get("rentable"),
            source="probed:vast",
        )
        for row in rows
    ]


def catalog_provider(name: str) -> str:
    """The catalog's name for the provider gpuhunt calls `name` (`vastai` -> `vast`).

    A catalog query narrows by exactly one name, so the feeds are reconciled here at the import
    seam instead of every reader learning both spellings.
    """
    return _FEED_NAMES.get(name, name)


def from_gpuhunt(rows: Iterable[CatalogItem]) -> list[Offer]:
    """gpuhunt catalog rows as `Offer`s, tagged imported and named as this workspace names them.

    A row with no GPU (a CPU-only offer) or no price is left out.
    """
    return [
        Offer(
            provider=catalog_provider(row.provider),
            gpu=gpu,
            gpu_count=row.gpu_count,
            spot=row.spot,
            region=row.location or "",
            rate_usd_hr=float(price),
            source="imported:gpuhunt",
        )
        for row in rows
        if (gpu := row.gpu_name) and (price := row.price) is not None
    ]
