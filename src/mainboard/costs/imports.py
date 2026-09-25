from typing import TYPE_CHECKING, Protocol

from .catalog import Offer

if TYPE_CHECKING:
    from collections.abc import Iterable

# gpuhunt's spelling of a provider this workspace names otherwise. Vast is the only collision
# today (`runpod`, `lambdalabs` and the rest agree), so an explicit map beats an alias mechanism.
_FEED_NAMES = {"vastai": "vast"}


class CatalogRow(Protocol):
    """One `gpuhunt.query` row, typed structurally so the optional dependency never imports."""

    provider: str
    gpu_name: str
    gpu_count: int
    spot: bool
    location: str | None
    price: float | None


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


def from_gpuhunt(rows: Iterable[CatalogRow]) -> list[Offer]:
    """gpuhunt catalog rows as `Offer`s, tagged imported and named as this workspace names them."""
    return [
        Offer(
            provider=catalog_provider(str(row.provider)),
            gpu=str(row.gpu_name),
            gpu_count=int(row.gpu_count),
            spot=bool(row.spot),
            region=str(row.location or ""),
            rate_usd_hr=float(price),
            source="imported:gpuhunt",
        )
        for row in rows
        if (price := row.price) is not None
    ]
