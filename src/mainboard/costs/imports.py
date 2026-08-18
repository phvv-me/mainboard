from typing import TYPE_CHECKING

from .catalog import Offer

if TYPE_CHECKING:
    from collections.abc import Iterable

# gpuhunt's own spelling for a provider this workspace already names something else. Vast is the
# only collision in the feed today (`runpod`, `lambdalabs` and the rest already agree), so the
# reconciliation stays an explicit map rather than an alias mechanism nothing else would use.
_FEED_NAMES = {"vastai": "vast"}


def from_vast(rows: Iterable[dict], *, spot: bool = False) -> list[Offer]:
    """Vast.ai offer rows as `Offer`s, tagged probed.

    The authed counterpart to the imported price feed: a live `/bundles` search already knows
    whether each machine is rentable right now, which is the one thing a scraped catalog can
    never tell the router.

    rows: the offer dicts a Vast offer search returns.
    spot: whether the search asked for interruptible capacity, which prices every row at its
        bid floor rather than its on-demand total.
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
    """The catalog's own name for the provider gpuhunt calls `name`.

    gpuhunt spells Vast `vastai` while the live probe, the host kind and the backend all spell
    it `vast`, and a catalog query narrows by exactly one name, so the two feeds are reconciled
    here at the import seam instead of every reader learning both spellings.

    name: the provider id as the imported row carries it.
    """
    return _FEED_NAMES.get(name, name)


def from_gpuhunt(rows: Iterable[object]) -> list[Offer]:
    """gpuhunt catalog rows as `Offer`s, tagged imported and named as this workspace names them.

    rows: objects with provider, gpu_name, gpu_count, price, spot, location,
        the shape `gpuhunt.query` returns; taken structurally so the optional
        dependency never imports here.
    """
    return [
        Offer(
            provider=catalog_provider(str(row.provider)),
            gpu=str(row.gpu_name),
            gpu_count=int(row.gpu_count),
            spot=bool(row.spot),
            region=str(row.location or ""),
            rate_usd_hr=float(row.price),
            source="imported:gpuhunt",
        )
        for row in rows
        if getattr(row, "price", None)
    ]
