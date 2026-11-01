from typing import TYPE_CHECKING

from .catalog import Offer

if TYPE_CHECKING:
    from collections.abc import Iterable


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
