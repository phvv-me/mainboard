# Offers come from gpuhunt, the catalog dstack publishes as its own library (MPL-2.0): public
# price lists and live marketplaces across AWS, GCP, Azure, Lambda, RunPod, Vast, Verda, Nebius
# and more, most of them readable without an account. It is the same row whichever cloud rents
# it, so `mb host offers` compares every cloud even where mb cannot rent yet. This module is the
# one adapter over it, so the one name read from its internals stays here.

import logging
from typing import TYPE_CHECKING

import gpuhunt
from gpuhunt._internal.catalog import OFFLINE_PROVIDERS

from ....costs.catalog import Offer
from ....log import logger
from ...jobs.arch import capability, card
from ...jobs.vocabulary import Pricing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gpuhunt import CatalogItem

    from ...jobs.vocabulary import OfferFilter


def as_offer(item: CatalogItem) -> Offer:
    """A gpuhunt row as the offer every lease and estimate prices."""
    return Offer(
        provider=item.provider,
        gpu=item.gpu_name or "cpu",
        gpu_count=item.gpu_count or 0,
        spot=bool(item.spot),
        region=item.location or "",
        rate_usd_hr=item.price,
        source=f"gpuhunt:{item.provider}:{item.instance_name}",
    )


def hunt(wanted: OfferFilter, *, providers: Sequence[str] = ()) -> list[CatalogItem]:
    """gpuhunt's offers for the filters, cheapest first; providers with no key are skipped.

    The architecture is filtered here rather than by gpuhunt, whose table has B300 at sm_100.
    """
    # It logs a warning per catalog it cannot read offline (azure), which says nothing to act on.
    logging.getLogger("gpuhunt").setLevel(logging.ERROR)
    try:
        found = _queried(wanted, providers=list(providers) or None)
    except OSError:
        found = _answering(wanted, providers=providers)
    span = wanted.arch
    kept = (
        item
        for item in found
        if span is None or (item.gpu_name and span.holds(capability(item.gpu_name)))
    )
    return sorted(kept, key=lambda item: item.price)


def _answering(wanted: OfferFilter, *, providers: Sequence[str]) -> list[CatalogItem]:
    """The offers of every catalog that answers, asked one at a time, each silent one named.

    gpuhunt asks its catalogs together and one failing fails them all: a 502 from Vultr's API
    ended `host offers` as a traceback and hid every other cloud's prices (2026-09-30).
    """
    loaded = (provider.NAME for provider in gpuhunt.default_catalog().providers)
    found: list[CatalogItem] = []
    for name in providers or [*OFFLINE_PROVIDERS, *loaded]:
        try:
            found += _queried(wanted, providers=[name])
        except OSError as silent:
            logger.warning("{} listed no offers: {}", name, silent)
    return found


def _queried(wanted: OfferFilter, *, providers: list[str] | None) -> list[CatalogItem]:
    """gpuhunt's offers for the filters from `providers`, every catalog's when None."""
    gpus = wanted.gpus or None
    return gpuhunt.query(
        provider=providers,
        gpu_name=[card(wanted.gpu_name)] if wanted.gpu_name else None,
        min_gpu_count=gpus,
        max_gpu_count=gpus,
        max_price=wanted.max_usd_hr or None,
        spot=None if wanted.pricing is Pricing.ANY else wanted.pricing is Pricing.SPOT,
    )
