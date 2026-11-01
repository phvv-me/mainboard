from datetime import datetime

from mainboard.costs import Offer
from mainboard.dispatch.hosts.lease import Cap, Lease
from mainboard.enums import Bound

from ..constants import DEADLINE


def lease(*, max_usd: float = 25.0, release_by: datetime = DEADLINE) -> Lease:
    """The terms accepted for a one-card rental, capped by a threshold the create carries."""
    return Lease(
        offer=Offer(provider="shadeform", gpu="H100", rate_usd_hr=3.29),
        release_by=release_by,
        max_usd=max_usd,
        cap=Cap(provider="shadeform", bound=Bound.THRESHOLD),
    )
