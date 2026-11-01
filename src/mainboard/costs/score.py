# The scorecard: what an offer costs per hour of useful GPU work, from the landings the lake kept.
#
# A listed rate prices a machine that works. A rental that never lands (a wrong card, a download
# too slow to start the job, a host that never answers) still bills its setup, and a provider that
# bills a minimum charges the whole minimum for a five-minute job, so the rate alone ranks the
# cheapest offer first and the one that delivers last. Every function here is pure over rows the
# lake already holds (`costs`), with no network call and no provider key.

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import isfinite
from typing import ClassVar

from patos import FrozenModel

from ..core.constants.units import DEFAULT_RUN_SECONDS, RELEASE_SECONDS, SECONDS_PER_HOUR
from .catalog import Offer, card_key
from .enums import Verdict
from .ledger import Ledger, Observation
from .model import BillingModel

# What a score sorts by: the useful price, the surer landing, the sooner ready.
type Rank = tuple[float, float, float]


def _quantile(values: Sequence[float], decile: int) -> float | None:
    """The `decile`-th tenth of `values`, None when there are none."""
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=10, method="inclusive")[decile - 1]


class Track(FrozenModel):
    """One provider's measured landings, for one card or for all its cards.

    gpu: the card the rows are of, empty when they are the provider's whole record.
    landings: rows whose landing said anything about the machine.
    landed: those that delivered a machine that ran the job.
    refused: failed landings whose create the provider declined, so nothing billed.
    ready_p50_s / ready_p90_s: seconds from create to ssh answering, None when never measured.
    mbps_p10: the slowest tenth of the landing download probes, None when never probed.

    PRIOR_LANDED / PRIOR_LANDINGS: the Laplace prior, one landing in two, which an offer never
    measured scores at and which a handful of rows moves only a little.
    """

    PRIOR_LANDED: ClassVar[int] = 1
    PRIOR_LANDINGS: ClassVar[int] = 2

    provider: str
    gpu: str = ""
    landings: int = 0
    landed: int = 0
    refused: int = 0
    ready_p50_s: float | None = None
    ready_p90_s: float | None = None
    mbps_p10: float | None = None

    @classmethod
    def of(cls, rows: Iterable[Observation], *, provider: str, gpu: str = "") -> Track:
        """The landings `rows` hold, timed and probed by their finite positive measurements only.

        An inf or NaN the lake kept would price every offer of the card at inf or NaN, and the
        billing arithmetic cannot round either, so such a measurement says nothing.
        """
        judged = [(row, outcome) for row in rows if (outcome := Verdict.of(row)) is not None]
        ready = [row.ready_s or row.setup_s for row, _ in judged]
        waits = [wait for wait in ready if isfinite(wait) and wait > 0]
        probes = [
            row.probe_mbps for row, _ in judged if isfinite(row.probe_mbps) and row.probe_mbps > 0
        ]
        return cls(
            provider=provider,
            gpu=gpu,
            landings=len(judged),
            landed=sum(outcome is Verdict.LANDED for _, outcome in judged),
            refused=sum(outcome is Verdict.REFUSED for _, outcome in judged),
            ready_p50_s=_quantile(waits, 5),
            ready_p90_s=_quantile(waits, 9),
            mbps_p10=_quantile(probes, 1),
        )

    @property
    def billed_share(self) -> float:
        """The share of failed landings that billed, 1 while none failed (assume the worse)."""
        failed = self.landings - self.landed
        return (failed - self.refused) / failed if failed else 1.0

    @property
    def delivered(self) -> float:
        """The Laplace share of landings that delivered, 1/2 for an offer never measured."""
        return (self.landed + self.PRIOR_LANDED) / (self.landings + self.PRIOR_LANDINGS)


class Score(FrozenModel):
    """One offer priced for one job at its provider's measured landing share.

    useful_usd_hr: dollars per useful GPU hour (`Scorecard.RULE`): the cost of one delivered job,
        failed landings included, over the GPU hours that job used.
    job_usd: what one landed attempt bills, the billed minimum and the release time included.
    billed_s: the seconds that attempt bills.
    setup_s: the provisioning seconds it was priced at, the measured median or the default.
    """

    offer: Offer
    track: Track
    useful_usd_hr: float
    job_usd: float
    billed_s: int
    setup_s: float

    @property
    def order(self) -> Rank:
        """The sort key, cheapest useful hour first, then the surer landing, then the sooner ready.

        Ties are on the cost to the millionth of a dollar, so offers of equal price rank by
        reliability and then by `setup_s`.
        """
        return (round(self.useful_usd_hr, 6), -self.track.delivered, self.setup_s)


class Scorecard:
    """The lake's `costs` rows grouped by provider, scoring offers and reporting landings.

    Read whole when built, since a year of dispatches stays small; an empty lake scores every
    offer at the prior and reports nothing.

    RULE: how offers rank, fixed before any number was seen and never the favorable one. An offer
        is worth the dollars one delivered job costs, failed landings included, per useful GPU
        hour. The landing share is the Laplace estimate (landed + 1) / (landings + 2): a rental
        that does not land still bills and the rate cannot say how often that happens, and a share
        read off two landings must not outrank one read off forty. An offer never measured scores
        at 1/2, so it ranks behind an offer of the same price whose landings mostly delivered, as
        `Catalog.quotes` already penalizes the unmeasured.
    """

    RULE: ClassVar[str] = "usd per useful GPU hour at the landing share measured for the offer"

    @classmethod
    def of(cls, ledger: Ledger) -> Scorecard:
        """The scorecard of every observation `ledger` holds."""
        return cls(ledger.observations())

    def __init__(self, observations: Iterable[Observation]) -> None:
        self.rows: dict[str, list[Observation]] = defaultdict(list)
        for row in observations:
            self.rows[row.provider].append(row)

    def report(self, kind: str) -> dict[str, float | int]:
        """The landing columns of provider `kind`, empty while it has no landing."""
        track = self.track(kind)
        if not track.landings:
            return {}
        return track.model_dump(exclude={"provider", "gpu"}, exclude_none=True)

    def score(self, offer: Offer, *, kind: str, run_s: float = DEFAULT_RUN_SECONDS) -> Score:
        """`offer` priced for a job of `run_s` command seconds, by provider `kind`'s record.

        One delivered job costs a landed attempt plus the failed attempts that came before it,
        (1 - share) / share of them per delivery, each billing its setup and release time (or
        its minimum) unless the provider refused the create.

        kind: the provider the offer is rented through, as `Observation.provider` names it.
        run_s: the command wall seconds the job needs, positive.
        """
        track = self.track(kind, gpu=offer.gpu)
        setup_s = track.ready_p50_s or BillingModel.DEFAULT_SETUP_S
        billing = offer.billing
        job_usd = billing.cost_usd(setup_s=setup_s, run_s=run_s, drain_s=RELEASE_SECONDS)
        failed_usd = billing.cost_usd(setup_s=setup_s, run_s=0.0, drain_s=RELEASE_SECONDS)
        failures = (1 - track.delivered) / track.delivered
        delivery_usd = job_usd + failures * track.billed_share * failed_usd
        gpu_hours = run_s / SECONDS_PER_HOUR * max(offer.gpu_count, 1)
        return Score(
            offer=offer,
            track=track,
            useful_usd_hr=delivery_usd / gpu_hours,
            job_usd=job_usd,
            billed_s=billing.billed_seconds(setup_s=setup_s, run_s=run_s, drain_s=RELEASE_SECONDS),
            setup_s=setup_s,
        )

    def track(self, kind: str, *, gpu: str = "") -> Track:
        """Provider `kind`'s landings on `gpu`, its whole record while none was on that card."""
        every = self.rows.get(kind, [])
        card = [row for row in every if gpu and card_key(row.gpu) == card_key(gpu)]
        return Track.of(card or every, provider=kind, gpu=gpu if card else "")
