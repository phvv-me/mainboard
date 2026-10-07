import statistics
from typing import TYPE_CHECKING

from patos import FrozenModel
from sqlalchemy import literal_column, select

from ..state import schema

if TYPE_CHECKING:
    from ..state.lake import Session


class Observation(FrozenModel):
    """One dispatched job's measured platform behavior, the fitting datum.

    Timestamps are epoch seconds; `billed_usd` stays zero until a provider API reports the real
    charge, letting fits calibrate against truth instead of inference.
    """

    provider: str
    gpu: str = ""
    region: str = ""
    t_submit: float
    t_running: float = 0.0
    t_ended: float = 0.0
    billed_usd: float = 0.0

    @property
    def run_s(self) -> float:
        """Command wall seconds, running to ended, zero when never observed."""
        if not self.t_running or not self.t_ended:
            return 0.0
        return max(0.0, self.t_ended - self.t_running)

    @property
    def setup_s(self) -> float:
        """Provisioning seconds, submit to running, zero when never observed."""
        if not self.t_running:
            return 0.0
        return max(0.0, self.t_running - self.t_submit)


class Ledger:
    """The workspace lake's `costs` table, one observation per dispatched job.

    Read whole at fit time, since a year of dispatches stays small.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def observations(self, *, provider: str = "", gpu: str = "") -> list[Observation]:
        """Every recorded observation in the order recorded, an empty filter matching all."""
        fields = tuple(Observation.model_fields)
        costs = schema.costs
        matching = {costs.c.provider: provider, costs.c.gpu: gpu}
        rows = self.session.rows(
            select(*(costs.c[name] for name in fields))
            .where(*(column == value for column, value in matching.items() if value))
            .order_by(literal_column("rowid"))
        )
        return [
            Observation.model_validate(
                {key: value for key, value in zip(fields, row, strict=True) if value is not None}
            )
            for row in rows
        ]

    def record(self, observation: Observation) -> None:
        self.session.append(schema.costs, [observation.model_dump()])


class SetupFit(FrozenModel):
    """A provider's fitted setup-time behavior, the stochastic half of cost."""

    provider: str
    gpu: str = ""
    samples: int
    mean_s: float
    p50_s: float
    p90_s: float

    @classmethod
    def from_ledger(cls, ledger: Ledger, *, provider: str, gpu: str = "") -> SetupFit | None:
        """Fit the setup distribution of `provider` (and `gpu` if given), None below 3 samples."""
        setups = [
            row.setup_s
            for row in ledger.observations(provider=provider, gpu=gpu)
            if row.setup_s > 0.0
        ]
        if len(setups) < 3:
            return None
        quantiles = statistics.quantiles(setups, n=10, method="inclusive")
        return cls(
            provider=provider,
            gpu=gpu,
            samples=len(setups),
            mean_s=statistics.fmean(setups),
            p50_s=statistics.median(setups),
            p90_s=quantiles[8],
        )
