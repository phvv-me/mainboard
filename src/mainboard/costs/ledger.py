import statistics
from typing import TYPE_CHECKING

from patos import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path


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
    """Append-only NDJSON observations, one line per dispatched job in `root/costs.ndjson`.

    Read whole at fit time, since a year of dispatches stays small.
    """

    def __init__(self, root: Path) -> None:
        self.path = root / "costs.ndjson"

    def observations(self, *, provider: str = "", gpu: str = "") -> list[Observation]:
        """Every recorded observation, an empty filter matching all."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        rows = (Observation.model_validate_json(line) for line in lines if line.strip())
        return [
            row
            for row in rows
            if (not provider or row.provider == provider) and (not gpu or row.gpu == gpu)
        ]

    def record(self, observation: Observation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(observation.model_dump_json() + "\n")


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
