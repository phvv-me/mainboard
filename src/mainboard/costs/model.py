import math

from patos import FrozenModel


class BillingModel(FrozenModel):
    """One provider's billing mechanics, the static half of the cost equation.

    The dynamic half, how long setup and drain take, is fitted from observations rather than
    believed from marketing pages, since a per-second rate means little when provisioning costs
    minutes.
    """

    provider: str
    gpu: str = ""
    rate_usd_hr: float
    granularity_s: int = 1
    minimum_s: int = 0
    fees_usd: float = 0.0

    def billed_seconds(self, *, setup_s: float, run_s: float, drain_s: float = 0.0) -> int:
        """The seconds the provider charges for one job's wall time.

        setup_s: provisioning before the command runs.
        drain_s: teardown still billed after the command ends.
        """
        wall = setup_s + run_s + drain_s
        return max(self.minimum_s, math.ceil(wall / self.granularity_s) * self.granularity_s)

    def cost_usd(self, *, setup_s: float, run_s: float, drain_s: float = 0.0) -> float:
        seconds = self.billed_seconds(setup_s=setup_s, run_s=run_s, drain_s=drain_s)
        return self.fees_usd + self.rate_usd_hr * seconds / 3600.0
