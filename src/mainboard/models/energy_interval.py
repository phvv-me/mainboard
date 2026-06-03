from __future__ import annotations

from .base import FrozenModel


class EnergyInterval(FrozenModel):
    """Energy consumed over a measured interval.

    energy_mj: millijoules consumed (end counter - start counter).
    duration_s: interval duration in seconds.
    peak_power_w: peak instantaneous power observed during the interval
        (only available if polling was active).
    """

    energy_mj: int = 0
    duration_s: float = 0.0
    peak_power_w: float = 0.0

    @property
    def energy_j(self) -> float:
        """Energy in joules."""
        return self.energy_mj / 1000.0

    @property
    def energy_wh(self) -> float:
        """Energy in watt-hours."""
        return self.energy_mj / 3_600_000.0

    @property
    def avg_power_w(self) -> float:
        """Average power in watts over the interval."""
        if self.duration_s <= 0:
            return 0.0
        return self.energy_j / self.duration_s
