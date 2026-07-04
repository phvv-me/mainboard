from .base import FrozenModel


class EnergyReading(FrozenModel):
    """Instantaneous power and monotonic energy counter.

    power_mw: instantaneous power draw in milliwatts.
    energy_mj: cumulative energy counter in millijoules (monotonic).
    """

    power_mw: int = 0
    energy_mj: int = 0

    @property
    def power_w(self) -> float:
        """Power in watts."""
        return self.power_mw / 1000.0
