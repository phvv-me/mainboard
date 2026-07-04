from typing import TYPE_CHECKING

from .base import Model

if TYPE_CHECKING:
    from .thermal_state import ThermalState


class ThermalTracker(Model):
    """Accumulate thermal snapshots and detect throttle events.

    Tracks peak temperature, minimum margin, and whether any concerning
    throttling occurred during the profiling session.
    """

    peak_temperature_c: int = 0
    min_margin_c: int = 999
    any_throttling: bool = False
    throttle_reasons_seen: int = 0
    sample_count: int = 0

    def record(self, snap: ThermalState) -> None:
        """Record a thermal snapshot.

        snap: snapshot to accumulate.
        """
        self.sample_count += 1
        if snap.temperature_c > self.peak_temperature_c:
            self.peak_temperature_c = snap.temperature_c
        if snap.margin_c < self.min_margin_c:
            self.min_margin_c = snap.margin_c
        if snap.is_throttling:
            self.any_throttling = True
            self.throttle_reasons_seen |= snap.throttle_reasons
