from .base import FrozenModel
from .throttle_reason import ThrottleReason


class ThermalState(FrozenModel):
    """Temperature, thresholds, and throttle reasons.

    temperature_c: GPU die temperature in Celsius.
    shutdown_threshold_c: temperature at which the GPU shuts down.
    slowdown_threshold_c: temperature at which the GPU starts throttling.
    throttle_reasons: bitmask from NVML, typed as ``ThrottleReason``.
    """

    temperature_c: int = 0
    shutdown_threshold_c: int = 0
    slowdown_threshold_c: int = 0
    throttle_reasons: int = 0

    @property
    def margin_c(self) -> int:
        """Degrees below slowdown threshold. Negative means throttling."""
        return self.slowdown_threshold_c - self.temperature_c

    @property
    def is_throttling(self) -> bool:
        """Whether any non-idle throttle reason is active."""
        return ThrottleReason(self.throttle_reasons).is_concerning

    @property
    def throttle_names(self) -> list[str]:
        """Human-readable names of active throttle reasons."""
        reasons = ThrottleReason(self.throttle_reasons)
        return [flag.name for flag in ThrottleReason if flag in reasons and flag.name]
