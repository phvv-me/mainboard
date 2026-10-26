from patos import FrozenModel

from .utilization import Utilization


class Energy(FrozenModel):
    """Instantaneous power draw of one unit and its own circuitry, 0 when no sensor answers."""

    power_w: float = 0.0


class Thermal(FrozenModel):
    """Die temperature and whatever is currently holding a unit below its clocks.

    temperature_c: 0 when no sensor answers.
    throttle_names: the active slowdowns; benign clock states (an idle device, an applied clock
        setting) never appear, so a non-empty tuple always means real lost performance.
    """

    temperature_c: int = 0
    throttle_names: tuple[str, ...] = ()

    @property
    def is_throttling(self) -> bool:
        """Whether a real slowdown is active right now."""
        return bool(self.throttle_names)


class UnitProcess(FrozenModel):
    """One process's memory footprint on a unit."""

    pid: int = 0
    used_bytes: int = 0


class Telemetry(FrozenModel):
    """One point-in-time reading of a unit's sensors.

    Every field carries its own neutral value, so a provider that reads power but cannot
    read per-process memory still reports the power rather than refusing the whole reading.

    region: the caller's name for whatever was running, so a reading says what it belongs to.
    """

    unit_name: str = ""
    region: str = ""
    energy: Energy = Energy()
    thermal: Thermal = Thermal()
    utilization: Utilization = Utilization()
    processes: tuple[UnitProcess, ...] = ()
