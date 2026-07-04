from enum import IntFlag


class ThrottleReason(IntFlag):
    """NVML clock throttle reason bitmask values.

    Mirrors ``nvmlClocksThrottleReasons*`` constants. Each member is
    a single bit; composing them with ``|`` produces the same bitmask
    that ``nvmlDeviceGetCurrentClocksThrottleReasons()`` returns.
    """

    NONE = 0x00
    GPU_IDLE = 0x01
    APPLICATIONS_CLOCKS = 0x02
    SW_POWER_CAP = 0x04
    HW_SLOWDOWN = 0x08
    SYNC_BOOST = 0x10
    SW_THERMAL_SLOWDOWN = 0x20
    HW_THERMAL_SLOWDOWN = 0x40
    HW_POWER_BRAKE = 0x80
    DISPLAY_CLOCK_SETTING = 0x100

    _BENIGN = GPU_IDLE | APPLICATIONS_CLOCKS

    @property
    def is_concerning(self) -> bool:
        """Whether any non-benign throttle reasons are active."""
        return bool(self & ~ThrottleReason._BENIGN)
