from ... import shell


def apple_system_profile() -> shell.SystemProfile:
    """Return macOS hardware and display profiler data."""
    return shell.system_profiler("SPHardwareDataType", "SPDisplaysDataType")


def hardware_record() -> shell.ProfileRecord:
    """The first `SPHardwareDataType` record; empty when the profiler reports none."""
    return (apple_system_profile().get("SPHardwareDataType") or [{}])[0]
