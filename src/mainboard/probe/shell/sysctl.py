from plumbum import CommandNotFound, local


def sysctl(name: str) -> str:
    """A macOS `sysctl` value, e.g. `machdep.cpu.brand_string`, stripped.

    Empty when `sysctl` is missing or the key unreadable, so callers probe Darwin-only keys
    without guarding the platform first.
    """
    try:
        return local["sysctl"]["-n", name]().strip()
    except CommandNotFound, OSError, KeyError:
        return ""
