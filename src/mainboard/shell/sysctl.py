from __future__ import annotations

from plumbum import CommandNotFound, local


def sysctl(name: str) -> str:
    """Read a macOS `sysctl` value by name, stripped.

    Returns an empty string when `sysctl` is missing or the key is unreadable, so
    callers can probe Darwin-only keys without guarding the platform first.

    name: sysctl key, e.g. `machdep.cpu.brand_string`.
    """
    try:
        return local["sysctl"]["-n", name]().strip()
    except (CommandNotFound, OSError, KeyError):
        return ""
