from __future__ import annotations

from contextlib import suppress
from pathlib import Path

DMI_ROOT = Path("/sys/class/dmi/id")


def read_dmi(field: str) -> str:
    """Read a DMI identity field from `/sys/class/dmi/id`, stripped.

    Returns an empty string when the field is absent or unreadable, so callers
    can probe Linux-only DMI files without guarding their existence first.

    field: DMI file name, e.g. `board_vendor` or `bios_version`.
    """
    with suppress(OSError):
        return (DMI_ROOT / field).read_text(encoding="utf-8").strip()
    return ""
