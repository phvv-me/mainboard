from contextlib import suppress
from pathlib import Path

_DMI_ROOT = Path("/sys/class/dmi/id")


def read_dmi(field: str) -> str:
    """A DMI identity field from `/sys/class/dmi/id`, e.g. `board_vendor`, stripped.

    Empty when the field is absent or unreadable, so callers probe Linux-only DMI files
    without guarding their existence first.
    """
    with suppress(OSError):
        return (_DMI_ROOT / field).read_text(encoding="utf-8").strip()
    return ""
