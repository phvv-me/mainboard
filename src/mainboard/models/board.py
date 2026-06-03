from __future__ import annotations

import json
import platform
from pathlib import Path

from .base import FrozenModel
from .commands import cached_run

_DMI_ROOT = Path("/sys/class/dmi/id")
_DMI_FIELDS = {
    "vendor": "board_vendor",
    "model": "board_name",
    "version": "board_version",
    "bios_vendor": "bios_vendor",
    "bios_version": "bios_version",
}


def _read_dmi(name: str) -> str:
    """Stripped contents of a DMI sysfs file, or "" when it cannot be read."""
    try:
        return (_DMI_ROOT / name).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _probe_linux() -> dict[str, str]:
    """Board and BIOS identity from the Linux DMI sysfs tree."""
    return {field: _read_dmi(source) for field, source in _DMI_FIELDS.items()}


def _probe_macos() -> dict[str, str]:
    """Board identity from `system_profiler`, with Apple as the vendor."""
    try:
        payload = json.loads(cached_run("system_profiler", "SPHardwareDataType", "-json"))
    except (OSError, ValueError):
        return {}
    hardware = (payload.get("SPHardwareDataType") or [{}])[0]
    model = hardware.get("machine_model") or hardware.get("chip_type") or ""
    return {
        "vendor": "Apple",
        "model": str(model),
        "version": str(hardware.get("machine_name") or ""),
    }


class Board(FrozenModel):
    """The host's motherboard and firmware identity.

    Probed from the OS so a tool can record which physical system board and BIOS a
    snapshot came from. Unreadable on a host means empty strings, never an error.

    vendor: motherboard manufacturer.
    model: motherboard product name.
    version: motherboard revision or model identifier.
    bios_vendor: firmware vendor.
    bios_version: firmware version string.
    """

    vendor: str = ""
    model: str = ""
    version: str = ""
    bios_vendor: str = ""
    bios_version: str = ""

    @classmethod
    def probe(cls) -> Board:
        """Detect the motherboard and BIOS identity for the current platform."""
        fields = _probe_macos() if platform.system() == "Darwin" else _probe_linux()
        return cls(**fields)
