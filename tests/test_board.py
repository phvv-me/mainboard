from __future__ import annotations

import pytest

from mainboard import Board
from mainboard.models import board as board_mod


def test_linux_probe_reads_dmi_sysfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux every field is read from its DMI sysfs file."""
    values = {
        "board_vendor": "ASUSTeK COMPUTER INC.",
        "board_name": "ROG STRIX X670E",
        "board_version": "Rev 1.xx",
        "bios_vendor": "American Megatrends",
        "bios_version": "2.10",
    }
    monkeypatch.setattr(board_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(board_mod.shell, "read_dmi", lambda field: values[field])

    board = Board.probe()

    assert board == Board(
        vendor="ASUSTeK COMPUTER INC.",
        model="ROG STRIX X670E",
        version="Rev 1.xx",
        bios_vendor="American Megatrends",
        bios_version="2.10",
    )


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        pytest.param(
            {"machine_model": "Mac16,8", "machine_name": "MacBook Pro"},
            Board(vendor="Apple", model="Mac16,8", version="MacBook Pro"),
            id="machine-model-and-name",
        ),
        pytest.param(
            {"chip_type": "Apple M4 Pro"},
            Board(vendor="Apple", model="Apple M4 Pro"),
            id="chip-fallback-no-version",
        ),
        pytest.param({}, Board(), id="empty-profile-is-all-empty"),
    ],
)
def test_macos_probe_maps_profiler_to_board(
    hardware: dict[str, str], expected: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS folds the profiler payload to a board: model prefers machine over chip, and
    a missing hardware record degrades to an all-empty board instead of raising."""
    profile = {"SPHardwareDataType": [hardware]} if hardware else {}
    monkeypatch.setattr(board_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(board_mod.shell, "system_profiler", lambda *types: profile)

    assert Board.probe() == expected
