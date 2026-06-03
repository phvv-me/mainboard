from __future__ import annotations

import json

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
    monkeypatch.setattr(board_mod, "_read_dmi", lambda name: values[name])

    board = Board.probe()

    assert board.vendor == "ASUSTeK COMPUTER INC."
    assert board.model == "ROG STRIX X670E"
    assert board.version == "Rev 1.xx"
    assert board.bios_vendor == "American Megatrends"
    assert board.bios_version == "2.10"


def test_read_dmi_strips_and_tolerates_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present file is stripped; a missing one yields an empty string."""

    class FakePath:
        def __init__(self, present: bool) -> None:
            self.present = present

        def __truediv__(self, _name: str) -> FakePath:
            return self

        def read_text(self, encoding: str) -> str:
            if not self.present:
                raise OSError("no such file")
            return "  ASUSTeK  \n"

    monkeypatch.setattr(board_mod, "_DMI_ROOT", FakePath(present=True))
    assert board_mod._read_dmi("board_vendor") == "ASUSTeK"

    monkeypatch.setattr(board_mod, "_DMI_ROOT", FakePath(present=False))
    assert board_mod._read_dmi("board_vendor") == ""


def test_macos_probe_uses_system_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """On macOS the board is read from `system_profiler` with Apple as vendor."""
    payload = json.dumps(
        {"SPHardwareDataType": [{"machine_model": "Mac16,8", "machine_name": "MacBook Pro"}]}
    )
    monkeypatch.setattr(board_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(board_mod, "cached_run", lambda *cmd: payload)

    board = Board.probe()

    assert board.vendor == "Apple"
    assert board.model == "Mac16,8"
    assert board.version == "MacBook Pro"
    assert board.bios_vendor == ""


def test_macos_probe_falls_back_to_chip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a machine model the chip name stands in as the board model."""
    payload = json.dumps({"SPHardwareDataType": [{"chip_type": "Apple M4 Pro"}]})
    monkeypatch.setattr(board_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(board_mod, "cached_run", lambda *cmd: payload)

    board = Board.probe()

    assert board.model == "Apple M4 Pro"
    assert board.version == ""


def test_macos_probe_tolerates_profiler_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing `system_profiler` call yields an all-empty board, never an error."""

    def boom(*_cmd: str) -> str:
        raise OSError("system_profiler missing")

    monkeypatch.setattr(board_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(board_mod, "cached_run", boom)

    board = Board.probe()

    assert board == Board()
