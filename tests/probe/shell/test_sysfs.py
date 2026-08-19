from pathlib import Path

import pytest

from mainboard.probe import shell
from mainboard.probe.shell import sysfs as sysfs_mod


def test_read_dmi_strips_present_field_and_tolerates_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`read_dmi` reads and strips a present DMI field and gives ``""`` for an absent one."""
    (tmp_path / "board_vendor").write_text("  ASUSTeK  \n")
    monkeypatch.setattr(sysfs_mod, "_DMI_ROOT", tmp_path)
    assert shell.read_dmi("board_vendor") == "ASUSTeK"
    assert shell.read_dmi("board_name") == ""
