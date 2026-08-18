from pathlib import Path

import pytest
from mainboard.probe import HostDisk
from mainboard.probe.facts import drive_info as drive_info_mod


def point_sys_block(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Redirect the model's `/sys/block` root at a tmp tree."""
    monkeypatch.setattr(drive_info_mod, "SYS_BLOCK", root)


def test_cards_lists_real_drives_and_skips_pseudo_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skip-listed prefix (loop, dm-, ...) and a zero-size device are both excluded."""
    point_sys_block(monkeypatch, tmp_path)
    (tmp_path / "nvme0n1").mkdir()
    (tmp_path / "nvme0n1" / "size").write_text("2000000")
    (tmp_path / "loop0").mkdir()
    (tmp_path / "loop0" / "size").write_text("100")
    (tmp_path / "sda").mkdir()
    (tmp_path / "sda" / "size").write_text("0")  # present but zero-sized, excluded

    cards = HostDisk().cards
    assert [c.name for c in cards] == ["nvme0n1"]


def test_cards_is_empty_when_sysfs_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host with no `/sys/block` (non-Linux) yields no cards instead of raising."""
    point_sys_block(monkeypatch, tmp_path / "absent")
    assert HostDisk().cards == ()


def test_total_bytes_and_gb_sum_every_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`total_bytes`/`total_gb` sum every listed card's own size."""
    point_sys_block(monkeypatch, tmp_path)
    for name, sectors in (("nvme0n1", 2_000_000), ("nvme1n1", 4_000_000)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "size").write_text(str(sectors))

    disk = HostDisk()
    expected = (2_000_000 + 4_000_000) * 512
    assert disk.total_bytes == expected
    assert disk.total_gb == pytest.approx(expected / 1024**3)
