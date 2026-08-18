from pathlib import Path

import pytest
from mainboard.probe import DiskKind, DriveInfo, PartitionInfo
from mainboard.probe.facts import drive_info as drive_info_mod


def point_sys_block(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Redirect the model's `/sys/block` root at a tmp tree."""
    monkeypatch.setattr(drive_info_mod, "SYS_BLOCK", root)


def make_nvme(root: Path, name: str = "nvme0n1") -> Path:
    """Create a minimal `<root>/<name>` NVMe device dir and return its `device` subdir."""
    device = root / name / "device"
    device.mkdir(parents=True)
    (root / name / "size").write_text("2000000")
    return device


def test_device_path_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`device` builds `/dev/<name>` and `size_bytes`/`size_gb` read the sysfs `size` file."""
    point_sys_block(monkeypatch, tmp_path)
    make_nvme(tmp_path)
    drive = DriveInfo(name="nvme0n1")
    assert drive.device == "/dev/nvme0n1"
    assert drive.size_bytes == 2000000 * 512
    assert drive.size_gb == pytest.approx(2000000 * 512 / 1024**3)


def test_kind_nvme_by_name_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A device whose name starts with `nvme` is classified NVMe without reading `rotational`."""
    point_sys_block(monkeypatch, tmp_path)
    make_nvme(tmp_path)
    assert DriveInfo(name="nvme0n1").kind == DiskKind.NVME


@pytest.mark.parametrize(("rotational", "expected"), [("1", DiskKind.HDD), ("0", DiskKind.SSD)])
def test_kind_ssd_or_hdd_from_rotational(
    rotational: str, expected: DiskKind, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-NVMe device's kind comes from the sysfs `rotational` flag."""
    point_sys_block(monkeypatch, tmp_path)
    (tmp_path / "sda" / "queue").mkdir(parents=True)
    (tmp_path / "sda" / "queue" / "rotational").write_text(rotational)
    assert DriveInfo(name="sda").kind == expected


def test_kind_unknown_when_rotational_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device with no `rotational` file (or an absent queue dir) is Unknown."""
    point_sys_block(monkeypatch, tmp_path)
    (tmp_path / "sda").mkdir(parents=True)
    assert DriveInfo(name="sda").kind == DiskKind.UNKNOWN


def test_model_and_serial_placeholder_values_become_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sysfs placeholder value (`unknown`, `none`, ...) reads back as `None`."""
    point_sys_block(monkeypatch, tmp_path)
    device_dir = make_nvme(tmp_path)
    (device_dir / "model").write_text("Samsung SSD 990 PRO")
    (device_dir / "serial").write_text("none")
    drive = DriveInfo(name="nvme0n1")
    assert drive.model == "Samsung SSD 990 PRO"
    assert drive.serial is None


def test_model_is_none_when_file_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing sysfs file degrades to `None` rather than raising."""
    point_sys_block(monkeypatch, tmp_path)
    make_nvme(tmp_path)
    assert DriveInfo(name="nvme0n1").model is None


def test_partitions_matches_by_device_name_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A drive's `partitions` are the mounted partitions whose device name it prefixes."""
    partitions = (
        PartitionInfo(device="/dev/nvme0n1p1", mountpoint="/", fstype="ext4"),
        PartitionInfo(device="/dev/nvme0n1p2", mountpoint="/boot", fstype="vfat"),
        PartitionInfo(device="/dev/sda1", mountpoint="/data", fstype="xfs"),
    )
    monkeypatch.setattr(PartitionInfo, "all", classmethod(lambda cls: partitions))
    drive = DriveInfo(name="nvme0n1")
    assert {p.mountpoint for p in drive.partitions} == {"/", "/boot"}
