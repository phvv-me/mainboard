from pathlib import Path
from typing import NoReturn

import pytest
from hypothesis import HealthCheck, example, given, settings

from mainboard.probe import DiskKind, DriveInfo, HostDisk, PartitionInfo
from mainboard.probe.facts import drive_info as drive_info_mod
from mainboard.probe.facts import partition_info as partition_mod

from ...strategies import TEXT

_GIB = 1024**3


class FakePsutilPartition:
    """One `psutil.disk_partitions()` entry, the four fields a `PartitionInfo` is built from."""

    def __init__(self, device: str, *, mountpoint: str, fstype: str, opts: str) -> None:
        self.device = device
        self.mountpoint = mountpoint
        self.fstype = fstype
        self.opts = opts


class FakeDiskUsage:
    """One `psutil.disk_usage()` reading."""

    def __init__(self, total: int, *, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


@pytest.fixture
def sys_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the `/sys/block` root at a tmp tree and hand it back to be populated."""
    monkeypatch.setattr(drive_info_mod, "SYS_BLOCK", tmp_path)
    return tmp_path


def make_drive(root: Path, name: str, *, sectors: str | None = None, **files: str) -> Path:
    """Create one `<root>/<name>` block device directory and return its `device` subdirectory.

    root: the fake `/sys/block`.
    name: the kernel device name, e.g. `nvme0n1`.
    sectors: the `size` file contents in 512-byte sectors, left absent when `None`.
    files: extra files to write under `device`, e.g. a `model` or a `serial`.
    """
    device = root / name / "device"
    device.mkdir(parents=True, exist_ok=True)
    if sectors is not None:
        (root / name / "size").write_text(sectors, encoding="utf-8")
    for field, value in files.items():
        (device / field).write_text(value, encoding="utf-8")
    return device


@pytest.mark.parametrize(
    ("name", "rotational", "expected"),
    [
        pytest.param("nvme0n1", None, DiskKind.NVME, id="nvme-by-name"),
        pytest.param("sda", "1", DiskKind.HDD, id="rotating"),
        pytest.param("sda", "0", DiskKind.SSD, id="solid-state"),
        pytest.param("sda", None, DiskKind.UNKNOWN, id="no-rotational-flag"),
    ],
)
def test_the_drive_kind_comes_from_the_name_then_the_rotational_flag(
    name: str, rotational: str | None, expected: DiskKind, sys_block: Path
) -> None:
    """The drive kind comes from the name first and the rotational flag second.

    An NVMe device is named as one and never needs the flag read, while anything else is
    rotating or not according to sysfs, and a device that will not say is left Unknown.
    """
    make_drive(sys_block, name, sectors="2000000")
    if rotational is not None:
        (sys_block / name / "queue").mkdir()
        (sys_block / name / "queue" / "rotational").write_text(rotational, encoding="utf-8")
    assert DriveInfo(name=name).kind is expected


# Each example writes a sysfs file, so the budget is trimmed and the readings that matter
# are pinned as explicit examples rather than left to the search.
@settings(max_examples=12, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw=TEXT)
@example(raw="Samsung SSD 990 PRO")
@example(raw="  padded  \n")
@example(raw="none")
@example(raw="Not Specified")
@example(raw="")
def test_a_sysfs_identity_field_reads_back_stripped_or_as_nothing_at_all(
    raw: str, sys_block: Path
) -> None:
    """A sysfs reading is a stripped string or nothing.

    sysfs pads its pseudo-files and fills the ones it has no answer for with a placeholder,
    so a reading is either a stripped non-empty string or `None`, and never the placeholder.
    """
    make_drive(sys_block, "nvme0n1", sectors="2000000", model=raw)
    model = DriveInfo(name="nvme0n1").model
    assert model is None or (model == raw.strip() and model != "")
    assert model is None or model.lower() not in {"unknown", "not specified", "none", "n/a"}


def test_a_drive_reports_its_device_path_capacity_and_partitions(
    sys_block: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capacity and partitions come straight from sysfs and the mount table.

    Capacity is the sysfs sector count scaled by the 512-byte sector sysfs always reports in,
    a drive with no size file reads as empty, and the partitions are the mounted ones whose
    device name this drive prefixes.
    """
    make_drive(sys_block, "nvme0n1", sectors="2000000", serial="S6B0NJ0T", model="none")
    make_drive(sys_block, "sda")
    partitions = (
        PartitionInfo(device="/dev/nvme0n1p1", mountpoint="/", fstype="ext4"),
        PartitionInfo(device="/dev/nvme0n1p2", mountpoint="/boot", fstype="vfat"),
        PartitionInfo(device="/dev/sda1", mountpoint="/data", fstype="xfs"),
    )
    monkeypatch.setattr(PartitionInfo, "all", classmethod(lambda cls: partitions))

    drive = DriveInfo(name="nvme0n1")
    assert drive.device == "/dev/nvme0n1"
    assert drive.size_bytes == 2000000 * 512
    assert drive.size_gb == pytest.approx(2000000 * 512 / _GIB)
    assert drive.serial == "S6B0NJ0T"
    assert drive.model is None  # sysfs wrote a placeholder rather than a model
    assert {p.mountpoint for p in drive.partitions} == {"/", "/boot"}
    assert DriveInfo(name="sda").size_bytes == 0


def test_the_host_lists_real_drives_and_skips_pseudo_and_empty_devices(sys_block: Path) -> None:
    """Only real, non-empty drives count toward capacity.

    A loop, device-mapper or ramdisk entry is not a physical drive and a zero-sized one is
    not a usable drive, so neither belongs in the capacity a caller sizes work against.
    """
    make_drive(sys_block, "nvme0n1", sectors="2000000")
    make_drive(sys_block, "nvme1n1", sectors="4000000")
    make_drive(sys_block, "loop0", sectors="100")
    make_drive(sys_block, "sda", sectors="0")
    make_drive(sys_block, "sdb")  # present but with no size file at all

    disk = HostDisk()
    assert [card.name for card in disk.cards] == ["nvme0n1", "nvme1n1"]
    assert disk.total_bytes == (2_000_000 + 4_000_000) * 512
    assert disk.total_gb == pytest.approx(disk.total_bytes / _GIB)


def test_a_size_no_kernel_could_have_written_reads_as_no_capacity(sys_block: Path) -> None:
    """A torn pseudo-file degrades quietly instead of killing the scan.

    The reader promises quiet degradation, so a torn or padded pseudo-file is zero bytes and
    the drive it belongs to is not counted, rather than a probe that dies mid-scan.
    """
    make_drive(sys_block, "nvme0n1", sectors="2000000")
    make_drive(sys_block, "sdb", sectors="4000000 4000000")
    assert DriveInfo(name="sdb").size_bytes == 0
    assert [card.name for card in HostDisk().cards] == ["nvme0n1"]


def test_the_host_lists_no_drives_when_sysfs_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-Linux host has no `/sys/block` at all, which reads as no drives, never a raise."""
    monkeypatch.setattr(drive_info_mod, "SYS_BLOCK", tmp_path / "absent")
    assert HostDisk().cards == ()


def test_a_partition_carries_its_mount_options_through_from_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partitions mirror psutil and read the `ro` flag exactly.

    `all` mirrors one psutil entry per mounted filesystem, and `readonly` reads the `ro` flag
    out of the raw option string rather than out of a substring match on it.
    """
    fake = FakePsutilPartition("/dev/nvme0n1p1", mountpoint="/", fstype="ext4", opts="ro,relatime")
    monkeypatch.setattr(partition_mod.psutil, "disk_partitions", lambda all: [fake])

    (partition,) = PartitionInfo.all()
    assert (partition.device, partition.mountpoint, partition.fstype) == (
        "/dev/nvme0n1p1",
        "/",
        "ext4",
    )
    assert partition.opts == "ro,relatime"
    assert partition.readonly is True
    assert PartitionInfo(device="d", mountpoint="/", fstype="ext4", opts="rw").readonly is False


@pytest.mark.parametrize("accessible", [True, False], ids=["mounted", "inaccessible"])
def test_partition_capacity_reads_through_disk_usage_and_zeroes_out_when_it_cannot(
    accessible: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable mount reads as zero bytes everywhere.

    A mount the caller has no permission to stat degrades every byte view to zero, so a
    utilization percentage over an unreadable mount never divides by zero either.
    """

    def refuse(mountpoint: str) -> NoReturn:
        raise PermissionError(mountpoint)

    usage = FakeDiskUsage(100 * _GIB, used=40 * _GIB, free=60 * _GIB)
    monkeypatch.setattr(
        partition_mod.psutil, "disk_usage", (lambda m: usage) if accessible else refuse
    )
    partition = PartitionInfo(device="d", mountpoint="/mnt", fstype="ext4")

    expected = (100 * _GIB, 40 * _GIB, 60 * _GIB) if accessible else (0, 0, 0)
    assert (partition.total_bytes, partition.used_bytes, partition.free_bytes) == expected
    assert (partition.total_gb, partition.used_gb, partition.free_gb) == tuple(
        value / _GIB for value in expected
    )
    assert partition.utilization_pct == (40.0 if accessible else 0.0)
