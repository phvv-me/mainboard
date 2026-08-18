from typing import TYPE_CHECKING

from mainboard.probe import PartitionInfo
from mainboard.probe.facts import partition_info as pi_mod

if TYPE_CHECKING:
    import pytest


class FakePsutilPartition:
    def __init__(self, device: str, *, mountpoint: str, fstype: str, opts: str) -> None:
        self.device = device
        self.mountpoint = mountpoint
        self.fstype = fstype
        self.opts = opts


class FakeDiskUsage:
    def __init__(self, total: int, *, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


def test_all_reads_psutil_partitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """`all` builds one `PartitionInfo` per `psutil.disk_partitions` entry."""
    fake = FakePsutilPartition("/dev/nvme0n1p1", mountpoint="/", fstype="ext4", opts="rw,relatime")
    monkeypatch.setattr(pi_mod.psutil, "disk_partitions", lambda all: [fake])
    (partition,) = PartitionInfo.all()
    assert partition.device == "/dev/nvme0n1p1"
    assert partition.mountpoint == "/"
    assert partition.fstype == "ext4"
    assert partition.opts == "rw,relatime"


def test_readonly_reads_the_ro_option() -> None:
    """`readonly` is true only when `ro` appears among the comma-separated opts."""
    assert PartitionInfo(device="d", mountpoint="/", fstype="ext4", opts="ro,relatime").readonly
    assert not PartitionInfo(device="d", mountpoint="/", fstype="ext4", opts="rw").readonly


def test_usage_fields_read_through_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """total/used/free bytes and their gibibyte and percentage views read `disk_usage`."""
    monkeypatch.setattr(
        pi_mod.psutil,
        "disk_usage",
        lambda mountpoint: FakeDiskUsage(100 * 1024**3, used=40 * 1024**3, free=60 * 1024**3),
    )
    partition = PartitionInfo(device="d", mountpoint="/", fstype="ext4")
    assert partition.total_bytes == 100 * 1024**3
    assert partition.used_bytes == 40 * 1024**3
    assert partition.free_bytes == 60 * 1024**3
    assert partition.total_gb == 100.0
    assert partition.used_gb == 40.0
    assert partition.free_gb == 60.0
    assert partition.utilization_pct == 40.0


def test_usage_is_none_when_mount_is_inaccessible(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inaccessible mount degrades every byte field to zero instead of raising."""

    def boom(mountpoint: str) -> FakeDiskUsage:
        raise PermissionError

    monkeypatch.setattr(pi_mod.psutil, "disk_usage", boom)
    partition = PartitionInfo(device="d", mountpoint="/mnt", fstype="ext4")
    assert partition.usage is None
    assert (partition.total_bytes, partition.used_bytes, partition.free_bytes) == (0, 0, 0)
    assert partition.utilization_pct == 0.0
