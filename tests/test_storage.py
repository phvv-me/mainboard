from __future__ import annotations

from pathlib import Path

import pytest

from mainboard.enums import DiskKind
from mainboard.models import memory_card as mc_mod
from mainboard.models.drive_info import DriveInfo
from mainboard.models.host_disk import HostDisk
from mainboard.models.host_memory import HostMemory
from mainboard.models.memory_card import MemoryCard
from mainboard.models.partition_info import PartitionInfo

DMIDECODE = """Memory Device
\tSize: 16384 MB
\tLocator: DIMM_A1
\tBank Locator: BANK 0
\tType: DDR5
\tForm Factor: DIMM
\tSpeed: 5600 MT/s
\tManufacturer: Samsung
\tPart Number: M321R2GA3BB6

Memory Device
\tSize: No Module Installed
\tLocator: DIMM_B1
\tType: Unknown
"""


def test_memory_card_parses_populated_slot() -> None:
    """A populated dmidecode section yields capacity, speed, and identity fields."""
    populated, empty = (MemoryCard(section=s) for s in DMIDECODE.split("\n\n")[:2])
    assert populated.populated is True
    assert populated.size_bytes == 16384 * 1024**2
    assert populated.size_gb == 16.0
    assert populated.slot == "DIMM_A1"
    assert populated.bank == "BANK 0"
    assert populated.speed_mhz == 5600
    assert populated.memory_type == "DDR5"
    assert populated.form_factor == "DIMM"
    assert populated.manufacturer == "Samsung"
    assert populated.part_number == "M321R2GA3BB6"
    assert empty.populated is False
    assert empty.size_bytes == 0
    assert empty.memory_type == ""  # "Unknown" is treated as a placeholder
    assert empty.manufacturer is None


@pytest.mark.parametrize(
    ("size_str", "expected"),
    [("512 GB", 512 * 1024**3), ("32768 MB", 32768 * 1024**2), ("1 TB", 1024**4), ("weird", 0)],
)
def test_memory_card_size_parsing(size_str: str, expected: int) -> None:
    """The dmidecode size field parses common units and rejects junk."""
    card = MemoryCard(section=f"Memory Device\n\tSize: {size_str}\n\tLocator: X\n")
    assert card.size_bytes == (expected if expected else 0)


def test_memory_card_parse_size_handles_none() -> None:
    """Parsing a missing size string yields zero bytes."""
    assert MemoryCard._parse_size(None) == 0


def test_host_memory_speed_none_without_populated_speeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """`speed_mhz` is None when no populated slot reports a speed."""
    vm = type("VM", (), {"total": 0, "available": 0, "used": 0})()
    monkeypatch.setattr("mainboard.models.host_memory.psutil.virtual_memory", lambda: vm)
    monkeypatch.setattr(MemoryCard, "all", classmethod(lambda cls: ()))
    assert HostMemory().speed_mhz is None
    assert HostMemory().slots_used == 0
    assert HostMemory().utilization_pct == 0.0


def test_memory_card_all_parses_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MemoryCard.all` splits dmidecode output into Memory Device sections."""

    class Result:
        returncode = 0
        stdout = DMIDECODE

    monkeypatch.setattr(mc_mod.subprocess, "run", lambda *a, **k: Result())
    cards = MemoryCard.all()
    assert len(cards) == 2
    assert [c.slot for c in cards] == ["DIMM_A1", "DIMM_B1"]


def test_memory_card_all_tolerates_missing_dmidecode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or non-root dmidecode yields an empty tuple, not an error."""

    def boom(*a: object, **k: object) -> object:
        raise FileNotFoundError("dmidecode")

    monkeypatch.setattr(mc_mod.subprocess, "run", boom)
    assert MemoryCard.all() == ()


def test_host_memory_reports_psutil_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HostMemory` exposes psutil totals, swap, and a populated-slot summary."""
    vm = type("VM", (), {"total": 32 * 1024**3, "available": 20 * 1024**3, "used": 12 * 1024**3})()
    sm = type("SM", (), {"total": 8 * 1024**3, "used": 1 * 1024**3})()
    monkeypatch.setattr("mainboard.models.host_memory.psutil.virtual_memory", lambda: vm)
    monkeypatch.setattr("mainboard.models.host_memory.psutil.swap_memory", lambda: sm)
    cards = (
        MemoryCard(section="Memory Device\n\tSize: 16384 MB\n\tLocator: A\n\tSpeed: 5600 MT/s\n"),
        MemoryCard(section="Memory Device\n\tSize: No Module Installed\n\tLocator: B\n"),
    )
    monkeypatch.setattr(MemoryCard, "all", classmethod(lambda cls: cards))
    memory = HostMemory()
    assert memory.total_gb == 32.0
    assert memory.available_gb == 20.0
    assert memory.used_gb == 12.0
    assert memory.swap_total_gb == 8.0
    assert memory.swap_used_bytes == 1 * 1024**3
    assert memory.utilization_pct == pytest.approx(12 / 32 * 100)
    assert memory.slots_total == 2
    assert memory.slots_used == 1
    assert memory.speed_mhz == 5600


def test_partition_info_reads_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PartitionInfo` builds from psutil mounts and reports usage ratios."""
    part = type(
        "P",
        (),
        {"device": "/dev/sda1", "mountpoint": "/", "fstype": "ext4", "opts": "rw,relatime"},
    )()
    usage = type("U", (), {"total": 100, "used": 40, "free": 60})()
    monkeypatch.setattr(
        "mainboard.models.partition_info.psutil.disk_partitions", lambda all: [part]
    )
    monkeypatch.setattr("mainboard.models.partition_info.psutil.disk_usage", lambda mp: usage)
    info = PartitionInfo.all()[0]
    assert info.device == "/dev/sda1"
    assert info.readonly is False
    assert info.total_bytes == 100
    assert info.utilization_pct == 40.0


def test_partition_info_tolerates_inaccessible_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mount that cannot be stat'd reports zero capacity instead of raising."""
    info = PartitionInfo(device="/dev/x", mountpoint="/locked", fstype="ext4", opts="ro")

    def boom(_mp: str) -> object:
        raise PermissionError

    monkeypatch.setattr("mainboard.models.partition_info.psutil.disk_usage", boom)
    assert info.readonly is True
    assert info.total_bytes == 0
    assert info.used_bytes == 0
    assert info.free_bytes == 0
    assert info.total_gb == 0.0
    assert info.used_gb == 0.0
    assert info.free_gb == 0.0
    assert info.utilization_pct == 0.0


def test_drive_info_missing_sysfs_yields_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """A drive with no sysfs files reports unknown model/serial and zero size."""
    monkeypatch.setattr(Path, "exists", lambda self: False)

    def boom(self: object) -> str:
        raise OSError

    monkeypatch.setattr(Path, "read_text", boom)
    monkeypatch.setattr(DriveInfo, "partitions", ())
    drive = DriveInfo(name="sdz")
    assert drive.model is None
    assert drive.serial is None
    assert drive.size_bytes == 0
    assert drive.kind == DiskKind.UNKNOWN


def test_drive_info_partitions_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A drive keeps only the partitions whose device name belongs to it."""
    mine = PartitionInfo(device="/dev/nvme0n1p1", mountpoint="/", fstype="ext4")
    other = PartitionInfo(device="/dev/sda1", mountpoint="/data", fstype="ext4")
    monkeypatch.setattr(PartitionInfo, "all", classmethod(lambda cls: (mine, other)))
    drive = DriveInfo(name="nvme0n1")
    assert drive.partitions == (mine,)


def test_drive_info_classifies_kind_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """`DriveInfo` reads sysfs to derive device kind, size, and partitions."""
    files = {
        "/sys/block/nvme0n1/size": "1000215216",
        "/sys/block/nvme0n1/device/model": "Samsung SSD 990",
        "/sys/block/nvme0n1/device/serial": "S1234",
    }
    monkeypatch.setattr(Path, "read_text", lambda self: files[str(self)])
    monkeypatch.setattr(Path, "exists", lambda self: str(self) in files)
    monkeypatch.setattr(DriveInfo, "partitions", ())
    drive = DriveInfo(name="nvme0n1")
    assert drive.device == "/dev/nvme0n1"
    assert drive.kind == DiskKind.NVME
    assert drive.size_bytes == 1000215216 * 512
    assert drive.size_gb == 1000215216 * 512 / 1024**3
    assert drive.model == "Samsung SSD 990"
    assert drive.serial == "S1234"


@pytest.mark.parametrize(("rotational", "expected"), [("1", DiskKind.HDD), ("0", DiskKind.SSD)])
def test_drive_info_rotational_kind(
    rotational: str, expected: DiskKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-NVMe drive is SSD or HDD based on the sysfs rotational flag."""
    rot_path = "/sys/block/sda/queue/rotational"
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == rot_path)
    monkeypatch.setattr(Path, "read_text", lambda self: rotational)
    assert DriveInfo(name="sda").kind == expected


def test_drive_info_read_sys_skips_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """sysfs placeholder values like `unknown` collapse to `None`."""
    monkeypatch.setattr(Path, "read_text", lambda self: "unknown")
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert DriveInfo._read_sys(Path("/sys/block/sda/device/model")) is None


def test_host_disk_enumerates_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HostDisk` lists qualifying block devices and sums their capacity."""

    class DevDir:
        def __init__(self, name: str) -> None:
            self.name = name

        def __lt__(self, other: DevDir) -> bool:
            return self.name < other.name

        def __truediv__(self, other: str) -> Path:
            return Path(f"/sys/block/{self.name}/{other}")

    monkeypatch.setattr(
        "mainboard.models.host_disk.Path.iterdir",
        lambda self: iter([DevDir("nvme0n1"), DevDir("loop0")]),
    )
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self: "200")
    monkeypatch.setattr(DriveInfo, "size_bytes", 200 * 512)
    monkeypatch.setattr(DriveInfo, "partitions", ())
    host = HostDisk()
    assert [d.name for d in host.cards] == ["nvme0n1"]
    assert host.total_bytes == 200 * 512
    assert host.total_gb == 200 * 512 / 1024**3
