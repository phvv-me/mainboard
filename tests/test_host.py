from __future__ import annotations

import pytest

import mainboard.host as host_mod
from mainboard.enums import Vendor
from mainboard.host import Host

LINUX_X86_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Platinum 8480C
cache size\t: 107520 KB

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8480C
"""

GRACE_CPUINFO = """processor\t: 0
BogoMIPS\t: 2000.00
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd4f

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd4f
"""

XELITE_CPUINFO = """processor\t: 0
CPU implementer\t: 0x51
CPU part\t: 0x001

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd85

processor\t: 2
CPU implementer\t: 0x41
CPU part\t: 0xd85
"""


@pytest.fixture
def linux_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the non-Darwin branch so `/proc/cpuinfo` parsing is exercised."""
    monkeypatch.setattr("mainboard.host.platform.system", lambda: "Linux")


def make_host(monkeypatch: pytest.MonkeyPatch, cpuinfo: str) -> Host:
    """Build a `Host` whose `/proc/cpuinfo` is the given text."""
    host = Host()
    monkeypatch.setattr(type(host), "cpuinfo_text", cpuinfo)
    return host


@pytest.mark.usefixtures("linux_host")
def test_cpu_model_name_from_cpuinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """An x86 `model name` line is used verbatim as the CPU name."""
    host = make_host(monkeypatch, LINUX_X86_CPUINFO)
    assert host.cpu == "Intel(R) Xeon(R) Platinum 8480C"


@pytest.mark.usefixtures("linux_host")
def test_arm_cpu_name_falls_back_to_midr(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no model-name line, ARM MIDR implementer/part IDs name the cores."""
    host = make_host(monkeypatch, GRACE_CPUINFO)
    assert host.cpu == host.arm_cpu_name
    assert host.arm_cpu_name == "2x Arm part 0xd4f"


@pytest.mark.usefixtures("linux_host")
def test_arm_cpu_name_mixes_known_parts_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MIDR mix names known cores and collapses repeats with a count prefix."""
    host = make_host(monkeypatch, XELITE_CPUINFO)
    assert host.arm_cpu_name == "Qualcomm part 0x001 + 2x Arm Cortex-X925"


@pytest.mark.usefixtures("linux_host")
@pytest.mark.parametrize(
    ("implementer", "expected"),
    [
        ("0x41", Vendor.ARM),
        ("0x61", Vendor.APPLE),
        ("0x4e", Vendor.NVIDIA),
        ("0x51", Vendor.QUALCOMM),
    ],
)
def test_cpu_vendor_from_implementer(
    implementer: str, expected: Vendor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU vendor is read from the MIDR implementer code."""
    host = make_host(monkeypatch, f"processor\t: 0\nCPU implementer\t: {implementer}\n")
    assert host.cpu_vendor == expected


@pytest.mark.usefixtures("linux_host")
def test_cpu_vendor_unknown_when_no_implementer(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognized implementer yields the unknown vendor."""
    host = make_host(monkeypatch, "processor\t: 0\nCPU implementer\t: 0xff\n")
    assert host.cpu_vendor == Vendor.UNKNOWN


@pytest.mark.usefixtures("linux_host")
def test_cpu_falls_back_to_platform_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty cpuinfo defers to `platform.processor`."""
    host = make_host(monkeypatch, "")
    monkeypatch.setattr("mainboard.host.platform.processor", lambda: "fallback-cpu")
    assert host.cpu == "fallback-cpu"


def test_darwin_cpu_uses_sysctl(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Darwin the CPU name comes from `sysctl machdep.cpu.brand_string`."""
    monkeypatch.setattr("mainboard.host.platform.system", lambda: "Darwin")

    class FakeCmd:
        def __getitem__(self, args: object) -> FakeCmd:
            return self

        def __call__(self) -> str:
            return "Apple M4 Pro\n"

    monkeypatch.setattr(host_mod, "local", {"sysctl": FakeCmd()})
    assert Host().cpu == "Apple M4 Pro"
    assert Host().cpu_vendor == Vendor.APPLE


def test_cpu_counts_and_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core counts come from psutil and frequency tolerates a missing reading."""
    monkeypatch.setattr(
        "mainboard.host.psutil.cpu_count", lambda logical=True: 14 if logical else 10
    )
    monkeypatch.setattr(
        "mainboard.host.psutil.cpu_freq", lambda: type("F", (), {"current": 3200.0})()
    )
    host = Host()
    assert host.logical_cpus == 14
    assert host.physical_cpus == 10
    assert host.cpu_freq_mhz == 3200.0


def test_darwin_sysctl_failure_falls_back_to_cpuinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing `sysctl` on Darwin falls through to cpuinfo/processor detection."""
    monkeypatch.setattr("mainboard.host.platform.system", lambda: "Darwin")

    class FailingCmd:
        def __getitem__(self, args: object) -> FailingCmd:
            return self

        def __call__(self) -> str:
            raise OSError("sysctl missing")

    monkeypatch.setattr(host_mod, "local", {"sysctl": FailingCmd()})
    host = make_host(monkeypatch, "")
    monkeypatch.setattr("mainboard.host.platform.processor", lambda: "fallback")
    assert host.cpu == "fallback"


@pytest.mark.usefixtures("linux_host")
def test_cpu_falls_back_when_arm_name_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cpuinfo with no model name and no MIDR data uses `platform.processor`."""
    host = make_host(monkeypatch, "processor\t: 0\nBogoMIPS\t: 100\n")
    monkeypatch.setattr("mainboard.host.platform.processor", lambda: "generic-cpu")
    assert host.arm_cpu_name == ""
    assert host.cpu == "generic-cpu"


def test_cpuinfo_text_reads_proc_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cpuinfo_text` returns file contents, or an empty string when absent."""
    monkeypatch.setattr("mainboard.host.Path.read_text", lambda self: "model name\t: X\n")
    assert "model name" in Host().cpuinfo_text

    def boom(self: object) -> str:
        raise OSError("no /proc on this OS")

    monkeypatch.setattr("mainboard.host.Path.read_text", boom)
    assert Host().cpuinfo_text == ""


def test_disk_is_a_host_disk() -> None:
    """`Host.disk` exposes the `HostDisk` enumerator."""
    from mainboard.models.host_disk import HostDisk

    assert isinstance(Host().disk, HostDisk)


def test_cpu_frequency_returns_none_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform that cannot report frequency yields `None` rather than raising."""

    def boom() -> None:
        raise NotImplementedError

    monkeypatch.setattr("mainboard.host.psutil.cpu_freq", boom)
    assert Host().cpu_freq_mhz is None
