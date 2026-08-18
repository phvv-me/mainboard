import pytest

from mainboard.probe import Host, HostDisk, Vendor
from mainboard.probe import host as host_mod

_LINUX_X86_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Platinum 8480C
cache size\t: 107520 KB

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8480C
"""

_GRACE_CPUINFO = """processor\t: 0
BogoMIPS\t: 2000.00
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd4f

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd4f
"""

_XELITE_CPUINFO = """processor\t: 0
CPU implementer\t: 0x51
CPU part\t: 0x001

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd85

processor\t: 2
CPU implementer\t: 0x41
CPU part\t: 0xd85
"""

_PI_CPUINFO = """processor\t: 0
CPU implementer\t: 0x41
CPU part\t: 0xd08

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd08
"""


@pytest.fixture
def linux_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the non-Darwin branch so `/proc/cpuinfo` parsing is exercised."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Linux")


def make_host(monkeypatch: pytest.MonkeyPatch, cpuinfo: str) -> Host:
    """Build a `Host` whose `/proc/cpuinfo` is the given text."""
    host = Host()
    monkeypatch.setattr(type(host), "cpuinfo_text", cpuinfo)
    return host


@pytest.mark.usefixtures("linux_host")
def test_cpu_model_name_from_cpuinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """An x86 `model name` line is used verbatim as the CPU name."""
    host = make_host(monkeypatch, _LINUX_X86_CPUINFO)
    assert host.cpu == "Intel(R) Xeon(R) Platinum 8480C"


@pytest.mark.usefixtures("linux_host")
def test_arm_cpu_name_falls_back_to_midr(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no model-name line, ARM MIDR implementer/part IDs name the cores."""
    host = make_host(monkeypatch, _GRACE_CPUINFO)
    assert host.cpu == host.arm_cpu_name
    assert host.arm_cpu_name == "2x Arm Neoverse-V2"
    assert make_host(monkeypatch, _PI_CPUINFO).arm_cpu_name == "2x Arm Cortex-A72"


@pytest.mark.usefixtures("linux_host")
def test_arm_cpu_name_mixes_known_parts_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MIDR mix names known cores and collapses repeats with a count prefix."""
    host = make_host(monkeypatch, _XELITE_CPUINFO)
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
@pytest.mark.parametrize(
    ("vendor_id", "expected"),
    [
        ("GenuineIntel", Vendor.INTEL),
        ("AuthenticAMD", Vendor.AMD),
    ],
)
def test_cpu_vendor_from_x86_vendor_id(
    vendor_id: str, expected: Vendor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux x86 the vendor comes from the cpuinfo `vendor_id` field."""
    host = make_host(monkeypatch, f"processor\t: 0\nvendor_id\t: {vendor_id}\n")
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
    monkeypatch.setattr(host_mod.platform, "processor", lambda: "fallback-cpu")
    assert host.cpu == "fallback-cpu"


def test_darwin_cpu_uses_sysctl(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Darwin the CPU name comes from `sysctl machdep.cpu.brand_string`."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(host_mod, "sysctl", lambda name: "Apple M4 Pro")
    assert Host().cpu == "Apple M4 Pro"
    assert Host().cpu_vendor == Vendor.APPLE


def test_cpu_counts_and_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core counts come from psutil and frequency tolerates a missing reading."""
    monkeypatch.setattr(host_mod.psutil, "cpu_count", lambda logical=True: 14 if logical else 10)
    monkeypatch.setattr(
        host_mod.psutil, "cpu_freq", lambda: type("F", (), {"current": 3200.0})(), raising=False
    )
    host = Host()
    assert host.logical_cpus == 14
    assert host.physical_cpus == 10
    assert host.cpu_freq_mhz == 3200.0


def test_darwin_sysctl_failure_falls_back_to_cpuinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty `sysctl` on Darwin falls through to cpuinfo/processor detection."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(host_mod, "sysctl", lambda name: "")
    host = make_host(monkeypatch, "")
    monkeypatch.setattr(host_mod.platform, "processor", lambda: "fallback")
    assert host.cpu == "fallback"


@pytest.mark.usefixtures("linux_host")
def test_cpu_falls_back_when_arm_name_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cpuinfo with no model name and no MIDR data uses `platform.processor`."""
    host = make_host(monkeypatch, "processor\t: 0\nBogoMIPS\t: 100\n")
    monkeypatch.setattr(host_mod.platform, "processor", lambda: "generic-cpu")
    assert host.arm_cpu_name == ""
    assert host.cpu == "generic-cpu"


def test_cpuinfo_text_reads_proc_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cpuinfo_text` returns `/proc/cpuinfo` text, or ``""`` when it is absent."""
    monkeypatch.setattr(host_mod.Path, "read_text", lambda self, **kw: "model name\t: X\n")
    assert "model name" in Host().cpuinfo_text

    def boom(self: host_mod.Path, **kw: str | None) -> str:
        raise FileNotFoundError(self)

    monkeypatch.setattr(host_mod.Path, "read_text", boom)
    assert Host().cpuinfo_text == ""


def test_disk_is_a_host_disk() -> None:
    """`Host.disk` exposes the `HostDisk` enumerator."""
    assert isinstance(Host().disk, HostDisk)


def test_cpu_frequency_returns_none_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform that cannot report frequency yields `None` rather than raising."""

    def boom() -> None:
        raise NotImplementedError

    monkeypatch.setattr(host_mod.psutil, "cpu_freq", boom, raising=False)
    assert Host().cpu_freq_mhz is None


def test_memory_defers_to_the_memory_model() -> None:
    """`Host.memory` returns a live system `Memory` reading."""
    assert Host().memory.scope == "system"
