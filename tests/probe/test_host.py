from typing import NoReturn

import pytest

from mainboard.probe import CgroupMemory, Host, HostDisk, Scratch, Vendor
from mainboard.probe import host as host_mod

_GIB = 1024**3
_X86_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Platinum 8480C
cache size\t: 107520 KB

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8480C
"""
# Grace and the Raspberry Pi both answer only with MIDR registers, no model-name line at all.
_GRACE_CPUINFO = """processor\t: 0
BogoMIPS\t: 2000.00
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd4f

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd4f
"""
_PI_CPUINFO = """processor\t: 0
CPU implementer\t: 0x41
CPU part\t: 0xd08

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd08
"""
# A big.LITTLE laptop part, one Qualcomm core the part table does not know beside two Arm ones.
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
_NO_IDENTITY_CPUINFO = "processor\t: 0\nBogoMIPS\t: 100\n"


def as_host(monkeypatch: pytest.MonkeyPatch, system: str, sysctl: str, cpuinfo: str) -> Host:
    """Build a `Host` reading the given OS name, `sysctl` answer, and `/proc/cpuinfo` text."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: system)
    monkeypatch.setattr(host_mod, "sysctl", lambda name: sysctl)
    monkeypatch.setattr(Host, "cpuinfo_text", cpuinfo)
    return Host()


@pytest.mark.parametrize(
    ("system", "sysctl", "cpuinfo", "expected"),
    [
        pytest.param(
            "Linux", "", _X86_CPUINFO, "Intel(R) Xeon(R) Platinum 8480C", id="x86-model-name"
        ),
        pytest.param("Linux", "", _GRACE_CPUINFO, "2x Arm Neoverse-V2", id="arm-midr-fallback"),
        pytest.param("Linux", "", "", "fallback-cpu", id="no-cpuinfo"),
        pytest.param("Linux", "", _NO_IDENTITY_CPUINFO, "fallback-cpu", id="cpuinfo-without-ids"),
        pytest.param("Darwin", "Apple M4 Pro", "", "Apple M4 Pro", id="darwin-sysctl"),
        pytest.param("Darwin", "", "", "fallback-cpu", id="darwin-sysctl-unreadable"),
    ],
)
def test_the_cpu_name_comes_from_the_first_identity_source_that_answers(
    system: str, sysctl: str, cpuinfo: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each platform names its CPU through its own best source.

    macOS names the SoC through `sysctl`, Linux prefers the cpuinfo model-name line and then
    the MIDR core mix, and a host that offers neither falls back to `platform.processor`.
    """
    monkeypatch.setattr(host_mod.platform, "processor", lambda: "fallback-cpu")
    assert as_host(monkeypatch, system, sysctl, cpuinfo).cpu == expected


@pytest.mark.parametrize(
    ("cpuinfo", "expected"),
    [
        pytest.param(_GRACE_CPUINFO, "2x Arm Neoverse-V2", id="grace"),
        pytest.param(_PI_CPUINFO, "2x Arm Cortex-A72", id="raspberry-pi"),
        pytest.param(
            _XELITE_CPUINFO, "Qualcomm part 0x001 + 2x Arm Cortex-X925", id="big-little-mix"
        ),
        pytest.param(_NO_IDENTITY_CPUINFO, "", id="no-midr-registers"),
    ],
)
def test_the_arm_core_mix_names_known_parts_and_counts_the_repeats(
    cpuinfo: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ARM core mix is read from MIDR IDs.

    ARM ships no model-name line, so the core mix is read from the MIDR implementer and part
    IDs, repeats collapse behind a count, and an ID the table does not know is named as raw.
    """
    assert as_host(monkeypatch, "Linux", "", cpuinfo).arm_cpu_name == expected


@pytest.mark.parametrize(
    ("system", "cpuinfo", "expected"),
    [
        pytest.param("Darwin", "", Vendor.APPLE, id="darwin"),
        pytest.param("Linux", "vendor_id\t: GenuineIntel\n", Vendor.INTEL, id="intel"),
        pytest.param("Linux", "vendor_id\t: AuthenticAMD\n", Vendor.AMD, id="amd"),
        pytest.param("Linux", "CPU implementer\t: 0x41\n", Vendor.ARM, id="arm"),
        pytest.param("Linux", "CPU implementer\t: 0x61\n", Vendor.APPLE, id="apple-silicon"),
        pytest.param("Linux", "CPU implementer\t: 0x4e\n", Vendor.NVIDIA, id="nvidia"),
        pytest.param("Linux", "CPU implementer\t: 0x51\n", Vendor.QUALCOMM, id="qualcomm"),
        pytest.param("Linux", "CPU implementer\t: 0xff\n", Vendor.UNKNOWN, id="unmapped"),
    ],
)
def test_the_cpu_vendor_is_read_from_whichever_identity_record_the_os_keeps(
    system: str, cpuinfo: str, expected: Vendor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CPU vendor comes from the field each architecture actually writes.

    x86 records a `vendor_id` and ARM a MIDR implementer code, macOS needs neither, and an
    implementer outside the table is Unknown rather than a guess.
    """
    assert as_host(monkeypatch, system, "", f"processor\t: 0\n{cpuinfo}").cpu_vendor is expected


@pytest.mark.parametrize("reports_frequency", [True, False], ids=["reported", "unsupported"])
def test_core_counts_and_frequency_read_psutil_and_tolerate_a_missing_reading(
    reports_frequency: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing frequency never takes the host probe down.

    Core counts always answer, but a container or a foreign platform has no frequency to
    report, and that reads as `None` rather than taking the whole host probe down.
    """

    def unsupported() -> NoReturn:
        raise NotImplementedError

    monkeypatch.setattr(host_mod.psutil, "cpu_count", lambda logical=True: 14 if logical else 10)
    monkeypatch.setattr(
        host_mod.psutil,
        "cpu_freq",
        (lambda: type("Freq", (), {"current": 3200.0})()) if reports_frequency else unsupported,
        raising=False,
    )
    probed = Host()
    assert (probed.logical_cpus, probed.physical_cpus) == (14, 10)
    assert probed.cpu_freq_mhz == (3200.0 if reports_frequency else None)


def test_cpuinfo_text_reads_proc_and_degrades_to_empty_when_it_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/proc/cpuinfo` is Linux-only, so a host without one parses no records, never raising."""
    monkeypatch.setattr(host_mod.Path, "read_text", lambda self, **kw: "model name\t: X\n")
    assert Host().cpuinfo_records == ({"model name": "X"},)

    def absent(self: host_mod.Path, **kw: str | None) -> NoReturn:
        raise FileNotFoundError(self)

    monkeypatch.setattr(host_mod.Path, "read_text", absent)
    assert Host().cpuinfo_text == ""
    assert Host().cpuinfo_records == ()


def test_the_host_hands_each_subsystem_to_the_model_that_probes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host delegates every fact to the model that owns it.

    Memory, disks, the cgroup ceiling and the scratch tier are each that model's own probe
    rather than a second implementation here.
    """
    cgroup = CgroupMemory(limit_bytes=5 * _GIB, capped=True)
    scratch = Scratch(free_bytes=_GIB, source="LOCALDIR")
    monkeypatch.setattr(CgroupMemory, "probe", classmethod(lambda cls: cgroup))
    monkeypatch.setattr(Scratch, "probe", classmethod(lambda cls: scratch))

    probed = Host()
    assert isinstance(probed.disk, HostDisk)
    assert probed.memory.scope == "system"
    assert probed.cgroup_memory is cgroup
    assert probed.scratch is scratch
    assert probed.arch == host_mod.platform.machine().lower()
