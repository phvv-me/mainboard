import platform
import re
from collections import Counter
from contextlib import suppress
from functools import cached_property
from pathlib import Path

import psutil

from . import shell
from .enums import Vendor
from .models.cgroup_memory import CgroupMemory
from .models.host_disk import HostDisk
from .models.memory import Memory
from .models.memory_hardware import MemoryHardware
from .models.scratch import Scratch

_CPU_MODEL_RE = re.compile(r"^(?:model name|hardware)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_CPU_IMPLEMENTER_RE = re.compile(r"^CPU implementer\s*:\s*(.+)$", re.MULTILINE)
_CPU_PART_RE = re.compile(r"^CPU part\s*:\s*(.+)$", re.MULTILINE)
_ARM_IMPLEMENTERS = {
    "0x41": "Arm",
    "0x4e": "NVIDIA",
    "0x51": "Qualcomm",
    "0x61": "Apple",
    "0xc0": "Ampere",
}
_VENDOR_BY_IMPLEMENTER = {
    "0x41": Vendor.ARM,
    "0x4e": Vendor.NVIDIA,
    "0x51": Vendor.QUALCOMM,
    "0x61": Vendor.APPLE,
}
_VENDOR_BY_VENDOR_ID = {
    "GenuineIntel": Vendor.INTEL,
    "AuthenticAMD": Vendor.AMD,
}
_ARM_PARTS = {
    ("0x41", "0xd08"): "Cortex-A72",
    ("0x41", "0xd4f"): "Neoverse-V2",
    ("0x41", "0xd85"): "Cortex-X925",
    ("0x41", "0xd87"): "Cortex-A725",
}


class Host:
    """Host CPU, memory, and disk identity for the current machine."""

    @cached_property
    def arch(self) -> str:
        """CPU architecture string, e.g. `x86_64` or `aarch64`."""
        return platform.machine().lower()

    @cached_property
    def cpu(self) -> str:
        """CPU model name from `platform` or `/proc/cpuinfo`."""
        if platform.system() == "Darwin" and (brand := shell.sysctl("machdep.cpu.brand_string")):
            return brand
        text = self.cpuinfo_text
        if not text:
            return platform.processor() or "unknown"
        if m := _CPU_MODEL_RE.search(text):
            return m.group(1).strip()
        if name := self.arm_cpu_name:
            return name
        return platform.processor() or "unknown"

    @cached_property
    def cpu_vendor(self) -> Vendor:
        """CPU core vendor inferred from OS identity records."""
        if platform.system() == "Darwin":
            return Vendor.APPLE
        for record in self.cpuinfo_records:
            if vendor := _VENDOR_BY_VENDOR_ID.get(record.get("vendor_id", "")):
                return vendor
            implementer = record.get("CPU implementer", "").lower()
            if vendor := _VENDOR_BY_IMPLEMENTER.get(implementer):
                return vendor
        return Vendor.UNKNOWN

    @cached_property
    def cpuinfo_text(self) -> str:
        """Raw Linux `/proc/cpuinfo`, or an empty string when unavailable."""
        with suppress(OSError):
            return Path("/proc/cpuinfo").read_text(encoding="utf-8")
        return ""

    @cached_property
    def cpuinfo_records(self) -> tuple[dict[str, str], ...]:
        """Parsed Linux `/proc/cpuinfo` records."""
        return tuple(
            {
                key.strip(): value.strip()
                for line in block.splitlines()
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            for block in self.cpuinfo_text.strip().split("\n\n")
            if block.strip()
        )

    @cached_property
    def arm_cpu_name(self) -> str:
        """Human-readable ARM CPU core mix from MIDR implementer and part IDs."""
        pairs = [
            (record.get("CPU implementer", "").lower(), record.get("CPU part", "").lower())
            for record in self.cpuinfo_records
            if record.get("CPU implementer") and record.get("CPU part")
        ]
        counts = Counter(pairs)
        names = []
        for implementer, part in dict.fromkeys(pairs):
            vendor = _ARM_IMPLEMENTERS.get(implementer, f"implementer {implementer}")
            model = _ARM_PARTS.get((implementer, part), f"part {part}")
            count = counts[(implementer, part)]
            names.append(f"{count}x {vendor} {model}" if count > 1 else f"{vendor} {model}")
        return " + ".join(names)

    @cached_property
    def logical_cpus(self) -> int:
        """Number of logical CPU threads (including hyperthreading)."""
        return psutil.cpu_count(logical=True) or 0

    @cached_property
    def physical_cpus(self) -> int:
        """Number of physical CPU cores."""
        return psutil.cpu_count(logical=False) or 0

    @property
    def cpu_freq_mhz(self) -> float | None:
        """Current CPU frequency in MHz; `None` if the platform cannot report it."""
        try:
            freq = psutil.cpu_freq()
        except AttributeError, NotImplementedError, OSError:
            return None
        return freq.current if freq else None

    @property
    def memory(self) -> Memory:
        """Live system RAM usage sampled from psutil."""
        return Memory.system()

    @property
    def memory_hardware(self) -> MemoryHardware:
        """Physical DIMM slots and swap space; lazily samples on first field access."""
        return MemoryHardware()

    @cached_property
    def disk(self) -> HostDisk:
        """All physical drives with their mounted partitions."""
        return HostDisk()

    @cached_property
    def cgroup_memory(self) -> CgroupMemory:
        """The enforced cgroup memory cap (v1 and v2), the real ceiling an OOM kill fires against.

        Walks both cgroup hierarchies up to the root and reports the tightest finite limit, or
        the host's total RAM when the job is uncapped. This is the ceiling a memory-bounded
        working set should size itself under, not psutil's free RAM (which on a coherent Grace
        Hopper OS double-counts HBM through the second NUMA node).
        """
        return CgroupMemory.probe()

    @cached_property
    def scratch(self) -> Scratch:
        """The fastest writable node-local scratch tier with its free space.

        The scheduler's node-local NVMe (`LOCALDIR`/`PBS_LOCALDIR`/`SLURM_TMPDIR`, then a bare
        local mount) a spill engine offloads to, or an unavailable tier when nothing is writable.
        """
        return Scratch.probe()
