from contextlib import suppress
from pathlib import Path

import psutil

from .base import FrozenModel

# the cgroup filesystem root and the per-process membership file. A line in `/proc/self/cgroup`
# is `hierarchy-id:controllers:path`. cgroup-v2 writes a single unified line whose hierarchy id
# is 0 and whose controllers field is empty (`0::/path`); cgroup-v1 writes one line per
# controller, so the memory cap lives on the line whose controllers field names `memory`
# (`N:memory:/path`). The cap files differ by version too: v2 keeps one `memory.max` per node
# under `<root>/<path>`, while v1 keeps `memory.limit_in_bytes` (RAM) and the memsw
# `memory.memsw.limit_in_bytes` (RAM + swap, the ceiling a `Cgroup memsw limit exceeded` OOM
# fires against) under `<root>/memory/<path>`.
CGROUP_PROC = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
V1_FILES = ("memory.limit_in_bytes", "memory.memsw.limit_in_bytes")
V2_FILE = "memory.max"


class CgroupMemory(FrozenModel):
    """The memory ceiling a job actually runs inside, read off the Linux cgroup tree.

    A scheduler (PBS, SLURM) caps a job's memory by writing the limit onto a cgroup the job
    shares, usually an ANCESTOR of the process's own leaf cgroup (the jobid scope), so the
    enforced ceiling is the tightest finite value found walking from the process cgroup up to
    the root. Both cgroup versions are read: v1 over `memory.limit_in_bytes` and the memsw
    `memory.memsw.limit_in_bytes` (the RAM + swap ceiling Miyabi's GH200 PBS enforces), v2 over
    `memory.max`. When no level carries a finite cap, the job is uncapped and `limit_bytes` is
    the host's total RAM, so a caller always reads a finite ceiling to size a working set under.

    limit_bytes: the tightest finite enforced cap in bytes, or the host total RAM when uncapped.
    capped: whether a real cgroup limit was found (`False` means the host total RAM is reported).
    """

    limit_bytes: int = 0
    capped: bool = False

    @classmethod
    def probe(cls) -> CgroupMemory:
        """Read the enforced cap from the cgroup tree, falling back to host RAM when uncapped."""
        if (limit := cls.enforced_limit()) is not None:
            return cls(limit_bytes=limit, capped=True)
        return cls(limit_bytes=psutil.virtual_memory().total, capped=False)

    @classmethod
    def enforced_limit(cls) -> int | None:
        """The tightest finite cgroup memory limit on this process, or `None` when uncapped.

        Resolves the process's cgroup membership from `/proc/self/cgroup`, then walks the
        relevant node up to the root collecting every finite cap file value, and returns the
        smallest. Returns `None` when `/proc/self/cgroup` is unreadable or no level is capped.
        """
        membership = cls.read_membership()
        if membership is None:
            return None
        node, cap_files = membership
        caps = [cap for level in cls.ancestors(node) for cap in cls.read_caps(level, cap_files)]
        return min(caps) if caps else None

    @classmethod
    def read_membership(cls) -> tuple[Path, tuple[str, ...]] | None:
        """The starting cgroup node and its version's cap files, or `None` when unreadable.

        v2's unified line wins when present (`0::/path` under `<root>/<path>`, `memory.max`);
        otherwise the v1 memory-controller line is used (`<root>/memory/<path>`, the two
        `limit_in_bytes` files). Returns `None` when the membership file cannot be read.
        """
        try:
            lines = CGROUP_PROC.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        records = [line.split(":", 2) for line in lines if line.count(":") >= 2]
        for hierarchy, controllers, path in records:
            if hierarchy == "0" and controllers == "":
                return CGROUP_ROOT / path.strip().lstrip("/"), (V2_FILE,)
        for _, controllers, path in records:
            if "memory" in controllers.split(","):
                return CGROUP_ROOT / "memory" / path.strip().lstrip("/"), V1_FILES
        return None

    @staticmethod
    def ancestors(node: Path) -> list[Path]:
        """The node and every parent up to and including `CGROUP_ROOT`."""
        chain = [node]
        while chain[-1] != CGROUP_ROOT and CGROUP_ROOT in chain[-1].parents:
            chain.append(chain[-1].parent)
        return chain

    @staticmethod
    def read_caps(node: Path, cap_files: tuple[str, ...]) -> list[int]:
        """Every finite cap, in bytes, among `cap_files` on one cgroup node.

        A file is finite when it holds a positive integer below the kernel's `unlimited`
        sentinel (v2's literal `max`, or v1's near-`2**63` page-aligned default). Missing or
        unreadable files contribute nothing, so a sparsely populated node never raises.
        """
        unlimited = 1 << 62
        caps: list[int] = []
        for name in cap_files:
            with suppress(OSError, ValueError):
                raw = (node / name).read_text(encoding="utf-8").strip()
                if raw != "max" and 0 < (value := int(raw)) < unlimited:
                    caps.append(value)
        return caps

    @property
    def limit_gb(self) -> float:
        """The enforced cap in gibibytes."""
        return self.limit_bytes / 1024**3
