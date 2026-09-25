from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import psutil
from patos import FrozenModel

_CGROUP_PROC = Path("/proc/self/cgroup")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_V1_FILES = ("memory.limit_in_bytes", "memory.memsw.limit_in_bytes")
_V2_FILE = "memory.max"


class CgroupMemory(FrozenModel):
    """The memory ceiling a job actually runs inside, read off the Linux cgroup tree.

    A scheduler (PBS, SLURM) writes the limit onto a cgroup the job shares, usually an ANCESTOR
    of the process's own leaf (the jobid scope), so the enforced ceiling is the tightest finite
    cap walking from the process cgroup up to the root. v1 caps are `memory.limit_in_bytes` and
    the memsw `memory.memsw.limit_in_bytes` (the RAM + swap ceiling Miyabi's GH200 PBS enforces),
    v2's is `memory.max`.

    limit_bytes: the tightest finite cap, or the host total RAM when uncapped, so a caller always
        reads a finite ceiling to size a working set under.
    capped: whether a real cgroup limit was found.
    """

    limit_bytes: int = 0
    capped: bool = False

    @property
    def limit_gb(self) -> float:
        """The enforced cap in gibibytes."""
        return self.limit_bytes / 1024**3

    @staticmethod
    def ancestors(node: Path) -> list[Path]:
        """The node and every parent up to and including `_CGROUP_ROOT`."""
        chain = [node]
        while chain[-1] != _CGROUP_ROOT and _CGROUP_ROOT in chain[-1].parents:
            chain.append(chain[-1].parent)
        return chain

    @staticmethod
    def read_caps(node: Path, cap_files: Sequence[str]) -> list[int]:
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

    @classmethod
    def enforced_limit(cls) -> int | None:
        """The tightest finite cgroup memory limit on this process, `None` when uncapped.

        `/proc/self/cgroup` being unreadable also answers `None`.
        """
        if (membership := cls.read_membership()) is None:
            return None
        node, cap_files = membership
        caps = (cap for level in cls.ancestors(node) for cap in cls.read_caps(level, cap_files))
        return min(caps, default=None)

    @classmethod
    def probe(cls) -> CgroupMemory:
        """Read the enforced cap from the cgroup tree, falling back to host RAM when uncapped."""
        if (limit := cls.enforced_limit()) is not None:
            return cls(limit_bytes=limit, capped=True)
        return cls(limit_bytes=psutil.virtual_memory().total, capped=False)

    @classmethod
    def read_membership(cls) -> tuple[Path, tuple[str, ...]] | None:
        """The starting cgroup node and its version's cap files, `None` when unreadable.

        v2's unified `0::/path` line wins when present (node `<root>/<path>`); otherwise the v1
        `N:memory:/path` line is used (node `<root>/memory/<path>`).
        """
        try:
            lines = _CGROUP_PROC.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        records = [line.split(":", 2) for line in lines if line.count(":") >= 2]
        for hierarchy, controllers, path in records:
            if hierarchy == "0" and controllers == "":
                return _CGROUP_ROOT / path.strip().lstrip("/"), (_V2_FILE,)
        for _, controllers, path in records:
            if "memory" in controllers.split(","):
                return _CGROUP_ROOT / "memory" / path.strip().lstrip("/"), _V1_FILES
        return None
