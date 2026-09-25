import os
import shutil
from pathlib import Path
from tempfile import gettempdir

from patos import FrozenModel

_SCRATCH_ENV = ("LOCALDIR", "PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR", "TEMP", "TMP")
_SCRATCH_DIRS = ("/local", "/scratch/local", "/tmp")  # noqa: S108  reason=fixed cluster-convention scratch roots, not attacker input since=2026-08-16


class Scratch(FrozenModel):
    """The host's fastest writable node-local scratch tier, with its free space.

    The scheduler-provided node-local NVMe a spill engine offloads to: the first existing,
    writable path among the PBS/SLURM env vars, then the bare local mounts, then the system temp
    dir. When none is writable the path is `None`, so a caller can tell node-local NVMe from a
    shared filesystem rather than guessing a directory.

    free_bytes: free on the chosen directory's filesystem, `0` when there is no path.
    source: the env var, mount or `system-temp` the path came from, for diagnostics.
    """

    path: Path | None = None
    free_bytes: int = 0
    source: str = ""

    @property
    def available(self) -> bool:
        """Whether a writable node-local scratch tier was found."""
        return self.path is not None

    @property
    def free_gb(self) -> float:
        """Free space on the scratch tier in gibibytes."""
        return self.free_bytes / 1024**3

    @classmethod
    def probe(cls) -> Scratch:
        """The first writable candidate, or the unavailable tier."""
        candidates = (
            *((key, os.environ[key]) for key in _SCRATCH_ENV if key in os.environ),
            *((mount, mount) for mount in _SCRATCH_DIRS),
            ("system-temp", gettempdir()),
        )
        for source, candidate in candidates:
            path = Path(candidate)
            if path.is_dir() and os.access(path, os.W_OK):
                return cls(path=path, free_bytes=shutil.disk_usage(path).free, source=source)
        return cls()
