import os
import shutil
from pathlib import Path

from patos import FrozenModel

# Node-local fast-scratch candidates, in cluster-convention order: PBS/SLURM env vars first,
# then the bare local mounts. The first that exists and is writable wins.
_SCRATCH_ENV = ("LOCALDIR", "PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR")
_SCRATCH_DIRS = ("/local", "/scratch/local", "/tmp")  # noqa: S108  reason=fixed cluster-convention scratch roots, not attacker input since=2026-08-16


class Scratch(FrozenModel):
    """The host's fastest writable node-local scratch tier, with its free space.

    The scheduler-provided node-local NVMe a spill engine offloads to, resolved by probing the
    cluster-convention env vars first and the bare local mounts second, taking the first path
    that exists and is writable. When nothing is writable (no env var set, no local mount), the
    path is `None` and the tier is unavailable, so a caller can tell node-local NVMe from a
    shared filesystem rather than guessing a directory.

    path: the chosen node-local scratch directory, or `None` when no candidate is writable.
    free_bytes: bytes free on the chosen directory's filesystem, `0` when there is no path.
    source: the env var or literal mount the path came from, for diagnostics.
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
        """The first writable node-local scratch dir among the env vars then the local mounts."""
        env_candidates = [(key, os.environ[key]) for key in _SCRATCH_ENV if key in os.environ]
        literal_candidates = [(candidate, candidate) for candidate in _SCRATCH_DIRS]
        for source, candidate in (*env_candidates, *literal_candidates):
            path = Path(candidate)
            if path.is_dir() and os.access(path, os.W_OK):
                return cls(path=path, free_bytes=shutil.disk_usage(path).free, source=source)
        return cls()
