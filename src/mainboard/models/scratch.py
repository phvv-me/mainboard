import os
import shutil
from pathlib import Path

from .base import FrozenModel

# the node-local fast-scratch candidates, in cluster-convention order. A scheduler points a job
# at its node-local NVMe through one of these env vars (PBS sets `PBS_LOCALDIR`/`LOCALDIR`, SLURM
# `SLURM_TMPDIR`); when none is set, the bare local mounts and finally the process temp dir are
# tried. The first that exists and is writable is the scratch tier a spill engine streams to,
# fast node-local NVMe rather than a shared Lustre path.
SCRATCH_ENV = ("LOCALDIR", "PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR")
SCRATCH_DIRS = ("/local", "/scratch/local", "/tmp")  # noqa: S108


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

    @classmethod
    def probe(cls) -> Scratch:
        """The first writable node-local scratch dir among the env vars then the local mounts."""
        env_candidates = [(key, os.environ[key]) for key in SCRATCH_ENV if key in os.environ]
        literal_candidates = [(candidate, candidate) for candidate in SCRATCH_DIRS]
        for source, candidate in (*env_candidates, *literal_candidates):
            path = Path(candidate)
            if path.is_dir() and os.access(path, os.W_OK):
                return cls(path=path, free_bytes=shutil.disk_usage(path).free, source=source)
        return cls()

    @property
    def available(self) -> bool:
        """Whether a writable node-local scratch tier was found."""
        return self.path is not None

    @property
    def free_gb(self) -> float:
        """Free space on the scratch tier in gibibytes."""
        return self.free_bytes / 1024**3
