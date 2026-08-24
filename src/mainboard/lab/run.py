from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.project import Project
from ..profile.spans import Span
from ..profile.spans import span as profile_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from .experiment import DeclaredExperiment, Fixture


def default_dataset_resolver(name: str) -> Path:
    """The hardware-free default: a dataset resolves under the project's staged-data cache."""
    return Path(Project().out_dir) / "data" / name


@dataclass(frozen=True, slots=True)
class Run:
    """The per-trial context `runnable` hands to `setup` and `measure`.

    model_id: the model this trial runs against.
    config: the validated experiment instance this trial was built from.
    artifact_dir: the directory this trial's artifacts are written under.
    dataset_resolver: resolves a declared dataset name to its local path, injected so a test
        never touches a real cache.
    """

    model_id: str
    config: DeclaredExperiment
    artifact_dir: Path
    dataset_resolver: Callable[[str], Path] = default_dataset_resolver

    def artifact(
        self, name: str, writer: Callable[[Path, Fixture], None], payload: Fixture
    ) -> Path:
        """Write one named artifact under `artifact_dir`, returning its path.

        name: the artifact's filename, relative to `artifact_dir`.
        writer: called as `writer(path, payload)` to actually serialize the payload; mainboard
            owns no serialization format, the caller's `writer` does.
        payload: whatever `writer` knows how to write.
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / name
        writer(path, payload)
        return path

    def dataset(self, name: str) -> Path:
        """Resolve `name` to its local path through this run's injected `dataset_resolver`."""
        return self.dataset_resolver(name)

    def span(self, name: str) -> Span:
        """Open a named profiling span, dormant when no `Profiler` is active."""
        return profile_span(name)
