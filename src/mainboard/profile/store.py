from pathlib import Path

from ..core.project import Project


def profiles_dir(*parts: str, start: Path | None = None) -> Path:
    """`<workspace>/<state directory>/profiles/<parts>`, created on demand.

    Profiling output is generated data, so it lives beside the other generated artifacts and
    never in a source tree, where it is neither reviewed nor reproducible.

    parts: e.g. a bench family (`cutoken`) then a host or date; none for the directory itself.
    start: the directory the workspace is found from, the current one when None.
    """
    project = Project()
    path = project.out(project.workspace(start)) / "profiles" / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
