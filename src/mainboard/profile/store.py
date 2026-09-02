# Profiling output is generated data: a trace, a span table, a JSON report a bench script wrote.
# It belongs beside the other generated artifacts under the workspace's own directory rather than
# in a source tree, where it is neither reviewed nor reproducible.

from pathlib import Path

from ..core.project import Project


def profiles_dir(*parts: str, start: Path | None = None) -> Path:
    """The generated home for profiling output, created on demand.

    `<workspace>/.mainboard/profiles/<parts>`, so a bench script names its family (`cutoken`)
    and a host name or a date underneath it, and nothing it writes lands in a source tree.

    parts: the path underneath the profiles directory, none for the directory itself.
    start: the directory the workspace is found from, the current one when None.
    """
    project = Project()
    path = project.workspace(start) / project.out_dir / "profiles" / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
