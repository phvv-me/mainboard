# Which files of a project analysis reads: git's listing, less what the workspace's `[lint]`
# table excludes and what git marks vendored.

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=lists the project's files with git's fixed argv since=2026-10-13
from functools import cache
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec

from ....core.errors import NoWorkspace
from ....core.project import Project
from ....manifest.loading import load

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

_ANALYSIS_TOOL = "mb"


def listed(root: Path) -> list[str]:
    """The project's files git does not ignore, relative to `root`, submodules' included.

    Outside git (a scratch mirror) every file under the root counts. Files the workspace's `[lint]`
    table excludes (references, venue style bundles, datasets) and vendored ones are left out.
    """
    present = project_files(root)
    skipped = vendored(root, present)
    excluded = lint_excludes(root)
    return [name for name in present if name not in skipped and not excluded(name)]


def project_files(root: Path) -> list[str]:
    """The same source listing before the lint and vendor exclusions.

    Inventories that account for excluded code read it. Ignored untracked and build files are not
    part of the git listing.
    """
    # The user's global excludes file is left out, so every machine lists the same files.
    git = ["git", "-C", str(root), "-c", "core.excludesFile=/dev/null", "ls-files", "-z"]
    try:
        names = [
            name
            for flags in (["--cached", "--recurse-submodules"], ["--others", "--exclude-standard"])
            for name in subprocess.run([*git, *flags], capture_output=True, check=True)
            .stdout.decode()
            .split("\0")
        ]
    except subprocess.CalledProcessError:
        names = _walked(root)
    return [
        name
        for name in dict.fromkeys(names)
        if name and os.path.lexists(root / name) and not (root / name).is_dir()
    ]


@cache
def lint_excludes(root: Path) -> Callable[[str], bool]:
    """Whether the workspace's `[lint]` table leaves alone a file named relative to `root`.

    Its patterns are `exclude`, the analysis tool's own `exclude` and the generated-state folders,
    the very ones `mainboard lint` reads, from the topmost workspace above `root` (a project with a
    manifest of its own is linted under its parent's). Nothing is excluded outside a workspace.
    """
    project = Project()
    workspace = None
    while True:
        try:
            workspace = project.find_root(workspace.parent if workspace else root.resolve())
        except NoWorkspace:
            break
    if workspace is None:
        return lambda name: False
    table = load(project.manifest(workspace)).lint
    tool = table.tools.get(_ANALYSIS_TOOL)
    generated = (f"/{name}/" for name in project.out_dirs)
    spec = GitIgnoreSpec.from_lines([*generated, *table.exclude, *(tool.exclude if tool else ())])
    prefix = root.resolve().relative_to(workspace).as_posix()
    return lambda name: spec.match_file(name if prefix == "." else f"{prefix}/{name}")


def vendored(root: Path, paths: Iterable[str]) -> set[str]:
    """The `paths` under `root` that git marks `linguist-vendored`; none outside git.

    That is the standard mark for third-party code, which analysis neither reads nor reports. Paths
    are files or folders, relative to `root`.
    """
    return marked(root, paths, "linguist-vendored")


def marked(root: Path, paths: Iterable[str], attribute: str) -> set[str]:
    """The paths git positively marks with ATTRIBUTE; none outside git."""
    if not (paths := list(paths)):
        return set()
    run = subprocess.run(
        ["git", "-C", str(root), "check-attr", "-z", "--stdin", attribute],
        input="".join(f"{path}\0" for path in paths).encode(),
        capture_output=True,
        check=False,
    )
    if run.returncode:
        return set()
    fields = run.stdout.decode().split("\0")
    return {
        path
        for path, value in zip(fields[0::3], fields[2::3], strict=False)
        if value not in ("unspecified", "unset", "false")
    }


def _walked(root: Path) -> list[str]:
    """Every file under `root`, relative to it, outside git's, Python's and mainboard's folders."""
    names: list[str] = []
    for folder, folders, files in os.walk(root):
        folders[:] = [
            name for name in folders if name not in {".git", "__pycache__", ".mainboard"}
        ]
        names += [os.path.relpath(os.path.join(folder, name), root) for name in files]
    return names
