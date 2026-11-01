# Which files of a project analysis reads: git's listing, less what the workspace's `[lint]`
# table excludes and what git marks vendored.

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=lists the project's files with git's fixed argv since=2026-10-13
import threading
from functools import cache, wraps
from typing import TYPE_CHECKING, ClassVar

from pathspec import GitIgnoreSpec

from ....core.errors import NoWorkspace
from ....core.project import Project
from ....manifest.loading import load

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from types import TracebackType

_ANALYSIS_TOOL = "mb"

type Lister = Callable[[Path], list[str]]


class Listing:
    """The files of one project, listed once while its fact base is derived.

    Providers, tools and rules each ask what the project holds. `listed` and `project_files`
    answer from the listing open for their root, so the whole derivation costs one `git ls-files`
    pair and one `git check-attr`. A listing lives only for the derivation: a fix writes new files
    between one fact base and the next. A verb that writes nothing opens one around its whole run,
    and the derivation inside joins it, so the findings and the files it records cost no more git.
    """

    active: ClassVar[dict[Path, Listing]] = {}

    @classmethod
    def of(cls, root: Path) -> Listing | None:
        """The listing open for `root`, if any."""
        return cls.active.get(root.resolve())

    def __enter__(self) -> Listing:
        """The listing open for the root: this one, or the one this joins."""
        current = Listing.active.setdefault(self.root, self)
        self.opened = current is self
        return current

    def __exit__(
        self,
        kind: type[BaseException] | None,
        raised: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self.opened:
            del Listing.active[self.root]
            self.opened = False

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.answers: dict[Lister, list[str]] = {}
        self.lock = threading.RLock()
        self.opened = False

    def answer(self, lister: Lister) -> list[str]:
        """What `lister` says of the root, asked once, a copy for each caller."""
        with self.lock:
            if lister not in self.answers:
                self.answers[lister] = lister(self.root)
            return list(self.answers[lister])


def _shared(lister: Lister) -> Lister:
    """`lister` answering from the listing open for its root, when there is one."""

    @wraps(lister)
    def answer(root: Path) -> list[str]:
        return current.answer(lister) if (current := Listing.of(root)) else lister(root)

    return answer


@_shared
def listed(root: Path) -> list[str]:
    """The project's files git does not ignore, relative to `root`, submodules' included.

    Outside git (a scratch mirror) every file under the root counts. Files the workspace's `[lint]`
    table excludes (references, venue style bundles, datasets) and vendored ones are left out.
    """
    present = project_files(root)
    skipped = vendored(root, present)
    excluded = lint_excludes(root)
    return [name for name in present if name not in skipped and not excluded(name)]


@_shared
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
    skipped = {".git", "__pycache__", *Project().out_dirs}
    names: list[str] = []
    for folder, folders, files in os.walk(root):
        folders[:] = [name for name in folders if name not in skipped]
        names += [os.path.relpath(os.path.join(folder, name), root) for name in files]
    return names
