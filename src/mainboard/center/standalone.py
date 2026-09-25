import ast
import os
import re
import tomllib
import warnings
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from plumbum import CommandNotFound, local
from plumbum.commands.processes import ProcessTimedOut

from ..core.errors import MissionError
from ..core.project import Project
from ..core.section import Section, Verdict
from ..engines.compile.backend.process import Process
from ..git.process import Git
from ..manifest.members import PYPROJECT

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from ..manifest.members import Composition, Member, Package
    from ..manifest.render.interpolate import Json
    from ..manifest.schema.environment import Task

# A fresh resolve of a GPU stack downloads gigabytes, and a solve that never ends is a finding.
_INSTALL_SECONDS = 1800.0

# A relative path starting with a climb (`../../packages/atpx`), which from a file at the member's
# top always leaves it, guarded against biting into a longer path or prose; and the manifest's
# own root spelled as a template, which a path may start from.
_CLIMB = re.compile(r"(?<![\w./+~@-])\.\.(?:/[\w.+~@-]+)*")
_CONFIG_ROOT = re.compile(r"\{\{\s*config_root\s*\}\}/?")

# Run inside the isolated environment: import every top-level name the member's distribution
# installs, and print them. Plain standard library, since nothing else is there to lean on.
_IMPORTS = """
import importlib, re, sys
from importlib.metadata import packages_distributions
wanted = sys.argv[1]
names = sorted(
    name
    for name, owners in packages_distributions().items()
    if not name.startswith("_")
    and wanted in {re.sub(r"[-_.]+", "-", owner).lower() for owner in owners}
)
for name in names:
    importlib.import_module(name)
print(" ".join(names))
"""

# The longest stretch of a tool's complaint one row carries.
_DETAIL_CHARS = 400


class Standalone:
    """Whether each member works for somebody who clones it alone, and every way it leans on us.

    A `fail` breaks that person: no installable project, no repository of its own to clone, a
    path climbing out of the member, a task depending on one it does not declare, an import only
    the monorepo satisfies (a sibling member it does not require, a directory the root's
    `PYTHONPATH` carries, or one beside a `src/` package, which installing leaves behind), or a
    clone that does not install and import with `uv` alone. A `warn`
    is coupling from our side: root-only settings the member declares, and root tasks, papers
    or variables reaching into it rather than living in it. A clean install is the one `pass`.
    """

    def __init__(self, composition: Composition) -> None:
        self.composition = composition
        self.root = composition.root

    def sections(self, names: Sequence[str] = ()) -> list[Section]:
        """Every finding for the members named by name or path, all of them when none is."""
        return [row for member in self._chosen(names) for row in self._judged(member)]

    def _chosen(self, names: Sequence[str]) -> list[Member]:
        members = self.composition.members
        if not members:
            raise MissionError("the workspace declares no `[workspace] members`")
        unknown = set(names) - {member.name for member in members} - {m.path for m in members}
        if unknown:
            raise MissionError(
                f"no member {sorted(unknown)[0]!r}; members are "
                f"{sorted(member.name for member in members)}"
            )
        return [m for m in members if not names or m.name in names or m.path in names]

    def _judged(self, member: Member) -> list[Section]:
        """One member's findings, the static ones first and the install, the slow one, last."""
        return [
            *self._shape(member),
            *self._ignored(member),
            *self._escapes(member),
            *self._depends(member),
            *self._imports(member),
            *self._reached(member),
            *self._installed(member),
        ]

    def _shape(self, member: Member) -> list[Section]:
        """Whether there is a project to install and a repository of its own to clone."""
        rows: list[Section] = []
        if member.package is None:
            rows.append(
                _row(
                    member,
                    "project",
                    Verdict.FAIL,
                    f"has no {PYPROJECT} [project], so nobody can install it alone",
                    fix=f"write a {PYPROJECT} whose [project] names its dependencies",
                )
            )
        if not self._repository(member):
            rows.append(
                _row(
                    member,
                    "clone",
                    Verdict.FAIL,
                    "is not a repository of its own, so nobody can clone it alone",
                    fix="make it a submodule with its own remote",
                )
            )
        return rows

    @staticmethod
    def _ignored(member: Member) -> list[Section]:
        """The root-only settings it declares, which govern it only when it stands alone."""
        if not member.ignored:
            return []
        detail = (
            f"declares {', '.join(member.ignored)}, which apply only when it stands alone; "
            "the root's govern it here"
        )
        return [_row(member, "standalone", Verdict.WARN, detail)]

    def _escapes(self, member: Member) -> list[Section]:
        """Every path its own files spell that climbs out of it, which a clone does not have."""
        rows: list[Section] = []
        for name in (PYPROJECT, Project().manifest):
            try:
                text = (self.root / member.path / name).read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            for key, value in _strings(tomllib.loads(text)):
                rows.extend(
                    _row(
                        member,
                        "escape",
                        Verdict.FAIL,
                        f"{name} {key} reaches {token}, outside the member",
                        fix="keep what it names inside the member, or require it by version "
                        "or git URL, which the workspace resolves locally",
                    )
                    for token in _CLIMB.findall(_CONFIG_ROOT.sub("", value))
                )
        return rows

    @staticmethod
    def _depends(member: Member) -> list[Section]:
        """Every task depending on a task the member itself does not declare."""
        if member.manifest is None:
            return []
        tables = [member.manifest.tasks, *(env.tasks for env in member.manifest.envs.values())]
        return [
            _row(
                member,
                "task",
                Verdict.FAIL,
                f"{name} depends on {need}, which the member does not declare",
                fix=f"declare {need} in {member.path}/{Project().manifest}",
            )
            for table in tables
            for name, spec in table.items()
            if isinstance(spec, dict)
            for need in spec.get("depends", [])
            if need not in member.manifest.tasks and need not in table
        ]

    def _imports(self, member: Member) -> list[Section]:
        """Every top-level import only the monorepo satisfies, one row per imported name.

        Under a `src/` layout the installed package carries only `src/`, so a module there
        importing a directory beside it (`experiments`) breaks as soon as it is installed.
        """
        directory = self.root / member.path
        installed = _provided(directory / "src")
        beside = _provided(directory) - installed
        sources = self._sources(member)
        found: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for path in self._python_files(member):
            shipped = path.startswith("src/")
            for name in _imported(directory / path):
                if name in beside and shipped:
                    where = "from beside src/, which the installed package does not carry"
                    found[name, where, "move it into the package"].append(path)
                elif name not in installed | beside and name in sources:
                    found[name, *sources[name]].append(path)
        return [
            _row(
                member,
                "import",
                Verdict.FAIL,
                f"imports {name} {where}; first of {len(paths)} files: {paths[0]}",
                fix=fix,
            )
            for (name, where, fix), paths in sorted(found.items())
        ]

    def _sources(self, member: Member) -> dict[str, tuple[str, str]]:
        """Each name only the monorepo provides `member`, with where from and the repair.

        A sibling member's package is provided honestly once `member` requires it; a directory
        on the root's `PYTHONPATH` never is.
        """
        sources: dict[str, tuple[str, str]] = {}
        for entry in self._python_path():
            where = f"from {entry}, which only the root's PYTHONPATH provides"
            fix = "move it into the member, or into a member it requires"
            sources |= dict.fromkeys(_provided(self.root / entry), (where, fix))
        requires = member.package.requires if member.package else {}
        for sibling in self.composition.members:
            if sibling is member or (sibling.package and sibling.package.name in requires):
                continue
            where, fix = _unrequired(sibling)
            sources |= dict.fromkeys(_packages(self.root / sibling.path, sibling), (where, fix))
        return sources

    def _python_path(self) -> list[str]:
        """The root `[env] PYTHONPATH` entries that lie inside the workspace, root-relative."""
        declared = self.composition.manifest.env.get("PYTHONPATH")
        if not isinstance(declared, str):
            return []
        entries = [Path(entry) for entry in declared.split(os.pathsep) if entry]
        inside = [self.root / entry for entry in entries]
        return [
            entry.relative_to(self.root).as_posix()
            for entry in inside
            if entry.is_relative_to(self.root)
        ]

    def _python_files(self, member: Member) -> list[str]:
        """Every Python file git tracks or would track in the member, member-relative."""
        listing = Git(self.root / member.path).out(
            "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"
        )
        return [path for path in listing.split("\0") if path]

    def _reached(self, member: Member) -> list[Section]:
        """Root tasks, papers and variables naming the member's directory, one row per kind."""
        mention = re.compile(rf"(?<![\w.-]){re.escape(member.path)}(?![\w.-])")
        root = self.composition.manifest
        declared: dict[str, Mapping[str, Task | bool]] = {
            "tasks": {
                **root.tasks,
                **{n: t for e in root.envs.values() for n, t in e.tasks.items()},
            },
            "papers": {name: paper.dir for name, paper in root.papers.items()},
            "[env]": root.env,
        }
        rows: list[Section] = []
        for kind, entries in declared.items():
            reaching = [name for name, value in entries.items() if mention.search(str(value))]
            if reaching:
                fix = (
                    "install the member rather than import it by path"
                    if kind == "[env]"
                    else f"declare them in {member.path}/{Project().manifest}"
                )
                detail = f"root {kind} {', '.join(reaching)} reach into it"
                rows.append(_row(member, "root", Verdict.WARN, detail, fix=fix))
        return rows

    def _installed(self, member: Member) -> list[Section]:
        """Clone it alone, then install and import it with nothing of the monorepo around."""
        if member.package is None or not self._repository(member):
            return []
        with TemporaryDirectory(prefix="mainboard-member-", ignore_cleanup_errors=True) as scratch:
            clone = Path(scratch) / member.name
            cloned = Git(self.root).run("clone", "-q", str(self.root / member.path), str(clone))
            if not cloned.succeeded:
                return [_row(member, "install", Verdict.FAIL, _squashed(cloned.stderr))]
            return [self._imported_alone(member, member.package, clone)]

    @staticmethod
    def _imported_alone(member: Member, package: Package, clone: Path) -> Section:
        """Whether `uv` installs the clone into an empty environment and every package imports."""
        python = ["--python", package.python] if package.python else []
        argv = ["run", "--isolated", "--no-project", *python, "--with", str(clone)]
        try:
            with local.cwd(str(clone.parent)):
                command = local["uv"][(*argv, "python", "-c", _IMPORTS, package.name)]
                done = Process.capture(command, timeout=_INSTALL_SECONDS)
        except CommandNotFound:
            return _row(member, "install", Verdict.FAIL, "uv is not on PATH", fix="install uv")
        except ProcessTimedOut:
            detail = f"uv gave no answer in {_INSTALL_SECONDS:.0f} s"
            return _row(member, "install", Verdict.FAIL, detail)
        names = done.stdout.split()
        if not done.succeeded:
            return _row(member, "install", Verdict.FAIL, _squashed(done.stderr))
        if not names:
            detail = f"{package.name} installs alone but provides no importable package"
            return _row(member, "install", Verdict.FAIL, detail)
        detail = f"{package.name} installs alone and imports {', '.join(names)}"
        return _row(member, "install", Verdict.PASS, detail)

    def _repository(self, member: Member) -> bool:
        return (self.root / member.path / ".git").exists()


def _row(member: Member, check: str, verdict: Verdict, detail: str, *, fix: str = "") -> Section:
    return Section(section=f"{member.path}: {check}", verdict=verdict, detail=detail, fix=fix)


def _unrequired(sibling: Member) -> tuple[str, str]:
    """Where a sibling's package comes from when the importer does not require it, and why."""
    if sibling.package is None:
        return f"from {sibling.path}, which has no installable project", "make it installable"
    return (
        f"from {sibling.path}, whose {sibling.package.name} its {PYPROJECT} does not require",
        f"add {sibling.package.name} to its dependencies",
    )


def _strings(value: Json, key: str = "") -> Iterator[tuple[str, str]]:
    """Every string in a parsed TOML tree, beside its dotted key."""
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _strings(item, f"{key}.{name}" if key else name)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, key)


def _provided(directory: Path) -> set[str]:
    """The top-level modules and packages directly in `directory`, none when it is missing."""
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return set()
    return {
        entry.stem
        for entry in entries
        if entry.stem.isidentifier()
        and ((entry.suffix == ".py" and entry.is_file()) or (entry / "__init__.py").is_file())
    }


def _packages(directory: Path, member: Member) -> set[str]:
    """The top-level packages `member`'s distribution installs, as far as its layout says.

    A `src/` layout installs what `src/` holds; a flat one is read as installing the package
    named after its distribution, since tests, examples and scripts beside it install nothing.
    """
    if (directory / "src").is_dir():
        return _provided(directory / "src")
    named = member.package.name.replace("-", "_") if member.package else ""
    return {named} & _provided(directory)


def _imported(path: Path) -> set[str]:
    """The top-level names a Python file imports absolutely, none when it is gone or unparsable."""
    try:
        with warnings.catch_warnings(action="ignore", category=SyntaxWarning):
            tree = ast.parse(path.read_bytes())
    except FileNotFoundError, SyntaxError, ValueError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.partition(".")[0])
    return names


def _squashed(text: str) -> str:
    """A tool's complaint as one line a table cell can hold."""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())[-_DETAIL_CHARS:]
