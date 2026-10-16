"""Who uses a project's names from outside it.

One repository's run sees the imports inside the project, never the ones in the rest of the
monorepo, so a rule that calls a public name dead (the reach rules) or a fix that renames or
removes one would break the projects that import it. The consumers of a project are the other
repositories its workspace owns: the topmost working tree's `[lint].owners` minus its
`[lint].exclude` (forks) and the `linguist-vendored` trees, and minus any owner holding a package
of the same name (a copy or the original of the project itself).

A consumer's imports are found by text first (`rg` lists the files that import one of the project's
top-level packages), then only those files are parsed, with the engine's own syntax-tree provider,
and read as the pairs `(module, name)` they import or reach through an alias. Names count by the
module's path inside the project, so `from patos.sql import Column` uses the `Column` declared in
`src/patos/sql/columns.py` (a re-export is followed no further: a name counts as used when any
module on its way is imported). The index is built on first use, once per run.
"""

import ast
import hashlib
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs ripgrep with fixed argv since=2026-10-16
import threading
import time
import tomllib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec

from ..lint.git import topmost
from ..log import logger
from .facts import Budget, Facts, listed, vendored
from .providers.ducks import Ducks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .apply import Plan

# The name Ruff's F401 says is imported and unused (`a.b.c` imported but unused: `c`), none for
# any other code.
_UNUSED_IMPORT = (
    "CASE WHEN code = 'F401' "
    "THEN regexp_extract(regexp_extract(message, '`([^`]+)`', 1), '[^.]+$') END"
)

# The SQL naming what a finding declares, for each rule that calls a name dead or a unit
# redundant: the name a consumer would import (a module's stem for the rules in MODULES).
# A finding about a route (an `__all__` entry, an unread import) is used only by an import from
# that very module (EXPORTS); a definition is also used through any package re-exporting it.
DECLARED = {
    "ALL-REAC0001": "regexp_extract(qualname, '[^.]+$')",
    "ALL-REAC0002": "regexp_extract(qualname, '[^.]+$')",
    "PY-MODU0005": "public_export_name",
    "PY-DEAD0001": "name",
    "PY-RUFF0001": _UNUSED_IMPORT,
    "PY-RUFF0002": _UNUSED_IMPORT,
    "ALL-FUNC0002": "name",
    "ALL-FUNC0003": "name",
    "ALL-FUNC0006": "name",
    "ALL-ARCH0005": "regexp_extract(path, '([^/]+?)(/__init__)?\\.pyi?$', 1)",
}
MODULES = frozenset({"ALL-ARCH0005"})
EXPORTS = frozenset({"PY-MODU0005", "PY-RUFF0001", "PY-RUFF0002"})
_SCRIPTS = frozenset({"conftest", "setup", "__main__", "__init__"})
_ROOTS = ((), ("src",))

_IMPORTS = """
    WITH statements AS (
      SELECT fact_id, path, node_id, type, name FROM ast_nodes
      WHERE type IN ('import_statement', 'import_from_statement')
    ), keyword AS (
      SELECT n.fact_id, n.parent_id AS statement, min(n.sibling_index) AS position
      FROM ast_nodes n JOIN statements s ON s.fact_id = n.fact_id AND s.node_id = n.parent_id
      WHERE n.type = 'import' GROUP BY ALL
    ), bound AS (
      SELECT s.path, s.type AS kind, s.name AS module, p.type AS part, p.name AS spelled,
             coalesce(d.name, p.name) AS imported
      FROM statements s
      JOIN keyword k ON k.fact_id = s.fact_id AND k.statement = s.node_id
      JOIN ast_nodes p ON p.fact_id = s.fact_id AND p.parent_id = s.node_id
        AND p.sibling_index > k.position
        AND p.type IN ('dotted_name', 'aliased_import', 'wildcard_import')
      LEFT JOIN ast_nodes d ON d.fact_id = p.fact_id AND d.parent_id = p.node_id
        AND p.type = 'aliased_import' AND d.type = 'dotted_name'
      WHERE s.type = 'import_statement' OR NOT starts_with(s.name, '.')
    ), imports AS (
      SELECT path,
             CASE kind WHEN 'import_statement' THEN imported ELSE module END AS module,
             CASE WHEN kind = 'import_statement' THEN ''
                  WHEN part = 'wildcard_import' THEN '*' ELSE imported END AS name,
             CASE part WHEN 'wildcard_import' THEN NULL
                       WHEN 'aliased_import' THEN spelled
                       WHEN 'dotted_name' THEN CASE kind WHEN 'import_statement'
                         THEN split_part(imported, '.', 1) ELSE imported END END AS local,
             CASE kind WHEN 'import_statement'
                       THEN CASE part WHEN 'aliased_import' THEN imported
                                      ELSE split_part(imported, '.', 1) END
                       ELSE module || '.' || imported END AS target
      FROM bound
    )
"""
_PAIRS = _IMPORTS + "SELECT DISTINCT module, name, min(path) FROM imports GROUP BY ALL"
_CHAINS = (
    _IMPORTS
    + """
    , chains AS (
      SELECT path, regexp_extract(name, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*') AS dotted
      FROM ast_nodes WHERE type = 'attribute'
    )
    SELECT DISTINCT i.target || substr(a.dotted, length(i.local) + 1) AS chain, min(a.path)
    FROM chains a
    JOIN (SELECT DISTINCT path, local, target FROM imports WHERE local IS NOT NULL) i
      ON i.path = a.path AND split_part(a.dotted, '.', 1) = i.local
    WHERE contains(a.dotted, '.')
    GROUP BY ALL
"""
)


def module_key(path: str) -> str:
    """A Python file's path without its suffix and without `/__init__`: `src/patos/sql`."""
    key = PurePosixPath(path).with_suffix("").as_posix()
    return key.removesuffix("/__init__")


def surface(data: bytes) -> set[str]:
    """The names a module binds at its top level, inside its `if`, `try`, `with`, `for` and
    `while` blocks too."""
    names: set[str] = set()

    def bind(target: ast.expr) -> None:
        match target:
            case ast.Name(id=name):
                names.add(name)
            case ast.Tuple(elts=items) | ast.List(elts=items):
                for item in items:
                    bind(item)
            case ast.Starred(value=inner):
                bind(inner)

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            match node:
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                    names.add(name)
                case ast.ClassDef(name=name):
                    names.add(name)
                case ast.Assign(targets=targets):
                    for target in targets:
                        bind(target)
                case ast.AnnAssign(target=target) | ast.AugAssign(target=target):
                    bind(target)
                case ast.TypeAlias(name=target):
                    bind(target)
                case ast.Import(names=aliases):
                    names.update(a.asname or a.name.split(".")[0] for a in aliases)
                case ast.ImportFrom(names=aliases):
                    names.update(a.asname or a.name for a in aliases)
                case ast.If() | ast.For() | ast.AsyncFor() | ast.While():
                    visit(node.body)
                    visit(node.orelse)
                case ast.With() | ast.AsyncWith():
                    visit(node.body)
                case ast.Try() | ast.TryStar():
                    for block in (node.body, node.orelse, node.finalbody):
                        visit(block)
                    for handler in node.handlers:
                        visit(handler.body)

    visit(ast.parse(data).body)
    return names


def positional(data: bytes) -> dict[tuple[str, str], int]:
    """How many parameters each top-level function takes by position, and each method of a
    top-level class, keyed by (name, method) with an empty method for a function."""
    counts: dict[tuple[str, str], int] = {}

    def count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        return len(node.args.posonlyargs) + len(node.args.args)

    for node in ast.parse(data).body:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                counts[node.name, ""] = count(node)
            case ast.ClassDef():
                for member in node.body:
                    if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                        counts[node.name, member.name] = count(member)
    return counts


class Workspace:
    """The repositories next to a project: where its consumers live.

    root: the project; its workspace is the topmost working tree that holds it.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.top = topmost(self.root)

    def packages(self) -> list[str]:
        """The top-level packages the project defines (directories holding an `__init__.py`, at its
        root or in `src/`), and the modules lying there."""
        files = [PurePosixPath(name) for name in listed(self.root) if name.endswith(".py")]
        names = {
            file.parent.name if file.name == "__init__.py" else file.stem
            for file in files
            if file.parent.parts in _ROOTS
            and file.stem not in _SCRIPTS
            or file.name == "__init__.py"
            and file.parent.parent.parts in _ROOTS
        }
        return sorted(name for name in names if name.isidentifier())

    def submodules(self) -> frozenset[str]:
        """Every trailing part of the paths of the project's Python modules."""
        names = [name for name in listed(self.root) if name.endswith((".py", ".pyi"))]
        keys = [module_key(name).split("/") for name in names]
        return frozenset("/".join(parts[start:]) for parts in keys for start in range(len(parts)))

    def consumers(self, packages: list[str]) -> list[str]:
        """The other owned repositories, relative to the workspace: its manifest's owners minus
        the excluded and vendored ones, the project's own and any holding a package of the
        project."""
        if self.top is None:
            return []
        try:
            table = tomllib.loads((self.top / "mainboard.toml").read_text()).get("lint", {})
        except FileNotFoundError:
            return []
        excluded = GitIgnoreSpec.from_lines(table.get("exclude", ()))
        found = {
            path.relative_to(self.top).as_posix()
            for pattern in table.get("owners", ())
            for path in self.top.glob(pattern)
            if path.is_dir() and not path.is_symlink()
        }
        found -= vendored(self.top, found)
        return sorted(
            owner
            for owner in found
            if not excluded.match_file(f"{owner}/")
            and not self._related(self.top / owner)
            and not self._defines(self.top / owner, packages)
        )

    def _related(self, owner: Path) -> bool:
        """Whether `owner` is the project or sits inside it. An owner holding the project
        (`research/cuda-tokenization` holds `cutoken`) stays a consumer through the files beside
        the project: the experiments that import it."""
        return owner == self.root or self.root in owner.parents

    @staticmethod
    def _defines(owner: Path, packages: list[str]) -> bool:
        return any(
            (owner / prefix / package / "__init__.py").is_file()
            for package in packages
            for prefix in ("", "src")
        )


class Index:
    """The `(module, name)` pairs a project's consumers import, with a consumer file for each.

    submodules: every trailing part of the paths of the project's modules (`log`, `mainboard/log`
        ...), so `mainboard.log` reads as a module imported, not as a name used.
    """

    def __init__(self, submodules: frozenset[str] = frozenset()) -> None:
        self.names: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        self.modules: dict[str, str] = {}
        self.submodules = submodules

    def add_module(self, module: str, consumer: str) -> None:
        self.modules.setdefault(module.replace(".", "/"), consumer)

    def add_name(self, module: str, name: str, consumer: str) -> None:
        path = module.replace(".", "/")
        if f"{path}/{name}" not in self.submodules:
            self.names[name].append((path, consumer))

    def add_chain(self, chain: str, consumer: str) -> None:
        """Every module and name a dotted path reaches: `a.b.c` is `c` of `a.b` and `b` of `a`."""
        parts = chain.split(".")
        self.add_module(parts[0], consumer)
        for end in range(1, len(parts)):
            self.add_name(".".join(parts[:end]), parts[end], consumer)
            self.add_module(".".join(parts[: end + 1]), consumer)

    def user(self, path: str, name: str, *, exact: bool = False) -> str | None:
        """A consumer file importing `name` from the module `path`, or from a package holding it
        (re-exporting it, as far as the text tells) unless `exact`."""
        inside = f"/{module_key(path)}/"
        return next(
            (
                consumer
                for wanted in (name, "*")
                for module, consumer in self.names.get(wanted, ())
                if (inside.endswith(f"/{module}/") if exact else f"/{module}/" in inside)
            ),
            None,
        )

    def user_of_module(self, path: str) -> str | None:
        """A consumer file importing the module `path` itself, whole or by a name."""
        key = module_key(path)
        parent, _, stem = key.rpartition("/")
        for module, consumer in self.modules.items():
            if f"/{key}".endswith(f"/{module}"):
                return consumer
        return next(
            (
                consumer
                for module, consumer in self.names.get(stem, ())
                if f"/{parent}".endswith(f"/{module}")
            ),
            None,
        )


class Consumers:
    """The imports of a project's consumers, built on first use.

    root: the project; its consumers are found through its workspace.
    memory: the DuckDB limit for parsing the consumer files.
    """

    def __init__(self, root: Path, *, memory: str = "1GB") -> None:
        self.root, self.memory = root, memory
        self._lock = threading.Lock()
        self._index: Index | None = None

    @property
    def index(self) -> Index:
        """The consumers' imports, built by the first caller while the others wait."""
        with self._lock:
            if self._index is None:
                self._index = self._build()
            return self._index

    def _build(self) -> Index:
        """Search, parse and read the consumers."""
        started = time.perf_counter()
        workspace = Workspace(self.root)
        index = Index(workspace.submodules())
        packages = workspace.packages()
        owners = workspace.consumers(packages) if packages else []
        if not owners or workspace.top is None:
            return index
        own = f"{workspace.root.relative_to(workspace.top).as_posix()}/**"
        files = _importers(workspace.top, owners, packages, [*_excluded(workspace.top), own])
        searched = time.perf_counter()
        if files:
            self._read(index, workspace.top, files, set(packages))
        logger.debug(
            "{} consumer files in {} repositories import {}: search {:.2f} s, read {:.2f} s",
            len(files),
            len(owners),
            packages,
            searched - started,
            time.perf_counter() - searched,
        )
        return index

    def _cache(self) -> Path:
        """The syntax trees of this project's consumers, kept between runs by file content: a
        later run parses only the consumer files that changed."""
        digest = hashlib.blake2b(str(self.root).encode(), digest_size=8).hexdigest()
        return Path.home() / f".cache/mainboard/analysis/consumers-{digest}.duckdb"

    def _read(self, index: Index, top: Path, files: list[str], packages: set[str]) -> None:
        """Parse `files` (linked into a folder of their own, which the parser reads) and add what
        they import of `packages`."""
        budget = Budget(memory=self.memory)
        mirror = budget.folder / "consumers"
        try:
            for file in files:
                (mirror / file).parent.mkdir(parents=True, exist_ok=True)
                (mirror / file).symlink_to(top / file)
            facts = Facts(mirror, budget)
            Ducks(budget=budget, cache=self._cache(), exclusive=True).register(facts, mirror)
            for module, name, consumer in facts.db.execute(_PAIRS).fetchall():
                if module.split(".")[0] not in packages:
                    continue
                if not name:
                    index.add_chain(module, consumer)
                    continue
                index.add_module(module, consumer)
                index.add_name(module, name, consumer)
                if name != "*":
                    index.add_module(f"{module}.{name}", consumer)
            for chain, consumer in facts.db.execute(_CHAINS).fetchall():
                if chain.split(".")[0] in packages:
                    index.add_chain(chain, consumer)
            facts.close()
        finally:
            budget.close()

    def used(self, rule: str, path: str, declared: str) -> str | None:
        """A consumer file using what a finding of `rule` declares, if any."""
        if rule in MODULES:
            return self.index.user_of_module(path)
        return self.index.user(path, declared, exact=rule in EXPORTS)

    def broken(self, plan: Plan) -> list[str]:
        """`path: reason` for each file whose edits remove or rename a name a consumer imports,
        or that the plan removes or renames while a consumer imports its module."""
        edited = {path: data for path, data in plan.written().items() if path.endswith(".py")}
        gone = [*plan.removed, *plan.renamed]
        if not edited and not any(path.endswith(".py") for path in gone):
            return []
        return list(self._broken(plan, edited, gone))

    def _broken(self, plan: Plan, edited: dict[str, bytes], gone: list[str]) -> Iterator[str]:
        for path, data in edited.items():
            try:
                lost = surface(plan.source(path).data) - surface(data)
                before, after = positional(plan.source(path).data), positional(data)
            except SyntaxError:
                continue
            for name in sorted(lost):
                if consumer := self.index.user(path, name):
                    yield f"{path}: the fix removes `{name}`, which {consumer} imports"
            narrowed = {key[0] for key, count in before.items() if after.get(key, count) < count}
            for name in sorted(narrowed):
                if consumer := self.index.user(path, name):
                    yield (
                        f"{path}: the fix makes parameters of `{name}` keyword-only, "
                        f"and {consumer} imports it, so its callers may pass them by position"
                    )
        for path in gone:
            if path.endswith(".py") and (consumer := self.index.user_of_module(path)):
                yield f"{path}: the fix removes the module, which {consumer} imports"


def _excluded(top: Path) -> list[str]:
    """The workspace's `[lint].exclude` patterns, which no consumer file may sit under."""
    try:
        return list(tomllib.loads((top / "mainboard.toml").read_text())["lint"]["exclude"])
    except FileNotFoundError, KeyError:
        return []


def _importers(top: Path, owners: list[str], packages: list[str], exclude: list[str]) -> list[str]:
    """The Python files of `owners` that import one of `packages`, by text, relative to `top`."""
    alternatives = "|".join(packages)
    pattern = rf"^\s*(from\s+({alternatives})(\.|\s)|import\s+[\w., ]*\b({alternatives})\b)"
    found = subprocess.run(
        [
            str(shutil.which("rg")),
            "--files-with-matches",
            "--hidden",
            "--no-messages",
            "--type",
            "py",
            "--glob",
            "!.git",
            *(f"--glob=!{glob}" for glob in exclude),
            "--regexp",
            pattern,
            "--",
            *owners,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=top,
    )
    if found.returncode > 1:
        raise RuntimeError(f"ripgrep failed: {found.stderr.strip()}")
    return sorted(found.stdout.splitlines())
