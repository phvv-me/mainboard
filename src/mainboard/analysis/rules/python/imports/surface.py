"""Rules over a package's public surface (mcmr's PY-IMPO hygiene family, continued)."""

import ast
from collections.abc import Iterable, Sequence
from itertools import chain, takewhile

from .... import edits
from ....catalog import Example, Relation, Rule
from ....edits import Edit
from ....findings import Hit, Span


class ImportsFacades(Relation):
    """The `imports_facades()` relation."""

    query = """
    WITH routed AS (
      SELECT * FROM export_bypasses
      WHERE is_cycle_safe AND module_node_id IS NOT NULL AND replacement_module IS NOT NULL
      QUALIFY count(*) OVER (PARTITION BY module_node_id, replacement_module)
            = min(binding_count) OVER (PARTITION BY module_node_id, replacement_module)
        AND min(binding_count) OVER (PARTITION BY module_node_id, replacement_module)
            = max(binding_count) OVER (PARTITION BY module_node_id, replacement_module)
    )
    SELECT * FROM routed QUALIFY ordinal = min(ordinal) OVER (PARTITION BY module_node_id)
"""


class BypassedPublicImport(Rule, id="PY-IMPO0006", applicability="safe"):
    """Imports that bypass the shortest explicit project package surface."""

    query = """
        SELECT b.*, b.span_path AS path, b.span_start_line AS start_line,
          b.span_start_column AS start_column, b.span_end_line AS end_line,
          b.span_end_column AS end_column, r.record_id IS NOT NULL AS routed
        FROM export_bypasses b
        LEFT JOIN (SELECT record_id FROM imports_facades()) r ON r.record_id = b.record_id
    """
    message = "`{expression}` bypasses the shorter public import `{public_module}.{name}`"
    examples = (
        Example(
            bad="from pkg.core.errors import MissionError\n",
            good="from pkg.core import MissionError\n",
            path="app.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/core/__init__.py": (
                    'from .errors import MissionError\n\n__all__ = ["MissionError"]\n'
                ),
                "src/pkg/core/errors.py": "class MissionError(Exception):\n    pass\n",
            },
        ),
    )
    # Once per statement, from its first bypass, when every name it imports takes the same route.
    fix = """
        SELECT finding_key, module_node_span_path AS path,
               module_node_span_start_line AS start_line,
               module_node_span_start_column AS start_column,
               module_node_span_end_line AS end_line, module_node_span_end_column AS end_column,
               CASE WHEN routed THEN replacement_module END AS text
        FROM finding
    """


class UnexportedPackageSurface(Rule, id="PY-IMPO0008", applicability="safe"):
    """Public names that `minimum_importers` modules outside a package import from one of its
    submodules, which the package's initializer does not export. Exported there, every importer
    takes the package's shorter path (PY-IMPO0006), and imports of one package merge into one
    statement. A package without an initializer is left alone."""

    minimum_importers: int = 2

    query = r"""
        WITH defined AS (
          SELECT f.module, d.name FROM core_def d JOIN core_file f USING (file_id)
          WHERE d.owner_id IS NULL AND f.language = 'python' AND NOT f.is_package
        ), wanted AS (
          SELECT regexp_replace(i.source_module, '\.[^.]+$', '') AS package,
                 regexp_extract(i.source_module, '[^.]+$') AS submodule, i.imported_name AS name,
                 count(DISTINCT i.file_id) AS importers
          FROM core_import i
          JOIN core_file u USING (file_id)
          JOIN defined d ON d.module = i.source_module AND d.name = i.imported_name
          WHERE i.is_project_owned AND i.statement_type = 'import_from_statement'
            AND NOT starts_with(i.imported_name, '_')
            AND NOT starts_with(u.module || '.',
                                regexp_replace(i.source_module, '\.[^.]+$', '') || '.')
          GROUP BY ALL
          HAVING importers >= $minimum_importers
        ), bound AS (
          SELECT f.module AS package, i.name FROM core_import i JOIN core_file f USING (file_id)
          WHERE f.is_package
          UNION
          SELECT f.module, d.name FROM core_def d JOIN core_file f USING (file_id)
          WHERE f.is_package AND d.owner_id IS NULL
        )
        SELECT p.path, 1 AS start_line, 0 AS start_column, w.package, s.content,
               list({'submodule': w.submodule, 'name': w.name} ORDER BY w.submodule, w.name)
                 AS exports,
               string_agg(w.name, ', ' ORDER BY w.name) AS names
        FROM wanted w
        JOIN core_file p ON p.module = w.package AND p.is_package
        JOIN source_files s ON s.path = p.path
        WHERE NOT EXISTS (SELECT 1 FROM bound b WHERE b.package = w.package AND b.name = w.name)
        GROUP BY p.path, w.package, s.content
    """
    message = "`{package}` does not export `{names}`, which modules outside it import"
    examples = (
        Example(
            bad='"""The core."""\n\n__all__ = []\n',
            good=(
                '''"""The core."""

from .errors import MissionError

__all__ = [
    "MissionError",
]
'''
            ),
            path="src/pkg/core/__init__.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/core/errors.py": "class MissionError(Exception):\n    pass\n",
                "src/pkg/app.py": "from .core.errors import MissionError\n",
                "src/pkg/cli.py": "from .core.errors import MissionError\n",
            },
        ),
        Example(
            bad='from .errors import Refused\n\n__all__ = ["Refused"]\n',
            good=(
                """from .errors import Refused, MissionError

__all__ = [
    "MissionError",
    "Refused",
]
"""
            ),
            path="src/pkg/core/__init__.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/core/errors.py": (
                    """class MissionError(Exception):
    pass


class Refused(Exception):
    pass
"""
                ),
                "src/pkg/app.py": "from .core.errors import MissionError\n",
                "src/pkg/cli.py": "from .core.errors import MissionError\n",
            },
        ),
        Example(
            bad="# The core of the package.\n",
            good=(
                """# The core of the package.

from .errors import MissionError as MissionError
"""
            ),
            path="src/pkg/core/__init__.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/core/errors.py": "class MissionError(Exception):\n    pass\n",
                "src/pkg/app.py": "from .core.errors import MissionError\n",
                "src/pkg/cli.py": "from .core.errors import MissionError\n",
            },
        ),
    )

    @staticmethod
    def _appended(path: str, tree: ast.Module, *, source: str, fresh: str) -> Edit:
        """The new imports `fresh`, after the initializer's imports, docstring or comment."""
        imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        lines = source.splitlines()
        if imports:
            line = (imports[-1].end_lineno or imports[-1].lineno) + 1
        elif tree.body and isinstance(tree.body[0], ast.Expr):
            line, fresh = (tree.body[0].end_lineno or 1) + 1, "\n" + fresh
        else:
            comments = len(list(takewhile(lambda text: text.startswith("#"), lines)))
            trailing = "\n" if comments < len(lines) else ""
            line, fresh = comments + 1, ("\n" + fresh + trailing if comments else fresh)
        return edits.replace(Span(path=path, start_line=line, end_line=line), fresh)

    @staticmethod
    def _exported(hit: Hit) -> dict[str, list[str]]:
        """The names the finding exports, by the submodule defining them."""
        exported: dict[str, list[str]] = {}
        for entry in hit.records("exports"):
            exported.setdefault(str(entry["submodule"]), []).append(str(entry["name"]))
        return exported

    @staticmethod
    def _joined(path: str, node: ast.ImportFrom, names: Sequence[str], *, listed: bool) -> Edit:
        """`names` joined to the initializer's import `node` of their submodule."""
        kept = [
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in node.names
        ]
        spelled = [*kept, *UnexportedPackageSurface._spelled(names, listed=listed)]
        return edits.replace(
            UnexportedPackageSurface._span(path, node),
            f"from .{node.module} import {', '.join(spelled)}",
        )

    @staticmethod
    def _listing(tree: ast.Module) -> ast.List | ast.Tuple | None:
        """The literal `__all__` the initializer assigns, None without one."""
        for node in tree.body:
            match node:
                case ast.Assign(
                    targets=[ast.Name(id="__all__")], value=ast.List() | ast.Tuple() as value
                ):
                    return value
        return None

    @staticmethod
    def _relisted(path: str, listing: ast.List | ast.Tuple, names: Iterable[str]) -> Edit:
        """The literal `__all__` rewritten to hold `names` beside what it listed, one per line."""
        listed = {
            item.value
            for item in listing.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        body = "".join(f'    "{name}",\n' for name in sorted(listed | set(names)))
        return edits.replace(UnexportedPackageSurface._span(path, listing), f"[\n{body}]")

    @staticmethod
    def _span(path: str, node: ast.stmt | ast.expr) -> Span:
        """Where `node` stands in the file at `path`."""
        return Span(
            path=path,
            start_line=node.lineno,
            start_column=node.col_offset,
            end_line=node.end_lineno,
            end_column=node.end_col_offset,
        )

    @staticmethod
    def _spelled(names: Iterable[str], *, listed: bool) -> list[str]:
        """`names` spelled bare beside an `__all__`, else each aliased to itself."""
        return [name if listed else f"{name} as {name}" for name in names]

    def repair(self, hit: Hit) -> list[Edit] | None:
        """The submodules' names imported into the initializer and exported from it.

        Each joins the initializer's import of its submodule, else one new import (see
        `_appended`), and is listed in its `__all__`, or aliased to itself without one.
        """
        path, source = str(hit.path), str(hit.content)
        tree = ast.parse(source)
        exported = UnexportedPackageSurface._exported(hit)
        listing = UnexportedPackageSurface._listing(tree)
        held = {
            node.module: node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in exported
        }
        listed = listing is not None
        made = [
            self._joined(path, held[module], names, listed=listed)
            for module, names in sorted(exported.items())
            if module in held
        ]
        if fresh := "".join(
            f"from .{module} import {', '.join(self._spelled(names, listed=listed))}\n"
            for module, names in sorted(exported.items())
            if module not in held
        ):
            made.append(self._appended(path, tree, source=source, fresh=fresh))
        if listing is not None:
            made.append(self._relisted(path, listing, chain.from_iterable(exported.values())))
        return made
