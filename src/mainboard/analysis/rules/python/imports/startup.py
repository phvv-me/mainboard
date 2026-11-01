"""Rules over what an entry point imports at startup: what it can defer, and what it must not reach.

Python 3.15 defers a module-level import whose module its file lists in `__lazy_modules__` (PEP
810) until a name the import binds is first read; 3.14 ignores the list. Both rules read the
imports as 3.15 runs them, from each `[project.scripts]` entry of the project.
"""

import sys

from ....catalog import Example, Relation, Rule

# The project an example runs in: one package whose `cli` module, the entry, is the console script
# and whose `__init__` is its facade.
_ENTRY = "src/pkg/cli.py"
_FACADE = "src/pkg/__init__.py"
_PACKAGE = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n\n[project.scripts]\npkg = "pkg.cli:main"\n',
    _FACADE: "",
}

# A heavy import whose name is read when the module runs, which 3.15 loads however it is listed.
_READ_AT_IMPORT = "import numpy\n\nZERO = numpy.zeros(1)\n\n\ndef main() -> None:\n    pass\n"


class ImportsLazyEntries(Relation):
    """Each module a Python file lists in its module-level `__lazy_modules__`, with the `line` the
    list starts on: Python reads the list as the imports after it run."""

    query = """
        SELECT f.file_id, c.name AS entry, a.start_line AS line
        FROM core_node a
        JOIN core_file f USING (file_id)
        JOIN core_node t ON t.file_id = a.file_id AND t.parent_id = a.node_id
          AND t.sibling_index = 0 AND t.type = 'identifier' AND t.name = '__lazy_modules__'
        JOIN core_node e ON e.file_id = a.file_id AND e.node_id = a.parent_id
        JOIN core_node m ON m.file_id = e.file_id AND m.node_id = e.parent_id
          AND m.type = 'module'
        JOIN core_node c ON c.file_id = a.file_id AND c.type = 'string_content'
          AND c.node_id BETWEEN a.node_id AND a.node_id + a.descendant_count
        WHERE a.type = 'assignment' AND f.language = 'python'
    """


class ImportsModuleEdges(Relation):
    """Each module a module-level import statement runs (`target`), as Python 3.15 runs it.

    `entry` is the name `__lazy_modules__` defers the statement by (its `from` module, else the
    module it imports) and `lazy` whether the file lists it above the statement. A project target
    is every package on the way to the module and the module itself, and a `from` member that is a
    module too, short of the packages the importer sits in, which ran before it; an outside one is
    its top-level name. A statement under `if`, `try` or `with` is not module-level: Python runs it
    eagerly whatever the list says, so it is no edge here and no candidate either.
    """

    stored = True
    query = """
        WITH bindings AS (
          SELECT i.file_id, i.path, f.module AS importer, i.statement_id, i.start_line,
                 i.start_column, coalesce(i.source_module, i.module) AS entry, i.imported_name,
                 i.statement_type, i.is_project_owned
          FROM core_import i
          JOIN core_file f USING (file_id)
          JOIN core_node s ON s.file_id = i.file_id AND s.node_id = i.statement_id
          JOIN core_node m ON m.file_id = s.file_id AND m.node_id = s.parent_id
            AND m.type = 'module'
          WHERE i.statement_type <> 'future_import_statement'
            AND NOT starts_with(coalesce(i.source_module, i.module), '.')
        ), named AS (
          SELECT *, unnest(list_transform(range(1, len(string_split(entry, '.')) + 1),
                     lambda n: array_to_string(string_split(entry, '.')[1:n], '.'))) AS target
          FROM bindings WHERE is_project_owned
          UNION ALL
          SELECT *, entry || '.' || imported_name FROM bindings
          WHERE is_project_owned AND statement_type = 'import_from_statement'
          UNION ALL
          SELECT *, split_part(entry, '.', 1) FROM bindings WHERE NOT is_project_owned
        )
        SELECT DISTINCT n.file_id, n.path, n.importer, n.statement_id, n.start_line,
               n.start_column, n.entry, n.target, z.entry IS NOT NULL AS lazy
        FROM named n
        LEFT JOIN imports_lazy_entries() z ON z.file_id = n.file_id AND z.entry = n.entry
          AND z.line < n.start_line
        WHERE NOT n.is_project_owned
           OR (n.target IN (SELECT module FROM core_file WHERE language = 'python')
               AND NOT starts_with(n.importer || '.', n.target || '.'))
    """


class ImportsStartup(Relation):
    """Each module a `[project.scripts]` entry runs at startup, with the shortest `chain` of
    imports that runs it (the entry first): the entry module, the packages Python runs before it,
    and every module their eager imports reach, an outside library by its top-level name.

    Of the chains that tie, a step keeps the one of the smaller entry, then the smaller chain: a
    `USING KEY` step otherwise keeps whichever row its join emits last, and that differs per run.
    """

    stored = True
    query = r"""
        WITH RECURSIVE entries AS (
          SELECT DISTINCT regexp_extract(line, '=\s*["'']\s*([A-Za-z_][\w.]*)\s*:', 1) AS entry
          FROM (SELECT unnest(string_split(regexp_extract(
                         content, '(?s)\[project\.scripts\][^\n]*\n(.*?)(\n\s*\[|$)', 1),
                       chr(10))) AS line
                FROM source_files WHERE regexp_matches(path, '(^|/)pyproject\.toml$'))
          WHERE entry <> ''
        ), starts AS (
          SELECT entry, unnest(list_transform(range(1, len(string_split(entry, '.')) + 1),
                   lambda n: array_to_string(string_split(entry, '.')[1:n], '.'))) AS module
          FROM entries
        ), walk(module, entry, chain) USING KEY (module) AS (
          SELECT module, entry, CASE WHEN module = entry THEN [entry] ELSE [entry, module] END
          FROM starts WHERE module IN (SELECT module FROM core_file WHERE language = 'python')
          QUALIFY row_number() OVER (PARTITION BY module ORDER BY module <> entry, entry) = 1
          UNION
          (SELECT e.target, w.entry, list_append(w.chain, e.target)
           FROM walk w
           JOIN imports_module_edges() e ON e.importer = w.module AND NOT e.lazy
           LEFT JOIN recurring.walk r ON r.module = e.target
           WHERE r.module IS NULL OR len(r.chain) > len(w.chain) + 1
           QUALIFY row_number() OVER (PARTITION BY e.target
                                      ORDER BY len(w.chain), w.entry, w.chain) = 1)
        )
        SELECT module, entry, chain FROM walk
    """


class ImportsRegistryHeld(Relation):
    """The imports a registry family needs eager, by file and `entry`.

    A family is a class deriving from `registry_bases` or defining an `__init_subclass__` hook,
    and its implementations are the classes below it. A lookup through the family's root has run
    the root's module and the packages above it, so its home is what their eager imports reach.
    An import a home module makes is held when its target runs an implementation of that family,
    itself or through its own eager imports: deferred, the family could be asked before the
    implementation is registered. Both walks stop short of the top-level package, which Python
    runs before any module of the project: it is the library facade, and no family meets its
    implementations there (all 18 of mainboard's families keep theirs inside the root's own
    subpackage and register the same classes with the facades lazy or eager, 2026-10-29).
    """

    registry_bases: list[str]

    query = """
        WITH RECURSIVE roots AS (
          SELECT DISTINCT k.class_id, k.module FROM class_keys k
          WHERE EXISTS (SELECT 1 FROM class_bases b WHERE b.class_id = k.class_id
                          AND list_contains(registry_bases, b.base_name))
             OR EXISTS (SELECT 1 FROM class_methods m WHERE m.class_id = k.class_id
                          AND m.name IN ('__init_subclass__', '__pydantic_init_subclass__'))
        ), implementations AS (
          SELECT DISTINCT r.module AS root, k.module
          FROM class_lineage l
          JOIN roots r ON r.class_id = l.ancestor
          JOIN class_keys k ON k.class_id = l.class_id
        ), edges AS (
          SELECT DISTINCT importer, target FROM imports_module_edges()
          WHERE NOT lazy AND contains(target, '.')
        ), home(root, module) AS (
          SELECT root, unnest(list_transform(range(2, len(string_split(root, '.')) + 1),
                   lambda n: array_to_string(string_split(root, '.')[1:n], '.')))
          FROM (SELECT DISTINCT root FROM implementations)
          UNION
          SELECT h.root, e.target FROM home h JOIN edges e ON e.importer = h.module
        ), registering(root, module) AS (
          SELECT root, module FROM implementations
          UNION
          SELECT g.root, e.importer FROM registering g JOIN edges e ON e.target = g.module
        )
        SELECT DISTINCT e.file_id, e.entry
        FROM imports_module_edges() e
        JOIN home h ON h.module = e.importer
        JOIN registering g ON g.root = h.root AND g.module = e.target
        WHERE NOT e.lazy
    """


class ImportsDeferrable(Relation):
    """Each module-level import of a startup module that `__lazy_modules__` can defer, by file and
    `entry`: every name it binds is read only later (a function body, annotations, `__all__`),
    none at import, it is no registry family's home import, and it binds something (no side
    effect). Only imports of outside libraries and project modules count, never of `stdlib`.

    A module listed for an import that reads a name at import is lazy only to the facts: Python
    3.15 loads it there all the same, so a repair lists only what this relation holds.
    """

    registry_bases: list[str]
    stdlib: list[str]

    query = """
        WITH defaults AS (
          SELECT v.file_id, d.def_id, v.node_id AS first, v.node_id + v.descendant_count AS last
          FROM core_def d
          JOIN core_node l ON l.file_id = d.file_id AND l.parent_id = d.def_id
            AND l.type IN ('parameters', 'lambda_parameters')
          JOIN core_node p ON p.file_id = l.file_id AND p.parent_id = l.node_id
            AND p.type IN ('default_parameter', 'typed_default_parameter')
          JOIN core_node v ON v.file_id = p.file_id AND v.parent_id = p.node_id
            AND v.sibling_index = p.children_count - 1
          WHERE d.kind = 'function' AND d.function_id IS NULL
        ), statements AS (
          SELECT DISTINCT e.file_id, e.path, e.importer, e.statement_id, e.start_line,
                 e.start_column, e.entry
          FROM imports_module_edges() e
          WHERE NOT e.lazy AND NOT list_contains(stdlib, split_part(e.entry, '.', 1))
            AND e.importer IN (SELECT module FROM imports_startup())
            AND NOT EXISTS (SELECT 1 FROM imports_registry_held(registry_bases) h
                            WHERE h.file_id = e.file_id AND h.entry = e.entry)
        ), reads AS (
          SELECT s.file_id, s.statement_id, i.name, i.imported_name, b.is_reexported,
                 count(r.ref_id) AS reads,
                 count(r.ref_id) FILTER (WHERE r.function_id IS NULL OR d.def_id IS NOT NULL)
                   AS immediate
          FROM statements s
          JOIN core_import i ON i.file_id = s.file_id AND i.statement_id = s.statement_id
          LEFT JOIN import_binding_facts b ON b.path = s.path AND b.name = i.name
          LEFT JOIN core_ref r ON r.file_id = i.file_id AND r.name = i.name AND r.is_load
          LEFT JOIN defaults d ON d.file_id = r.file_id AND d.def_id = r.function_id
            AND r.ref_id BETWEEN d.first AND d.last
          GROUP BY ALL
        ), used AS (
          SELECT file_id, statement_id, sum(immediate) AS immediate,
                 bool_or(reads > 0 OR coalesce(is_reexported, false)) AS used,
                 bool_or(imported_name = '*') AS wildcard
          FROM reads GROUP BY ALL
        ), deferred AS (
          SELECT s.file_id, s.path, s.importer, s.entry, min(s.start_line) AS start_line,
                 arg_min(s.start_column, s.start_line) AS start_column
          FROM statements s JOIN used u USING (file_id, statement_id)
          GROUP BY ALL
          HAVING sum(u.immediate) = 0 AND bool_and(u.used) AND NOT bool_or(u.wildcard)
        )
        SELECT file_id, path, importer, entry, start_line, start_column FROM deferred
    """


class StartupImports(Rule):
    """What the rules over startup imports share: the registry families whose home imports stay
    eager (a family's root is a class deriving from one of `registry_bases`) and the standard
    library, whose imports are never judged."""

    registry_bases: tuple[str, ...] = ("Registry", "Component")

    constants = {"stdlib": sorted(sys.stdlib_module_names)}


class DeferrableImport(StartupImports, id="PY-IMPO0009", applicability="review"):
    """Imports a startup module makes eagerly though it reads their names only after import.

    Every name such an import binds is read where the code runs later, in a function body or a
    function's annotations, or is exported through `__all__`, and nothing reads one at import
    (module level, a class body, a base, a decorator, a default). Only the modules an entry point
    runs at startup are judged, outside libraries and project modules alike, never the standard
    library. An import a registry family's home needs stays eager (`imports_registry_held()`),
    and so does one whose statement binds nothing that is read or exported: it runs for its side
    effect. `mb --help` took 0.57 s on Python 3.15 with every import eager and 0.17 s with the
    startup modules' lists (2026-10-29).
    """

    query = """
        SELECT d.path, min(d.start_line) AS start_line,
               arg_min(d.start_column, d.start_line) AS start_column, d.importer, c.content,
               list(d.entry ORDER BY d.entry) AS entries,
               string_agg(d.entry, ', ' ORDER BY d.entry) AS modules
        FROM imports_deferrable($registry_bases, $stdlib) d
        JOIN source_files c ON c.path = d.path
        GROUP BY d.path, d.importer, c.content
    """
    fix = """
        SELECT finding_key, 'lazy-modules' AS operation, path, content, importer, entries
        FROM finding
    """
    message = "`{importer}` reads `{modules}` only inside functions; __lazy_modules__ defers them"
    examples = (
        Example(
            bad="""from .board import Board


def main() -> None:
    Board().run()
""",
            good="""__lazy_modules__ = [
    "pkg.board",
]

from .board import Board


def main() -> None:
    Board().run()
""",
            path=_ENTRY,
            context={
                **_PACKAGE,
                "src/pkg/board.py": "class Board:\n    def run(self) -> None:\n        pass\n",
            },
        ),
        Example(
            bad="""# The library facade.

from .board import Board

__all__ = ["Board"]
""",
            good="""# The library facade.

__lazy_modules__ = [
    "pkg.board",
]

from .board import Board

__all__ = ["Board"]
""",
            path=_FACADE,
            context={
                **_PACKAGE,
                _ENTRY: "def main() -> None:\n    pass\n",
                "src/pkg/board.py": "class Board:\n    pass\n",
            },
        ),
        Example(
            bad='''"""The entry."""

import json

from .board import Board

__lazy_modules__ = ["pkg.gone", "pkg.board"]

from .light import shine


def main() -> None:
    json.dumps(shine(Board()))
''',
            good='''"""The entry."""

__lazy_modules__ = [
    "pkg.board",
    "pkg.light",
]

import json

from .board import Board
from .light import shine


def main() -> None:
    json.dumps(shine(Board()))
''',
            path=_ENTRY,
            context={
                **_PACKAGE,
                "src/pkg/board.py": "class Board:\n    pass\n",
                "src/pkg/light.py": "def shine(board):\n    return board\n",
            },
        ),
    )


class HeavyStartupImport(StartupImports, id="PY-IMPO0010", applicability="review"):
    """Heavy libraries an entry point imports at startup through eager imports alone.

    The message names the shortest chain from the entry to a module of `heavy`, one finding per
    library. The cure is PY-IMPO0009 at one of its links, or a command whose module loads only
    when it is typed, so the finding stands on the first import of the chain that can wait
    (`imports_deferrable()`), else on the one that reaches the library. The fix lists that import's
    module in its file's `__lazy_modules__`, which spares whatever else the import reaches. A
    chain none of whose links can wait keeps its finding: 3.15 loads an import whose name is read
    at import however it is listed. A library other chains reach keeps a finding on the next
    shortest one, which the next run cuts. `mb --help` loaded sqlalchemy, duckdb,
    duckdb_sqlalchemy, gpuhunt and requests it never used and took 0.57 s on Python 3.15, and
    0.17 s once no chain reached them (2026-10-29).
    """

    heavy: tuple[str, ...] = (
        *("duckdb", "duckdb_sqlalchemy", "fastmcp", "gpuhunt", "httpx", "matplotlib"),
        *("numpy", "pyarrow", "requests", "sqlalchemy"),
    )

    query = """
        WITH links AS (
          SELECT entry, module AS library, chain, unnest(range(1, len(chain))) AS k
          FROM imports_startup() WHERE list_contains($heavy, module)
        )
        SELECT e.path, e.start_line, e.start_column, l.entry, l.library,
               array_to_string(l.chain, ' -> ') AS chain, e.importer, e.entry AS imported,
               d.entry IS NOT NULL AS deferrable
        FROM links l
        JOIN imports_module_edges() e
          ON e.importer = l.chain[l.k] AND e.target = l.chain[l.k + 1] AND NOT e.lazy
        LEFT JOIN imports_deferrable($registry_bases, $stdlib) d
          ON d.file_id = e.file_id AND d.entry = e.entry
        QUALIFY row_number() OVER (
          PARTITION BY l.library
          ORDER BY d.entry IS NULL, CASE WHEN d.entry IS NULL THEN -l.k ELSE l.k END,
                   e.path, e.start_line) = 1
    """
    fix = """
        SELECT f.finding_key, 'lazy-modules' AS operation, f.path, s.content, f.importer,
               [f.imported] AS entries
        FROM finding f JOIN source_files s ON s.path = f.path
        WHERE f.deferrable
    """
    message = "`{entry}` imports `{library}` at startup: {chain}"
    examples = (
        Example(
            bad="""import numpy


def main() -> None:
    numpy.zeros(3)
""",
            good="""__lazy_modules__ = [
    "numpy",
]

import numpy


def main() -> None:
    numpy.zeros(3)
""",
            path=_ENTRY,
            context=_PACKAGE,
        ),
        Example(
            bad="""from .tables import frame


def main() -> None:
    frame()
""",
            good="""__lazy_modules__ = [
    "pkg.tables",
]

from .tables import frame


def main() -> None:
    frame()
""",
            path=_ENTRY,
            context={
                **_PACKAGE,
                "src/pkg/tables.py": "import numpy\n\n\ndef frame() -> None:\n    numpy.zeros(1)\n",
            },
        ),
        Example(
            bad="import numpy\n\n\ndef frame() -> None:\n    numpy.zeros(1)\n",
            good='__lazy_modules__ = [\n    "numpy",\n]\n\nimport numpy\n\n\ndef frame() -> None:\n    numpy.zeros(1)\n',
            path="src/pkg/tables.py",
            context={
                **_PACKAGE,
                _ENTRY: "from .tables import frame\n\nFRAME = frame()\n\n\ndef main() -> None:\n    pass\n",
            },
        ),
        Example(
            bad=_READ_AT_IMPORT,
            good=_READ_AT_IMPORT,
            path=_ENTRY,
            context=_PACKAGE,
        ),
    )
