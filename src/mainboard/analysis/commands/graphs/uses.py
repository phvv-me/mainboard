"""`uses`: where a function, class or constant is defined and who imports or calls it.

A symbol is found by the end of its dotted name (`Board`, `board.Board`,
`mainboard.fleet.board.Board`, `Board.verdicts`), so a name several modules define lists each
definition. The uses follow the imports: a plain recursive reach over the files that bind the name
(`from m import name as alias`, also through a package that re-exports it) finds each file's local
name, then the syntax trees give every place that name is read or called, and `module.name` for a
file that binds the module (`import pkg.m as m`). Names are matched, not types: a method is found
by its name in the files that import its class, and a local variable shadowing an import is a
false positive.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from patos import FrozenModel

from ....log import logger
from ..report import Report
from ..workspace import Workspace

_SHOWN = 40
_KINDS = ("import", "call", "reference")

_DEFINITIONS = """
    SELECT path, module, name, qualname, kind, line, top_level
    FROM module_definitions()
    WHERE module || '.' || qualname = $symbol
       OR ends_with(module || '.' || qualname, '.' || $symbol)
       OR qualname = $symbol
    ORDER BY module, qualname
"""

# The names a file binds for the exports of one definition: its own name in the defining file and,
# in every file that imports it (`from m import name as alias`) or re-exports it, the local name.
_EXPORTS = """
    exports(path, name, depth) AS (
      SELECT $path, $name, 0
      UNION
      SELECT b.path, b.local, x.depth + 1
      FROM exports x
      JOIN module_bindings() b ON b.imported = x.path AND b.member = x.name
      WHERE x.depth < 16
    )
"""

# A name is read or called where an identifier or a dotted attribute spells it, outside the places
# an identifier is a binder: imports, definition names, parameters, keyword names, assignment
# targets and the member names of an attribute. sitting_duck tags the value of an unpacking
# assignment (`low, high = BAND`) `assignment_target` too, so only a first child is a target.
_NOT_A_BINDER = """
    NOT (n.type = 'identifier' AND (
         list_has_any(n.modifiers, ['in_dotted_name', 'in_aliased_import', 'in_parameters',
                                    'in_lambda_parameters', 'in_pattern_list'])
         OR (list_contains(n.modifiers, 'assignment_target') AND n.sibling_index = 0)
         OR coalesce(n.signature_type, '') IN ('function_name', 'class_name')
         OR (n.sibling_index = 0 AND list_has_any(n.modifiers,
               ['in_keyword_argument', 'in_typed_parameter', 'in_default_parameter',
                'in_typed_default_parameter']))
         OR (list_contains(n.modifiers, 'in_attribute') AND n.sibling_index > 0)))
"""

_FUNCTION_USES = f"""
    WITH RECURSIVE {_EXPORTS}, spelled AS (
      SELECT path, name AS text FROM exports
      UNION
      SELECT b.path, b.local || '.' || x.name
      FROM module_bindings() b JOIN exports x ON b.member IS NULL AND b.imported = x.path
    ), found AS (
      SELECT n.path, n.start_line AS line,
             CASE WHEN list_has_any(n.modifiers, ['function_call', 'method_call'])
                  THEN 'call' ELSE 'reference' END AS kind
      FROM python.ast_nodes n JOIN spelled s ON s.path = n.path AND s.text = n.name
      WHERE n.type IN ('identifier', 'attribute') AND {_NOT_A_BINDER}
      UNION ALL
      SELECT b.path, b.line, 'import'
      FROM module_bindings() b JOIN exports x ON b.imported = x.path AND b.member = x.name
    )
"""

_METHOD_USES = f"""
    WITH RECURSIVE {_EXPORTS}, found AS (
      SELECT n.path, n.start_line AS line,
             CASE WHEN list_contains(n.modifiers, 'method_call') THEN 'call' ELSE 'reference' END
               AS kind
      FROM python.ast_nodes n
      WHERE n.type = 'attribute' AND ends_with(n.name, '.' || $method)
        AND n.path IN (SELECT path FROM exports)
    )
"""

_SITUATED = """
    SELECT f.path, f.line, f.kind, e.name AS experiment, t.path IS NOT NULL AS trial
    FROM found f
    LEFT JOIN experiment_nodes() e ON starts_with(f.path, e.dir || '/')
    LEFT JOIN trial_files() t ON t.path = f.path
    ORDER BY f.path, f.line
"""


class Use(FrozenModel):
    """One place a symbol is imported, called or read."""

    path: str
    line: int
    kind: str
    experiment: str | None
    trial: bool


class FileUses(FrozenModel):
    """The uses of a symbol in one file, the lines of each kind."""

    path: str
    experiment: str | None
    trial: bool
    lines: dict[str, list[int]]


class Definition(FrozenModel):
    """One definition a symbol names, with the files that use it."""

    path: str
    module: str
    qualname: str
    kind: str
    line: int
    experiments: list[str]
    trials: int
    files: list[FileUses]


class UsesReport(Report):
    """Every definition `symbol` names and who uses each."""

    symbol: str
    definitions: list[Definition]

    @staticmethod
    def _file(file: FileUses) -> str:
        tags = [tag for tag in (file.experiment, "trial" * file.trial) if tag]
        kinds = "; ".join(
            f"{kind} {','.join(map(str, numbers[:8]))}{'..' if len(numbers) > 8 else ''}"
            for kind, numbers in file.lines.items()
        )
        return f"  {file.path}{f'  [{", ".join(tags)}]' if tags else ''}  {kinds}"

    def text(self) -> str:
        if not self.definitions:
            return f"no definition of {self.symbol}"
        lines = []
        for definition in self.definitions:
            lines.append(
                f"{definition.module}.{definition.qualname}  {definition.kind}  "
                f"{definition.path}:{definition.line}"
            )
            lines.append(
                f"  used in {len(definition.files)} files, {len(definition.experiments)} "
                f"experiments, {definition.trials} trials"
            )
            lines += [self._file(file) for file in definition.files[:_SHOWN]]
            more = len(definition.files) - _SHOWN
            lines += [f"  +{more} more files (--json lists all)"] * (more > 0)
        return "\n".join(lines)


class Symbol:
    """A symbol's definitions in one project and the uses of each."""

    @staticmethod
    def _files(uses: Sequence[Use]) -> list[FileUses]:
        """The uses grouped by file, their lines by kind."""
        grouped: dict[str, list[Use]] = defaultdict(list)
        for use in uses:
            grouped[use.path].append(use)
        return [
            FileUses(
                path=path,
                experiment=found[0].experiment,
                trial=found[0].trial,
                lines={
                    kind: sorted({use.line for use in found if use.kind == kind})
                    for kind in _KINDS
                    if any(use.kind == kind for use in found)
                },
            )
            for path, found in grouped.items()
        ]

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def _definition(self, definition: Mapping) -> Definition:
        if definition["top_level"]:
            sql = _FUNCTION_USES + _SITUATED
            parameters = {"path": definition["path"], "name": definition["name"]}
        else:
            sql = _METHOD_USES + _SITUATED
            owner = definition["qualname"].split(".")[0]
            parameters = {
                "path": definition["path"],
                "name": owner,
                "method": definition["name"],
            }
        files = self._files([Use(**row) for row in self.workspace.rows(sql, parameters)])
        return Definition(
            path=definition["path"],
            module=definition["module"],
            qualname=definition["qualname"],
            kind=definition["kind"],
            line=definition["line"],
            experiments=sorted({file.experiment for file in files if file.experiment}),
            trials=sum(file.trial for file in files),
            files=files,
        )

    def report(self, symbol: str) -> UsesReport:
        """Each definition `symbol` names, with its uses."""
        definitions = self.workspace.rows(_DEFINITIONS, {"symbol": symbol})
        return UsesReport(symbol=symbol, definitions=[self._definition(d) for d in definitions])


def uses(
    symbol: str,
    *,
    root: Path = Path(),
    as_json: Annotated[bool, Parameter(name="--json")] = False,
    memory: str = "2GB",
) -> None:
    """Where a function, class, method or constant is defined and who imports or calls it.

    symbol: its name or the tail of its dotted name (`Board`, `board.Board`, `Board.verdicts`).
    root: the project to analyze.
    as_json: print JSON for a program.
    memory: a DuckDB memory limit such as `4GB`.
    """
    logger.result(Symbol(Workspace(root, memory=memory)).report(symbol).render(as_json=as_json))
