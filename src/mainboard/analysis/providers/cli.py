"""The command tree a project declares, read from its source as data.

A project states its CLI once as `TREE = Node(...)`: nested `Node` and `Command` calls whose
`name`, `show`, `default`, `commands` and `children` keywords are literals, tuples, sums of tuples
and comprehensions over those. Reading the source instead of importing it keeps the rules on the
project's own files (a copy, a worktree) and runs none of its code, verb modules least of all.
"""

import ast
from collections.abc import Iterator, Sequence
from enum import StrEnum, auto
from functools import singledispatchmethod
from pathlib import Path

from patos import FrozenModel

from ...log import logger
from ..catalog import Family
from ..engine.facts import Facts, listed

_DECLARATION = "TREE = Node("
_ENTRIES_TABLE_SQL = (
    "CREATE TABLE cli_entries (path VARCHAR, command VARCHAR, parent VARCHAR, kind VARCHAR, "
    'shown BOOLEAN, depth BIGINT, line BIGINT, "column" BIGINT)'
)
_INSERT_ENTRY_SQL = "INSERT INTO cli_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
_READ = frozenset({"children", "commands", "default", "name", "show"})

type Value = str | int | bool | Declared | tuple[Value, ...]


class Kind(StrEnum):
    COMMAND = auto()
    GROUP = auto()


_KINDS = {"Command": Kind.COMMAND, "Node": Kind.GROUP}


class Declared(FrozenModel):
    """One `Node` or `Command` call and what it states about itself.

    default: a command run when its node is typed alone; it is no entry of the tree.
    """

    children: tuple[Declared, ...] = ()
    column: int
    commands: tuple[Declared, ...] = ()
    default: bool = False
    kind: Kind
    line: int
    name: str
    show: bool = True


class Entry(FrozenModel):
    """A node or a command of the tree, as the CLI facts hold it.

    command: its words from the root, `mb host list`.
    depth: the words typed after the root's.
    parent: the words of the node holding it, empty for the root.
    """

    column: int
    command: str
    depth: int
    kind: Kind
    line: int
    parent: str
    shown: bool


class Reader(FrozenModel):
    """What the expressions of the module `path` state, `scope` binding a comprehension's name."""

    path: str
    scope: dict[str, Value] = {}

    @singledispatchmethod
    def value(self, node: ast.expr) -> Value:
        """The value that `node` states, a ValueError for one that is no data."""
        raise self._refused(node)

    @value.register
    def _added(self, node: ast.BinOp) -> Value:
        if not isinstance(node.op, ast.Add):
            raise self._refused(node)
        return self._sequence(node.left) + self._sequence(node.right)

    @value.register
    def _called(self, node: ast.Call) -> Value:
        match node.func, node.args:
            case ast.Name(id=name), _ if name in _KINDS:
                return self.declared(node, _KINDS[name])
            case ast.Name(id="tuple" | "list"), [inner]:
                return self._sequence(inner)
            case _:
                raise self._refused(node)

    @value.register
    def _constant(self, node: ast.Constant) -> Value:
        if not isinstance(node.value, str | int | bool):
            raise self._refused(node)
        return node.value

    @value.register
    def _generated(self, node: ast.GeneratorExp) -> Value:
        match node.generators:
            case [ast.comprehension(target=ast.Name(id=name), iter=source, ifs=[])]:
                return tuple(
                    Reader(path=self.path, scope={**self.scope, name: each}).value(node.elt)
                    for each in self._sequence(source)
                )
            case _:
                raise self._refused(node)

    @value.register
    def _listed(self, node: ast.Tuple | ast.List) -> Value:
        return self._items(node.elts)

    @value.register
    def _named(self, node: ast.Name) -> Value:
        try:
            return self.scope[node.id]
        except KeyError:
            raise self._refused(node) from None

    def _items(self, items: Sequence[ast.expr]) -> tuple[Value, ...]:
        """The values of a tuple's or a list's elements, a starred one spread."""
        return tuple(
            value
            for item in items
            for value in (
                self._sequence(item.value)
                if isinstance(item, ast.Starred)
                else (self.value(item),)
            )
        )

    def _refused(self, node: ast.expr) -> ValueError:
        """The error for an expression that is no data."""
        return ValueError(f"{self.path}:{node.lineno}: not data: {ast.unparse(node)}")

    def _sequence(self, node: ast.expr) -> tuple[Value, ...]:
        """The value of `node`, which must be a tuple."""
        if not isinstance(value := self.value(node), tuple):
            raise self._refused(node)
        return value

    def declared(self, call: ast.Call, kind: Kind) -> Declared:
        """The `Node` or `Command` that `call` makes."""
        fields = {
            keyword.arg: self.value(keyword.value)
            for keyword in call.keywords
            if keyword.arg in _READ
        }
        return Declared.model_validate(
            fields | {"kind": kind, "line": call.lineno, "column": call.col_offset}
        )


class CommandTree(FrozenModel):
    """The `TREE` that the module `path` declares, in `source`."""

    path: str
    source: str

    @classmethod
    def declared_in(cls, root: Path) -> list[CommandTree]:
        """The modules of the project at `root` that declare a tree."""
        trees = []
        for name in listed(root):
            if name.endswith(".py") and (root / name).is_file():
                source = (root / name).read_text(encoding="utf-8", errors="replace")
                if _DECLARATION in source:
                    trees.append(cls(path=name, source=source))
        return trees

    def _walk(self, node: Declared, above: tuple[str, ...]) -> Iterator[Entry]:
        """`node` and, below it, every member that is no default command."""
        words = (*above, node.name)
        yield Entry(
            column=node.column,
            command=" ".join(words),
            depth=len(words) - 1,
            kind=node.kind,
            line=node.line,
            parent=" ".join(above),
            shown=node.show,
        )
        for member in (*node.commands, *node.children):
            if not member.default:
                yield from self._walk(member, words)

    def entries(self) -> list[Entry]:
        """Every node and command under the root, the root included, hidden ones too.

        Raises ValueError for a declaration that is not plain data.
        """
        for statement in ast.parse(self.source).body:
            match statement:
                case ast.Assign(
                    targets=[ast.Name(id="TREE")], value=ast.Call(func=ast.Name(id="Node")) as call
                ):
                    root = Reader(path=self.path).declared(call, Kind.GROUP)
                    return list(self._walk(root, ()))
        return []


class Cli(Family):
    """mcmr's CLI facts for a project's declared command tree: one fact per node and command,
    held by the module declaring `TREE = Node(...)`."""

    family = "CliFact"
    relations = {
        "facts": """
            SELECT row_number() OVER (ORDER BY path, line, "column") - 1 AS fact_order,
                   'cli:' || path || ':' || command AS fact_id, path, line AS start_line,
                   "column" AS start_column, line AS end_line, "column" AS end_column,
                   'python' AS language, command, parent, kind, shown, depth,
                   0 AS evidence_length, true AS evidence_present
            FROM cli_entries
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Load the declared trees as `cli_entries`."""
        facts.connection.execute(_ENTRIES_TABLE_SQL)
        rows = [
            (tree.path, e.command, e.parent, e.kind, e.shown, e.depth, e.line, e.column)
            for tree in CommandTree.declared_in(root)
            for e in cls._read(tree)
        ]
        if rows:
            facts.connection.executemany(_INSERT_ENTRY_SQL, rows)

    @staticmethod
    def _read(tree: CommandTree) -> list[Entry]:
        """The entries of `tree`, none for one that does not parse or is no plain data.

        A half-written declaration is logged, so it never stops the lint that reports it.
        """
        try:
            return tree.entries()
        except (SyntaxError, ValueError) as fault:
            logger.warning("{}: no command tree read: {}", tree.path, fault)
            return []
