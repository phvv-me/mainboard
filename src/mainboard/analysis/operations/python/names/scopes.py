# One Python source's scopes as symtable states them, and where a name binds and is read in them.

import ast
import re
import symtable
from collections import Counter
from operator import attrgetter
from typing import TYPE_CHECKING

from ....findings import Span
from .spelling import Position, Spelling

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

# A scope as symtable names it: its name and the line it opens on.
type Key = tuple[str, int]
type Scope = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef

# The names that read a scope's names by their spelling.
_REFLECTIVE = frozenset({"locals", "vars", "eval", "exec"})
# What opens a scope whose body binds names of its own.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.GeneratorExp)
# The keywords opening a definition, whose name follows, and a string literal's prefix and quote.
_DEFINING = re.compile(rb"(?:async\s+)?(?:def|class)\s+")
_OPENING = rb"[A-Za-z]*('''|\"\"\"|'|\")"


class Scopes:
    """One Python source's scopes as symtable states them, and where a name binds and is read.

    A scope is the module (line 0) or the function whose `def` opens on a line. Spans are the
    tokens a rename replaces, and the methods finding them answer None when they cannot place
    every one with certainty.
    """

    @staticmethod
    def _arguments(args: ast.arguments) -> list[ast.AST]:
        """What a signature evaluates around its function: defaults and annotations."""
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        notes = [each.annotation for each in every if each and each.annotation]
        return [*args.defaults, *filter(None, args.kw_defaults), *notes]

    @staticmethod
    def _exported(node: ast.stmt) -> list[ast.expr]:
        """The entries of the list or tuple `node` assigns to `__all__`, none for any other."""
        match node:
            case (
                ast.Assign(targets=[ast.Name(id="__all__")], value=listed)
                | ast.AugAssign(target=ast.Name(id="__all__"), value=listed)
                | ast.AnnAssign(target=ast.Name(id="__all__"), value=listed)
            ) if isinstance(listed, ast.List | ast.Tuple):
                return listed.elts
        return []

    @staticmethod
    def _header(node: ast.AST) -> list[ast.AST]:
        """What of a hidden scope still reads the scope around it, its body aside.

        Decorators, defaults and annotations are evaluated around a function, and the first
        iterable around a generator.
        """
        match node:
            case ast.GeneratorExp(generators=[first, *_]):
                return [first.iter]
            case ast.Lambda(args=args):
                return Scopes._arguments(args)
            case (
                ast.FunctionDef(args=args, decorator_list=wrapped, returns=returns)
                | ast.AsyncFunctionDef(args=args, decorator_list=wrapped, returns=returns)
            ):
                return [*wrapped, *filter(None, [returns]), *Scopes._arguments(args)]
        return []

    @staticmethod
    def _key(node: ast.AST) -> Key | None:
        """The key of the scope `node` opens, None for a node that opens none."""
        match node:
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                return name, node.lineno
            case ast.Lambda():
                return "lambda", node.lineno
            case ast.GeneratorExp():
                return "genexpr", node.lineno
        return None

    @staticmethod
    def _seen(node: ast.AST, hidden: Collection[Key]) -> list[ast.AST]:
        """The syntax under `node` that reads the scope `node` stands in.

        That is all of it, but for the body of a hidden scope.
        """
        if Scopes._key(node) in hidden:
            return Scopes._header(node)
        return list(ast.iter_child_nodes(node))

    @staticmethod
    def _tables(table: symtable.SymbolTable) -> list[symtable.SymbolTable]:
        """`table` and every table inside it."""
        return [
            table,
            *(inner for child in table.get_children() for inner in Scopes._tables(child)),
        ]

    def __init__(self, content: str, *, path: str) -> None:
        """Raises SyntaxError when `content` is no Python."""
        self.content, self.path = content, path
        self.lines = content.split("\n")
        self.tree = ast.parse(content)
        self.table = symtable.symtable(content, path, "exec")
        self.keys = Counter(key for node in ast.walk(self.tree) if (key := Scopes._key(node)))

    def _content(self, entry: ast.Constant, *, old: str) -> Span | None:
        """The span inside the quotes of the one-line string literal `entry` holding `old`."""
        text = self.lines[entry.lineno - 1].encode()[entry.col_offset : entry.end_col_offset]
        quoted = re.fullmatch(_OPENING + re.escape(old.encode()) + rb"\1", text)
        if quoted is None or entry.end_lineno != entry.lineno:
            return None
        first = entry.col_offset + quoted.end(1)
        return self._span((entry.lineno, first), end=(entry.lineno, first + len(old.encode())))

    def _definitions(self, old: str) -> list[Span]:
        """The spans of the name of each function or class `old` the module's scope defines."""
        defined = [
            node
            for node in self._levels()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.name == old
        ]
        return [span for node in defined if (span := self._named(node)) is not None]

    def _exports(self, old: str) -> list[Span] | None:
        """The spans inside the `__all__` strings naming `old`, None when one is unplaceable."""
        entries = [
            entry
            for node in self._levels()
            for entry in Scopes._exported(node)
            if isinstance(entry, ast.Constant) and entry.value == old
        ]
        spans = [self._content(entry, old=old) for entry in entries]
        return None if None in spans else [span for span in spans if span is not None]

    def _extent(self, scope: Scope) -> tuple[Position, Position]:
        """Where the body of `scope` starts and ends."""
        if isinstance(scope, ast.Module):
            return (1, 0), (len(self.lines) + 1, 0)
        first = scope.body[0]
        return (first.lineno, first.col_offset), (scope.end_lineno or 0, scope.end_col_offset or 0)

    def _hidden(
        self,
        table: symtable.SymbolTable,
        old: str,
        *,
        reading: Callable[[symtable.Symbol], bool],
    ) -> frozenset[Key] | None:
        """The scopes inside `table` where `old` names another binding than the one renamed.

        `reading` tells a scope that reads that binding from one that binds its own. None when a
        class is among them, or the syntax tree cannot tell which scope on a line is which.
        """
        inside = [
            each
            for each in Scopes._tables(table)[1:]
            if old in each.get_identifiers() and not reading(each.lookup(old))
        ]
        keys = {(each.get_name(), each.get_lineno()) for each in inside}
        classes = any(each.get_type() == symtable.SymbolTableType.CLASS for each in inside)
        return None if classes or any(self.keys[key] != 1 for key in keys) else frozenset(keys)

    def _levels(self) -> Iterator[ast.stmt]:
        """The statements in the module's own scope, those inside compound ones among them."""
        pending = list(ast.iter_child_nodes(self.tree))
        while pending:
            node = pending.pop()
            if isinstance(node, ast.stmt):
                yield node
            if not isinstance(node, _SCOPES):
                pending.extend(ast.iter_child_nodes(node))

    def _named(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> Span | None:
        """The span of the name a definition gives, None when the line does not spell it so."""
        line, name = self.lines[node.lineno - 1].encode(), node.name.encode()
        opening = _DEFINING.match(line, node.col_offset)
        if opening is None or not line.startswith(name, opening.end()):
            return None
        return self._span(
            (node.lineno, opening.end()), end=(node.lineno, opening.end() + len(name))
        )

    def _read(self, roots: list[ast.stmt], old: str, hidden: Collection[Key]) -> list[Span]:
        """The spans of the variables `old` read or bound under `roots`, hidden bodies aside."""
        spans: list[Span] = []
        pending: list[ast.AST] = list(roots)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Name) and node.id == old:
                end = (node.end_lineno or node.lineno, node.end_col_offset or 0)
                spans.append(self._span((node.lineno, node.col_offset), end=end))
            pending.extend(Scopes._seen(node, hidden))
        return spans

    def _scope(self, line: int) -> tuple[Scope, symtable.SymbolTable] | None:
        """The module (line 0) or the one function opening on `line`, with its table."""
        if not line:
            return self.tree, self.table
        nodes = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.lineno == line
        ]
        tables = [
            table
            for table in Scopes._tables(self.table)
            if nodes
            and table.get_type() == symtable.SymbolTableType.FUNCTION
            and (table.get_name(), table.get_lineno()) == Scopes._key(nodes[0])
        ]
        return (nodes[0], tables[0]) if len(nodes) == 1 and len(tables) == 1 else None

    def _span(self, start: Position, *, end: Position) -> Span:
        """The span from `start` to `end`."""
        return Span(
            path=self.path,
            start_line=start[0],
            start_column=start[1],
            end_line=end[0],
            end_column=end[1],
        )

    def clash(self, line: int, new: str) -> bool:
        """Whether `new` is a name of the scope at `line` or of any scope inside it.

        It is one as a local, a global or builtin read, a free variable or a nested scope's own.
        """
        found = self._scope(line)
        return found is None or any(
            new in each.get_identifiers() for each in Scopes._tables(found[1])
        )

    def imported(self, line: int) -> bool:
        """Whether the `from ... import` statement opening on `line` binds names in the module."""
        return any(
            isinstance(node, ast.ImportFrom) and node.lineno == line for node in self._levels()
        )

    def local_spans(self, line: int, old: str) -> list[Span] | None:
        """The spans of `old`, a plain local of the function opening on `line`, and its reads.

        Scopes inside that bind `old` themselves keep their own. None when `old` is no plain
        local (a parameter, a global or nonlocal name), or a scope inside binds it in a way the
        syntax tree cannot place.
        """
        found = self._scope(line) if line else None
        if found is None or old not in found[1].get_identifiers():
            return None
        scope, table = found
        symbol = table.lookup(old)
        hidden = self._hidden(table, old, reading=symtable.Symbol.is_free)
        if not symbol.is_local() or symbol.is_parameter() or hidden is None:
            return None
        return sorted(
            self._read(scope.body, old, hidden), key=attrgetter("start_line", "start_column")
        )

    def module_spans(self, old: str) -> list[Span] | None:
        """The spans of `old`, a name the module binds: reads, definitions and `__all__` strings.

        None when `old` is no module-level binding, or a scope inside binds it in a way the
        syntax tree cannot place.
        """
        if old not in self.table.get_identifiers() or not self.table.lookup(old).is_local():
            return None
        hidden = self._hidden(self.table, old, reading=symtable.Symbol.is_global)
        exports = self._exports(old)
        if hidden is None or exports is None:
            return None
        spans = [*self._read(self.tree.body, old, hidden), *self._definitions(old), *exports]
        return sorted(spans, key=attrgetter("start_line", "start_column"))

    def reflective(self, line: int) -> bool:
        """Whether the scope at `line` reads names by their spelling (`locals`, `eval`)."""
        found = self._scope(line)
        return found is not None and any(
            (isinstance(node, ast.Name) and node.id in _REFLECTIVE)
            or (isinstance(node, ast.Attribute) and node.attr == "f_locals")
            for node in ast.walk(found[0])
        )

    def spelled(self, spans: Collection[Span], old: str, *, line: int = 0) -> bool:
        """Whether `spans` and the syntax tree account for every token spelling `old` at `line`.

        A token the spans leave must be a name the tree places apart from the renamed one.
        """
        found = self._scope(line)
        if found is None:
            return False
        start, end = self._extent(found[0])
        return Spelling(self.content, self.tree).covers(spans, old, start=start, end=end)
