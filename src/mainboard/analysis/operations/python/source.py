# A Python source as the operations read it: its top-level definitions, names and reads.

import ast
import re
from itertools import takewhile
from typing import TYPE_CHECKING, NamedTuple

from ...edits import Delete
from ...findings import Span

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

# The builtins a literal constant may call on literals: `frozenset({...})`, `tuple(...)`.
_BUILDERS = frozenset({"frozenset", "tuple", "set", "dict", "list"})

# What opens a scope of its own, whose body binds nothing in the scope around it.
_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

# The syntax a literal value is made of, operators and load contexts included.
_LITERAL = (
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.UnaryOp,
    ast.BinOp,
    ast.operator,
    ast.unaryop,
    ast.expr_context,
)


class Block(NamedTuple):
    """One top-level definition with the comment lines right above it.

    first: the first line, its leading comments included.
    last: the definition's last line.
    text: those lines as written.
    shape: the definition's syntax tree, dumped, which two spellings of one value share.
    """

    first: int
    last: int
    text: str
    shape: str


class Module:
    """One Python source as the operations read it: its definitions, names and mentions."""

    @staticmethod
    def _alone(node: ast.stmt, lines: Sequence[str], *, literal: bool) -> bool:
        """Whether `node` stands alone on its lines, and, when `literal`, assigns a literal."""
        tail = lines[(node.end_lineno or node.lineno) - 1][node.end_col_offset or 0 :].strip()
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else node
        if value is None:
            return False
        kept = not literal or (value is not node and Module._is_literal(value))
        return kept and (not tail or tail.startswith("#"))

    @staticmethod
    def _is_literal(node: ast.AST) -> bool:
        """Whether `node` is a literal value, a builtin collection of literals among them."""
        if isinstance(node, ast.Call):
            called = node.func.id if isinstance(node.func, ast.Name) else ""
            literal = all(map(Module._is_literal, node.args))
            return called in _BUILDERS and not node.keywords and literal
        children = ast.iter_child_nodes(node)
        return isinstance(node, _LITERAL) and all(map(Module._is_literal, children))

    @staticmethod
    def binds(node: ast.stmt) -> set[str]:
        """Every name `node` binds in the scope it stands in, an import's names among them.

        A function or class binds its own name there; what its body binds stays inside it, as do
        a lambda's parameters and a comprehension's variables.
        """
        names: set[str] = set()
        pending: list[ast.AST] = [node]
        while pending:
            match each := pending.pop():
                case (
                    ast.FunctionDef(name=name)
                    | ast.AsyncFunctionDef(name=name)
                    | ast.ClassDef(name=name)
                ):
                    names.add(name)
                case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                    names |= {alias.asname or alias.name.partition(".")[0] for alias in aliases}
                case ast.Name(id=name, ctx=ast.Store()):
                    names.add(name)
            pending.extend(() if isinstance(each, _SCOPES) else ast.iter_child_nodes(each))
        return names

    @staticmethod
    def target(node: ast.stmt) -> str:
        """The one name a top-level assignment, function or class binds, else empty."""
        match node:
            case ast.Assign(targets=[ast.Name(id=name)]) | ast.AnnAssign(target=ast.Name(id=name)):
                return name
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                return name
        return ""

    def __init__(self, content: str | None) -> None:
        self.content = content or ""
        self.tree = ast.parse(self.content)

    def block(self, name: str, *, literal: bool = True) -> Block | None:
        """The one top-level definition of `name`, alone on its lines, its decorators included.

        None when `name` is bound more than once or shares a line with another statement, and,
        when `literal`, when it is anything but an assignment of a literal.
        """
        assigned = [node for node in self.tree.body if Module.target(node) == name]
        lines = self.content.splitlines(keepends=True)
        if len(assigned) != 1 or not Module._alone(node := assigned[0], lines, literal=literal):
            return None
        last = node.end_lineno or node.lineno
        start = min([node.lineno, *(each.lineno for each in getattr(node, "decorator_list", []))])
        above = takewhile(lambda line: line.lstrip().startswith("#"), reversed(lines[: start - 1]))
        first = start - len(list(above))
        text = "".join(lines[first - 1 : last]).rstrip("\n") + "\n"
        return Block(first, last, text, ast.dump(node))

    def cleared(self, path: str, *, first: int, last: int) -> Delete:
        """Deleting lines `first` to `last` of the source at `path`, blank lines above too."""
        lines = self.content.splitlines()
        blank = takewhile(lambda line: not line.strip(), reversed(lines[: first - 1]))
        return Delete(span=Span(path=path, start_line=first - len(list(blank)), end_line=last))

    def mentions(self, name: str) -> bool:
        """Whether the source spells `name` anywhere, comments and strings included."""
        return re.search(rf"\b{re.escape(name)}\b", self.content) is not None

    def names(self) -> set[str]:
        """Every name the source binds or reads."""
        return {node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)}

    def read(self, names: Collection[str], *, outside: Iterable[Block]) -> list[str]:
        """The `names` read outside the blocks `outside`, `__all__` entries included."""
        skipped = {line for block in outside for line in range(block.first, block.last + 1)}
        spelled = {
            node.id if isinstance(node, ast.Name) else str(node.value)
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name | ast.Constant) and node.lineno not in skipped
        }
        return sorted(spelled & set(names))

    def spans(self, name: str, *, path: str, outside: Block) -> list[Span]:
        """Where the source spells `name` as a bare name, the lines of `outside` aside."""
        return [
            Span(
                path=path,
                start_line=node.lineno,
                start_column=node.col_offset,
                end_line=node.end_lineno,
                end_column=node.end_col_offset,
            )
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name)
            and node.id == name
            and not outside.first <= node.lineno <= outside.last
        ]

    def twin(self, block: Block) -> bool:
        """Whether the source defines the constant of `block` the same way at its top level."""
        return any(ast.dump(node) == block.shape for node in self.tree.body)

    def uses(self, bound: str) -> int:
        """How often the source reads the module bound as `bound` (`baseline`, `pkg.limits`)."""
        return sum(
            ast.unparse(node) == bound
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name | ast.Attribute)
        )
