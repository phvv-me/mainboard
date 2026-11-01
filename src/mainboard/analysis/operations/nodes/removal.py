# Removing syntax: a statement or definition goes with what belongs to it, an argument with the
# comma that parts it from its neighbor.

from itertools import takewhile
from typing import TYPE_CHECKING

from ...claims import Claim
from ...edits import Delete, Replace
from ...findings import Span
from .site import NodeOperation, Tree

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...edits import Edit
    from .dialects import Dialect
    from .site import Node, Site

# The semantic type prefixes of nodes whose body is one statement with no block around it: control
# constructs (an `if`, a loop, a `case`) and labeled statements.
_CONSTRUCTS = ("FLOW_", "EXECUTION_")


class DeleteNode(NodeOperation):
    """Delete the node a row names with its wrapper (a decorated definition's decorators), the
    comment lines right above it among its siblings and the blank lines above those, which stay
    when code follows the node at once and would otherwise run into what is above.

    The dialect's filler takes its place when its block holds nothing else, and the block is
    claimed, so a second deletion there waits a pass and sees what is left. A body that needs no
    statement (a module, a C++ scope) is not claimed.

    It refuses a node sharing a line with other code, one under a syntax error, the root, a
    language no dialect describes, a wrapper inside a wrapper, a node an annotation above would
    pass on to the next one (a Rust attribute, which the dialects do not model), and a node a
    control construct holds with no block around it (the body of a C `if` or loop, a `case`, a
    label), where the statement after it would become the construct's body.
    """

    @staticmethod
    def _alone(tree: Tree, node: Node, holder: Sequence[Node]) -> bool:
        """Whether `node` has its lines to itself.

        Nothing but blanks stands before it on its first line, and nothing but a comment ending on
        its last line stands after it.
        """
        if tree.indent(node.start) != node.column:
            return False
        if node.end_column == 0 and node.end_line > node.line:
            return True
        line = tree.data.find(b"\n", node.end)
        rest = tree.data[node.end : None if line < 0 else line]
        start = node.end + len(rest) - len(rest.lstrip())
        last = DeleteNode._last(node)
        return not rest.strip() or any(
            each.comment and each.start == start and DeleteNode._last(each) == last
            for each in holder
        )

    @staticmethod
    def _annotated(dialect: Dialect, node: Node, holder: Sequence[Node]) -> bool:
        """Whether the sibling before `node` that is no comment annotates the item after it.

        A wrapper is a whole statement of its own (a decorated definition), so it does not.
        """
        before = [each for each in holder if each.index < node.index and not each.comment]
        previous = [each for each in before if not each.trivial][-1:]
        return any(
            each.semantic == "METADATA_ANNOTATION" and each.type not in dialect.wrappers
            for each in previous
        )

    @staticmethod
    def _blanks(rows: Sequence[bytes], line: int, *, step: int) -> int:
        """The line reached from `line` over the blank lines beside it, up (-1) or down (1)."""
        while 0 <= line - 1 + step < len(rows) and not rows[line - 1 + step].strip():
            line += step
        return line

    @staticmethod
    def _first(tree: Tree, node: Node, holder: Sequence[Node]) -> Node:
        """The topmost of `node` and the comments on lines of their own right above it.

        A comment opening the file describes the file, not the node.
        """
        first = node
        for each in reversed([sibling for sibling in holder if sibling.index < node.index]):
            owned = each.start > 0 and tree.indent(each.start) == each.column
            if not (each.comment and owned and DeleteNode._last(each) == first.line - 1):
                break
            first = each
        return first

    @staticmethod
    def _last(node: Node) -> int:
        """The line `node`'s text ends on.

        A node ending with its newline ends on the line before.
        """
        return node.end_line - (node.end_column == 0 and node.end_line > node.line)

    @staticmethod
    def _lines(tree: Tree, node: Node, holder: Sequence[Node], *, path: str) -> Span:
        """The whole lines to delete with `node`.

        Those are the comments right above it and the blank lines above those, unless code follows
        at once, where those blank lines keep parting what is above from what is below. When the
        lines start the file, the blank lines below go as well.
        """
        first, last = DeleteNode._first(tree, node, holder).line, DeleteNode._last(node)
        rows = tree.data.removesuffix(b"\n").split(b"\n")
        top, bottom = (
            DeleteNode._blanks(rows, first, step=-1),
            DeleteNode._blanks(rows, last, step=1),
        )
        followed = bottom == last and any(
            each.index > node.index and not each.trivial for each in holder
        )
        start = first if followed else top
        return Span(path=path, start_line=start, end_line=bottom if start == 1 else last)

    def _dialect(self, site: Site) -> Dialect | None:
        """The syntax of the site's language, None when no dialect describes it."""
        try:
            return site.dialect
        except KeyError:
            self.refused(f"no dialect describes {site.language}")
            return None

    def _refusal(
        self, site: Site, tree: Tree, node: Node, holder: Sequence[Node], *, wrapped: int
    ) -> str:
        """Why the node cannot be deleted, empty when it can.

        node: what is deleted, the wrapper when the site's node has one.
        holder: the children of the node's parent.
        wrapped: how many wrappers stand between the site's node and the node deleted.
        """
        types = {node.type, *(each.type for each in site.ancestors)}
        parent = site.ancestors[wrapped : wrapped + 1]
        reasons = [
            (len(site.ancestors) == wrapped, "the node is the root"),
            (wrapped > 1, "wrappers nest"),
            ("ERROR" in types, "the node sits in a syntax error"),
            (not DeleteNode._alone(tree, node, holder), "the node shares a line with other code"),
            (
                DeleteNode._annotated(site.dialect, node, holder),
                "an annotation above the node would pass to the next one",
            ),
            (
                any(each.semantic.startswith(_CONSTRUCTS) for each in parent),
                "a control construct holds the node with no block around it",
            ),
        ]
        return next((why for refused, why in reasons if refused), "")

    def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
        if (dialect := self._dialect(site)) is None:
            return None
        tree = Tree.of(site)
        lifted = list(takewhile(lambda each: each.type in dialect.wrappers, site.ancestors))
        node, holder = (lifted[0], site.uncles) if lifted else (site.target, site.siblings)
        if why := self._refusal(site, tree, node, holder, wrapped=len(lifted)):
            self.refused(why)
            return None
        block = site.ancestors[len(lifted)]
        claim = [Claim(span=block.span(site.path))] if block.type in dialect.blocks else []
        statements = {each.node for each in holder if not (each.comment or each.trivial)}
        if dialect.filler and claim and statements == {node.node}:
            return [Replace(span=node.span(site.path), text=dialect.filler), *claim]
        return [Delete(span=DeleteNode._lines(tree, node, holder, path=site.path)), *claim]


class DropArgument(NodeOperation):
    """Drop the argument a row names from its call, keyword or positional (a class's base too).

    The comma that parts it from the next argument goes with it, or the one from the previous
    argument when it is the last. A decorator's call left without arguments loses its parentheses
    (`@f()` becomes `@f`), which the callee must accept bare, as `cuda.jit` and `helion.kernel`
    do. Two arguments of one call overlap, so the second drop waits a pass.

    It refuses a node that is no argument of a call, one under a syntax error, and an argument
    with a comment in it or between it and its neighbors, which the drop would lose or hand to
    the neighbor.
    """

    @staticmethod
    def _between(parent: Node, args: Sequence[Node], place: int) -> tuple[int, int]:
        """The bytes where `args[place]` and its comments may stand.

        That is from the end of the argument before it (the opening parenthesis for the first) to
        the start of the one after it (the closing parenthesis for the last).
        """
        return (
            args[place - 1].end if place else parent.start + 1,
            args[place + 1].start if place + 1 < len(args) else parent.end - 1,
        )

    @staticmethod
    def _reach(site: Site, parent: Node, args: Sequence[Node], place: int) -> tuple[int, int]:
        """The bytes dropped with `args[place]`.

        That is the argument and the comma beside it, or the parentheses it leaves empty.
        """
        if len(args) == 1:
            if [each.type for each in site.ancestors[1:3]] == ["call", "decorator"]:
                return parent.start, parent.end
            return parent.start + 1, parent.end - 1
        if place + 1 < len(args):
            return args[place].start, args[place + 1].start
        return args[place - 1].end, args[place].end

    @staticmethod
    def _span(data: bytes, path: str, start: int, *, end: int) -> Span:
        """Where the bytes `[start, end)` of the file `data` at `path` stand."""

        def at(offset: int) -> tuple[int, int]:
            return data.count(b"\n", 0, offset) + 1, offset - (data.rfind(b"\n", 0, offset) + 1)

        (first, column), (last, ending) = at(start), at(end)
        return Span(
            path=path, start_line=first, start_column=column, end_line=last, end_column=ending
        )

    def _arguments(self, site: Site) -> Node | None:
        """The argument list holding the site's node, None (with the reason) when none does."""
        parent, target = site.parent, site.target
        if parent is None or parent.type != "argument_list" or target.comment or target.trivial:
            self.refused("the node is no argument of a call")
        elif any(each.type == "ERROR" for each in site.ancestors):
            self.refused("the node sits in a syntax error")
        else:
            return parent
        return None

    def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
        if (parent := self._arguments(site)) is None:
            return None
        args = [each for each in site.siblings if not (each.comment or each.trivial)]
        place = [each.node for each in args].index(site.target.node)
        low, high = DropArgument._between(parent, args, place)
        if any(
            each.comment and low <= each.start and each.end <= high
            for each in (*site.siblings, *site.tree)
        ):
            self.refused("a comment stands in the argument or between it and its neighbors")
            return None
        start, end = DropArgument._reach(site, parent, args, place)
        data = site.content.encode()
        return [Replace(span=DropArgument._span(data, site.path, start, end=end), text="")]
