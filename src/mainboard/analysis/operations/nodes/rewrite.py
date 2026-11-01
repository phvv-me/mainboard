# Rewriting one node: by the first declared pattern that unifies with it, or by the captures and
# the template the fix row's SQL bound, and one Replace of the node either way.

from itertools import pairwise, product
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict, Json, ValidationError

from ...edits import Import, Replace
from .patterns import Capture, Rewrite, Template
from .site import Node, NodeOperation, Site, Tree

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from ...edits import Edit


class RewriteRow(FrozenModel):
    """What a `rewrite-node` fix row states beside its node.

    That is the declared rewrites with the parse of each pattern, or the captures and template
    that its SQL bound.

    captures: the ids of the nodes each hole takes (row-bound).
    imports: the statements the template needs (row-bound).
    patterns: the nodes of each declared pattern's parse, one list per rewrite.
    rewrites: the declared rewrites, as JSON.
    template: the text that replaces the node, holes naming the captures (row-bound).
    """

    model_config = ConfigDict(extra="ignore")

    captures: dict[str, tuple[int, ...] | None] = {}
    imports: tuple[str, ...] = ()
    patterns: tuple[tuple[Node, ...], ...] = ()
    rewrites: Json[tuple[Rewrite, ...]] = ()
    template: str | None = None


class Unifier:
    """Binds a pattern's tree to a target subtree, each hole capturing target nodes.

    Two nodes unify when their types are equal and, over the children that are neither trivia nor
    comments, they unify in order; leaves compare by bytes. A hole is an identifier opening with
    the expando, or a statement holding only one, which takes any one statement.
    """

    @staticmethod
    def _closes(kids: Sequence[Node], index: int) -> bool:
        """Whether nothing but comments follows `kids[index - 1]`."""
        return all(kid.comment for kid in kids[index:])

    @staticmethod
    def _trailing(kids: Sequence[Node], index: int) -> bool:
        """Whether the comment `kids[index]` ends the line of the code before it, which owns it."""
        before = next((kid for kid in reversed(kids[:index]) if not kid.comment), None)
        return before is not None and before.end_line == kids[index].line

    def __init__(self, pattern: Tree, *, target: Tree, expando: str) -> None:
        self.pattern, self.target, self.expando = pattern, target, expando
        self.captures: dict[str, Capture] = {}
        self._nodes: dict[int | None, Node] = {node.node: node for node in target.nodes}

    def _capture(self, nodes: Sequence[Node], parent: Node | None) -> Capture:
        """The capture of `nodes`, a run of the children of `parent` when there is one."""
        if not nodes:
            return Capture(start=0, end=0, nodes=())
        start, end = nodes[0].start, nodes[-1].end
        if parent is not None:
            start, end = self._extent(nodes, parent)
        return Capture(start=start, end=end, nodes=tuple(node.node for node in nodes))

    def _captured(self, hole: str, nodes: Sequence[Node], parent: Node | None) -> bool:
        """Capture `nodes` under the hole's name, False when the name holds other bytes already."""
        if (name := hole.lstrip(self.expando)) == "_":
            return True
        capture = self._capture(nodes, parent)
        held = self.captures.setdefault(name, capture)
        data = self.target.data
        return data[held.start : held.end] == data[capture.start : capture.end]

    def _extent(self, nodes: Sequence[Node], parent: Node) -> tuple[int, int]:
        """The bytes of a run of `parent`'s children and the comments it owns.

        A comment above the run goes with it, unless it ends the line of the code before, which
        keeps it. One ending the run's last line goes with the run, and so do all that follow a
        run which closes its parent. A comment between two runs therefore belongs to the one
        below it, and no comment is held twice.
        """
        kids = self._kids(parent)
        first, last = kids.index(nodes[0]), kids.index(nodes[-1])
        while first and kids[first - 1].comment and not self._trailing(kids, first - 1):
            first -= 1
        while (
            last + 1 < len(kids)
            and kids[last + 1].comment
            and (kids[last + 1].line == kids[last].end_line or self._closes(kids, last + 1))
        ):
            last += 1
        return self._opening(kids, first, parent), kids[last].end

    def _hole(self, node: Node) -> str | None:
        """The hole the pattern node stands for, None when it is syntax."""
        if node.type == "identifier":
            text = self.pattern.text(node).decode()
            return text if text.startswith(self.expando) else None
        inner = self.pattern.significant(node)
        lone = node.type == "expression_statement" and len(inner) == 1
        return self._hole(inner[0]) if lone and not self.pattern.significant(inner[0]) else None

    def _kids(self, node: Node) -> list[Node]:
        """The children of `node` in order, comments included and trivia left out."""
        return [kid for kid in self.target.children(node) if not kid.trivial]

    def _opening(self, kids: Sequence[Node], first: int, parent: Node) -> int:
        """The byte a run starting at `kids[first]` opens at.

        A run that opens its parent also owns the comments above the parent (those between `try:`
        and its body belong to the body).
        """
        start = kids[first].start
        above = self._kids(up) if (up := self._nodes.get(parent.parent)) else []
        if above and not first:
            index = above.index(parent)
            while index and above[index - 1].comment and not self._trailing(above, index - 1):
                index -= 1
                start = above[index].start
        return start

    def _run(self, patterns: Sequence[Node], *, targets: Sequence[Node], parent: Node) -> bool:
        """Whether the pattern nodes unify with the target nodes one for one."""
        if not patterns:
            return not targets
        first, rest = patterns[0], patterns[1:]
        hole = self._hole(first)
        if hole is not None and hole.startswith(self.expando * 3):
            return self._spread(hole, rest, targets=targets, parent=parent)
        saved = dict(self.captures)
        unified = bool(targets) and self.unify(first, target=targets[0])
        if unified and self._run(rest, targets=targets[1:], parent=parent):
            return True
        self.captures = saved
        return False

    def _spread(
        self, hole: str, rest: Sequence[Node], *, targets: Sequence[Node], parent: Node
    ) -> bool:
        """Whether a `$$$` hole can take the longest run of targets that lets the rest unify."""
        saved = dict(self.captures)
        for cut in range(len(targets), -1, -1):
            if self._captured(hole, targets[:cut], parent) and self._run(
                rest, targets=targets[cut:], parent=parent
            ):
                return True
            self.captures = dict(saved)
        return False

    def bind(self, name: str, nodes: Sequence[Node]) -> bool:
        """Capture `nodes`, siblings, as `name`: the row-bound form, where SQL names the nodes."""
        parent = self._nodes.get(nodes[0].parent) if nodes else None
        return self._captured(name, nodes, parent)

    def texts(self) -> dict[str, str]:
        """The text each hole took, by name."""
        data = self.target.data
        return {name: data[c.start : c.end].decode() for name, c in self.captures.items()}

    def unify(self, pattern: Node, *, target: Node) -> bool:
        """Whether the pattern node unifies with the target node, holes captured as it goes."""
        if (hole := self._hole(pattern)) is not None:
            return self._captured(hole, [target], None)
        if pattern.type != target.type:
            return False
        if not (inner := self.pattern.significant(pattern)):
            return self.pattern.text(pattern) == self.target.text(target)
        return self._run(inner, targets=self.target.significant(target), parent=target)


class RewriteNode(NodeOperation):
    """Rewrite one node and leave nothing it holds behind.

    Declared: the first pattern that unifies with the node, or with a node of its subtree that
    starts where it starts, rendered through its template. Row-bound: the template of the row,
    over the nodes its SQL names. The node becomes the text, and the imports it needs are added.
    A comment the text would lose refuses the rewrite. So does, in the row-bound form, any other
    text outside every capture (keywords aside), since SQL chose them: a row drops text by
    capturing it under a name its template leaves out.
    """

    @staticmethod
    def _comment(tree: Tree, node: Node) -> bool:
        """Whether `node` is a comment."""
        return node.comment

    @staticmethod
    def _disjoint(unifier: Unifier) -> None:
        """Raise ValueError when two captures share bytes: the row's SQL is wrong."""
        spans = sorted((c.start, c.end, name) for name, c in unifier.captures.items() if c.nodes)
        for (_, end, one), (start, _, other) in pairwise(spans):
            if start < end:
                raise ValueError(f"the captures {one} and {other} overlap")

    @staticmethod
    def _replacement(site: Site, at: Node, text: str, imports: Sequence[str]) -> list[Edit]:
        """`text` where `at` stood, and the imports it needs."""
        return [
            Replace(span=at.span(site.path), text=text),
            *(Import(path=site.path, statement=statement) for statement in imports),
        ]

    @staticmethod
    def _roots(pattern: Tree) -> list[Node]:
        """The nodes of a pattern to unify: its statement, and the expression that fills it."""
        module = next(node for node in pattern.nodes if node.parent is None)
        if len(statements := pattern.significant(module)) != 1:
            raise ValueError(f"a pattern is one statement, not {len(statements)}")
        (top,) = statements
        inner = pattern.significant(top)[:1] if top.type == "expression_statement" else []
        return [top, *inner]

    @staticmethod
    def _written(tree: Tree, node: Node) -> bool:
        """Whether `node` is text a rewrite must keep: a comment, or a leaf that is no keyword."""
        return (
            not tree.children(node) and not node.trivial and tree.text(node).decode() != node.type
        )

    def _bound(self, site: Site, row: RewriteRow) -> list[Edit] | None:
        """The row-bound form: the captures bind the nodes SQL names, the template is its own."""
        tree = Tree.of(site)
        nodes = {node.node: node for node in site.tree}
        template = Template(row.template or "", dialect=site.dialect)
        unifier = Unifier(tree, target=tree, expando=site.dialect.expando)
        for name, ids in row.captures.items():
            unifier.bind(name, [nodes[node] for node in ids or ()])
        self._disjoint(unifier)
        if unbound := template.holes() - unifier.captures.keys():
            raise ValueError(f"the template keeps {sorted(unbound)}, which no capture names")
        text = self._rewritten(site, unifier, template, site.target, among=self._written)
        return None if text is None else self._replacement(site, site.target, text, row.imports)

    def _declared(self, site: Site, row: RewriteRow) -> list[Edit] | None:
        """The declared form: the first pattern that unifies and whose `where` admits it."""
        for rewrite, unifier, at in self._unified(site, row):
            if rewrite.admits(unifier.texts()):
                template = Template(rewrite.to, dialect=site.dialect)
                text = self._rewritten(site, unifier, template, at, among=self._comment)
                return None if text is None else self._replacement(site, at, text, rewrite.imports)
        self.refused("no declared pattern unifies")
        return None

    def _lost(
        self,
        unifier: Unifier,
        template: Template,
        at: Node,
        *,
        among: Callable[[Tree, Node], bool],
    ) -> Node | None:
        """The first node of `at` that `among` selects and the captures do not hold.

        A comment must lie in a capture the template keeps. Any other node must lie in some
        capture, kept or not: a row that captures text under a name its template leaves out has
        chosen to drop it.
        """
        tree, used = unifier.target, template.holes()
        named = {name: (c.start, c.end) for name, c in unifier.captures.items() if c.nodes}
        kept, held = [s for name, s in named.items() if name in used], [*named.values()]
        return next(
            (
                node
                for node in tree.within(at.start, end=at.end)
                if among(tree, node)
                and not any(
                    start <= node.start and node.end <= end
                    for start, end in (kept if node.comment else held)
                )
            ),
            None,
        )

    def _rewritten(
        self,
        site: Site,
        unifier: Unifier,
        template: Template,
        at: Node,
        *,
        among: Callable[[Tree, Node], bool],
    ) -> str | None:
        """The text that replaces `at`, None (with the reason) when it cannot be made safely.

        That is when it would lose a node `among` selects, or when a capture cannot stand where
        its hole does without changing its meaning.
        """
        if (lost := self._lost(unifier, template, at, among=among)) is not None:
            left = unifier.target.text(lost).decode()
            self.refused(
                "a comment outside the captures the template keeps would be lost"
                if lost.comment
                else f"the captures leave out `{left}`"
            )
            return None
        parents: dict[int | None, Node] = {
            node.node: node for node in (*site.ancestors[:1], *site.tree)
        }
        text = template.rendered(
            unifier.captures, unifier.target, at, parent=parents.get(at.parent)
        )
        if text is None:
            self.refused("a capture cannot stand where its hole does without changing its meaning")
        return text

    def _tried(self, site: Site, row: RewriteRow) -> list[tuple[Rewrite, Tree, Node, Node]]:
        """The declared patterns of the site's language, each with the nodes it may unify.

        That is a node of the row's subtree that starts where the row's node does (largest first)
        and a pattern node of its type, in the order tried.
        """
        starting = [node for node in site.tree if node.start == site.target.start]
        largest = sorted(starting, key=lambda node: node.start - node.end)
        patterns = [
            (rewrite, Tree(nodes, rewrite.expando().encode()))
            for rewrite, nodes in zip(row.rewrites, row.patterns, strict=True)
            if rewrite.language == site.language
        ]
        return [
            (rewrite, pattern, at, root)
            for rewrite, pattern in patterns
            for at, root in product(largest, self._roots(pattern))
            if root.type == at.type
        ]

    def _unified(self, site: Site, row: RewriteRow) -> Iterator[tuple[Rewrite, Unifier, Node]]:
        """Each declared pattern with the node it unifies with, bound, in the order tried."""
        tree = Tree.of(site)
        for rewrite, pattern, at, root in self._tried(site, row):
            unifier = Unifier(pattern, target=tree, expando=site.dialect.expando)
            if unifier.unify(root, target=at):
                yield rewrite, unifier, at

    def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
        try:
            wanted = RewriteRow.model_validate(row)
        except ValidationError as error:
            self.refused(f"the row's {error.errors()[0]['loc'][0]} is malformed")
            return None
        return (
            self._declared(site, wanted) if wanted.template is None else self._bound(site, wanted)
        )
