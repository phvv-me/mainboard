# Tests that differ only in literals, walked together: each position where their literals differ is
# an argument a parametrized test takes.

import re
from collections.abc import Collection, Sequence
from itertools import chain, count, takewhile

from patos import FrozenModel

from ...nodes import Node, Tree

# The types of the literal leaves a name can stand for.
_LITERALS = frozenset(
    {"concatenated_string", "false", "float", "integer", "none", "string", "true"}
)
# A string opening with an f or b prefix is no literal: its text is computed or bytes.
_PREFIXED = re.compile(rb"[A-Za-z]*[fFbB]")
# The parent types where a name cannot stand for a literal, with what they are: a bare string is a
# docstring, and a piece of an implicit concatenation is not an expression of its own.
_KEPT_BY = {
    "concatenated_string": "a piece of an implicit string concatenation",
    "expression_statement": "a bare string such as a docstring",
}
# The ancestor types that read a literal as syntax: a `case` pattern matches it, an annotation
# names a type by it.
_SYNTAX = frozenset({"case_pattern", "type", "type_parameter"})


class Slot(FrozenModel):
    """The literal positions where the aligned tests differ alike, one parameter for all of them.

    name: the parameter the positions become.
    nodes: the positions in the first test.
    texts: each test's literal there, in test order.
    """

    name: str
    nodes: tuple[Node, ...]
    texts: tuple[str, ...]


class Alignment:
    """Test bodies walked in parallel over their node trees: the same shape everywhere, the literal
    leaves that differ becoming slots, where leaves differing alike share a slot.

    A slot is named after the keyword it is the value of, the variable it is assigned to,
    `expected` when it is the right side of an asserted `==`, else `case_<n>` for its place `n`
    among the slots, by its first position. A name the first body already uses, or that `reserved`
    holds, gets `_<n>` appended, unless it is the variable the slot is assigned to, which the body
    binds already.
    """

    @staticmethod
    def _unused(name: str, *, blocked: Collection[str]) -> str:
        """`name`, or the first of `name_2`, `name_3` ... that `blocked` does not hold."""
        return next(
            each
            for each in chain([name], (f"{name}_{n}" for n in count(2)))
            if each not in blocked
        )

    def __init__(
        self, tree: Tree, bodies: Sequence[Node], *, reserved: Collection[str] = ()
    ) -> None:
        self.tree, self.bodies, self.reserved = tree, bodies, set(reserved)
        self.why = ""
        self._by_id: dict[int | None, Node] = {node.node: node for node in tree.nodes}
        self._found: dict[tuple[str, ...], list[Node]] = {}

    def _above(self, node: Node) -> list[Node]:
        """The ancestors of `node` in the tree, nearest first."""
        found, parent = [], self._by_id.get(node.parent)
        while parent is not None:
            found.append(parent)
            parent = self._by_id.get(parent.parent)
        return found

    def _asserted(self, node: Node) -> bool:
        """Whether `node` is part of an `assert` statement of the body."""
        kinds = takewhile(lambda kind: kind != "block", (each.type for each in self._above(node)))
        return "assert_statement" in kinds

    def _identifiers(self) -> set[str]:
        """The names the first body mentions, but for keywords of calls and members after a dot."""
        body = self.bodies[0]
        found = set()
        for node in self.tree.within(body.start, end=body.end):
            parent = self._by_id.get(node.parent)
            kids = self.tree.significant(parent) if parent else []
            member = parent is not None and (
                (parent.type == "keyword_argument" and kids[0].node == node.node)
                or (parent.type == "attribute" and kids[-1].node == node.node != kids[0].node)
            )
            if node.type == "identifier" and not member:
                found.add(self.tree.text(node).decode())
        return found

    def _literal(self, node: Node) -> bool:
        """Whether `node` is a literal a name can stand for."""
        if node.type == "string":
            return not _PREFIXED.match(self.tree.text(node))
        if node.type == "concatenated_string":
            return all(self._literal(kid) for kid in self.tree.significant(node))
        return node.type in _LITERALS

    def _misplaced(self, node: Node) -> str:
        """Why a name cannot stand for the literal at `node`, empty when it can."""
        types = [each.type for each in self._above(node)]
        if types[0] in _KEPT_BY:
            return f"a literal differs in {_KEPT_BY[types[0]]}"
        if _SYNTAX & set(types):
            return "a literal differs in a pattern or an annotation"
        return ""

    def _named(self, node: Node, number: int) -> tuple[str, bool]:
        """The name the slot at `node` takes, and whether the body assigns that variable already.

        number: the slot's place among the slots.
        """
        parent = self._by_id[node.parent]
        kids = self.tree.significant(parent)
        text = [self.tree.text(kid) for kid in kids]
        if parent.type == "keyword_argument" and kids[-1].node == node.node:
            return text[0].decode(), False
        if (
            parent.type == "assignment"
            and kids[0].type == "identifier"
            and kids[-1].node == node.node
            and text[-2] == b"="
        ):
            return text[0].decode(), True
        if (
            parent.type == "comparison_operator"
            and text[1:2] == [b"=="]
            and kids[-1].node == node.node
            and self._asserted(parent)
        ):
            return "expected", False
        return f"case_{number}", False

    def _walk(self, nodes: Sequence[Node]) -> bool:
        """Align `nodes`, one of each body, recording a slot where literals differ.

        Nodes align when their types are equal and, over the children that are neither trivia nor
        comments, they align in order. A leaf aligns when its bytes are equal, and a literal with
        any literal.
        """
        if all(self._literal(node) for node in nodes):
            texts = tuple(self.tree.text(node).decode() for node in nodes)
            if len(set(texts)) > 1:
                self._found.setdefault(texts, []).append(nodes[0])
            return True
        kids = [self.tree.significant(node) for node in nodes]
        alike = (
            len({node.type for node in nodes}) == 1
            and not any(self._literal(node) for node in nodes)
            and len({len(each) for each in kids}) == 1
            and (kids[0] or len({self.tree.text(node) for node in nodes}) == 1)
        )
        if not alike:
            self.why = "the tests differ in more than literals"
            return False
        return all(self._walk(group) for group in zip(*kids, strict=True))

    def slots(self) -> list[Slot] | None:
        """The slots in the order of the first body, None when the bodies cannot be parametrized.

        `why` then says so: their shapes differ, a non-literal differs, or a literal differs where
        a name cannot stand.
        """
        if not self._walk(self.bodies):
            return None
        nodes = [node for group in self._found.values() for node in group]
        self.why = next((why for node in nodes if (why := self._misplaced(node))), "")
        if self.why:
            return None
        used, mentioned, slots = set(self.reserved), self._identifiers(), []
        for number, (texts, group) in enumerate(self._found.items()):
            name, bound = self._named(group[0], number)
            name = Alignment._unused(name, blocked=used if bound else used | mentioned)
            used.add(name)
            slots.append(Slot(name=name, nodes=tuple(group), texts=texts))
        return slots
