# Merging tests that differ only in literals into one parametrized test with a case for each.

import re
from collections.abc import Mapping, Sequence
from operator import attrgetter

from patos import FrozenModel
from pydantic import ConfigDict, ValidationError

from ....edits import Edit, Import, Replace
from ...nodes import DeleteNode, Node, NodeOperation, Site, Tree
from .alignment import Alignment, Slot

# What marks a test parametrized (or property based) already.
_PARAMETRIZED = re.compile(rb"parametrize|\bgiven\b")
# Pytest's own fixtures, which a parameter of the same name would stand in for.
_FIXTURES = frozenset(
    {
        "cache",
        "capfd",
        "capfdbinary",
        "caplog",
        "capsys",
        "capsysbinary",
        "doctest_namespace",
        "monkeypatch",
        "pytestconfig",
        "record_property",
        "recwarn",
        "request",
        "tmp_path",
        "tmp_path_factory",
        "tmpdir",
        "tmpdir_factory",
    }
)
# The statements that name a variable of an outer scope, which a parameter cannot share.
_OUTER = frozenset({"global_statement", "nonlocal_statement"})
# The parameters a parameter without a default may not follow: one with a default, a splat, or the
# bare `*` that makes the rest keyword-only.
_OPTIONAL = frozenset(
    {
        "default_parameter",
        "dictionary_splat_pattern",
        "keyword_separator",
        "list_splat_pattern",
        "typed_default_parameter",
    }
)


class MergeRow(FrozenModel):
    """What a `merge-tests` fix row states beside its node, the holder of the tests.

    ids: each test's case id, in test order.
    name: the merged test's name.
    tests: the ids of the tests' function nodes.
    """

    model_config = ConfigDict(extra="ignore")

    ids: tuple[str, ...]
    name: str
    tests: tuple[int, ...]


class Group:
    """The tests a merge row names, found in the site of their holder, in source order."""

    @property
    def first(self) -> Node:
        """The test the others merge into."""
        return self.tests[0]

    @staticmethod
    def _decorator(slots: Sequence[Slot], ids: Sequence[str]) -> str:
        """The `parametrize` mark over the slots' names with a case for each id."""
        names = [f'"{slot.name}"' for slot in slots]
        argnames = names[0] if len(names) == 1 else f"({', '.join(names)})"
        cases = [
            f'pytest.param({", ".join(slot.texts[place] for slot in slots)}, id="{each}")'
            for place, each in enumerate(ids)
        ]
        return f"@pytest.mark.parametrize({argnames}, [{', '.join(cases)}])"

    def __init__(self, site: Site, row: MergeRow) -> None:
        self.site, self.row = site, row
        self.tree = Tree.of(site)
        self.by_id: dict[int | None, Node] = {node.node: node for node in site.tree}
        self.tests = sorted((self.by_id[number] for number in row.tests), key=attrgetter("start"))
        for test in self.tests:
            for kind in ("identifier", "parameters", "block"):
                self.part(test, kind)

    def _added(self, params: Node, names: str) -> tuple[int, str]:
        """Where `names` go among the parameters, and the text that adds them.

        They stand before the first parameter with a default or a splat, else after the last one.
        """
        later = next((kid for kid in self.tree.children(params) if kid.type in _OPTIONAL), None)
        if later is not None:
            return later.start, f"{names}, "
        inner = self.tree.data[params.start + 1 : params.end - 1].rstrip()
        end = params.start + 1 + len(inner)
        return end, names if not inner else f"{' ' if inner.endswith(b',') else ', '}{names}"

    def _commented(self) -> bool:
        """Whether a comment stands in a test the merge deletes or in the first one's signature."""
        body = self.part(self.first, "block")
        ends = [(test, test.end) for test in self.tests[1:]] + [(self.first, body.start)]
        return any(
            node.comment for test, end in ends for node in self.tree.within(test.start, end=end)
        )

    def _frame(self, test: Node) -> tuple[tuple[bytes, ...], bytes]:
        """The decorators of `test` and the bytes from its name to its body.

        Tests merged share both.
        """
        wrapper = self.by_id[test.parent]
        decorators = tuple(
            self.tree.text(kid) for kid in self.tree.children(wrapper) if kid.type == "decorator"
        )
        name, body = self.part(test, "identifier"), self.part(test, "block")
        return decorators, self.tree.data[name.end : body.start]

    def _head(self) -> bytes:
        """The bytes before the first test on its line."""
        data = self.tree.data
        return data[data.rfind(b"\n", 0, self.first.start) + 1 : self.first.start]

    def _inside(self, node: Node) -> bool:
        """Whether `node` lies in one of the tests."""
        return any(test.start <= node.start and node.end <= test.end for test in self.tests)

    def _outside(self) -> set[str]:
        """The identifiers spelled outside the tests."""
        return {
            self.tree.text(node).decode()
            for node in self.site.tree
            if node.type == "identifier" and not self._inside(node)
        }

    def descended(self, test: Node) -> Site:
        """The site of `test`, built from the site of the holder."""
        chain = [self.by_id[test.parent]]
        while chain[-1].node != self.site.node:
            chain.append(self.by_id[chain[-1].parent])
        return self.site.model_copy(
            update={
                "node": test.node,
                "tree": tuple(self.tree.within(test.start, end=test.end)),
                "siblings": tuple(self.tree.children(chain[0])),
                "uncles": tuple(self.tree.children(chain[1]))
                if len(chain) > 1
                else self.site.siblings,
                "ancestors": (*chain, *self.site.ancestors),
            }
        )

    def part(self, test: Node, kind: str) -> Node:
        """The first child of `test` of the type `kind`, a KeyError when it has none."""
        try:
            return next(kid for kid in self.tree.children(test) if kid.type == kind)
        except StopIteration:
            raise KeyError(kind) from None

    def refusal(self) -> str:
        """Why the tests cannot merge, empty when they can."""
        frames = {self._frame(test) for test in self.tests}
        decorators, _ = next(iter(frames))
        params = self.part(self.first, "parameters")
        reasons = [
            (len(self.tests) != len(self.row.ids), "the row names tests and ids that differ"),
            (len(frames) > 1, "the tests differ in decorators or signature"),
            (
                bool(_PARAMETRIZED.search(b"".join(decorators))),
                "the tests are parametrized or property based already",
            ),
            (b"/" in self.tree.text(params), "the tests take positional-only parameters"),
            (self._commented(), "a comment stands in a test the merge deletes or in a signature"),
            (
                any(node.type in _OUTER and self._inside(node) for node in self.site.tree),
                "a test names a variable of an outer scope",
            ),
            (self.row.name in self._outside(), "the merged name is used outside the tests"),
            (bool(self._head().strip()), "the first test shares its line with other code"),
        ]
        return next((why for refused, why in reasons if refused), "")

    def reserved(self) -> frozenset[str]:
        """The names a parameter must not take: pytest's fixtures and the test's parameters."""
        params = self.part(self.first, "parameters")
        return _FIXTURES | {
            self.tree.text(node).decode()
            for node in self.tree.within(params.start, end=params.end)
            if node.type == "identifier"
        }

    def template(self, slots: Sequence[Slot]) -> str:
        """The first test as the merged test, under the `parametrize` mark of the slots.

        It takes the merged name, the slots' names among its parameters and a slot's name in place
        of each of its literals.
        """
        data, first = self.tree.data, self.first
        name, params = self.part(first, "identifier"), self.part(first, "parameters")
        end, added = self._added(params, ", ".join(slot.name for slot in slots))
        pieces = sorted(
            [
                (name.start, name.end, self.row.name),
                (end, end, added),
                *((node.start, node.end, slot.name) for slot in slots for node in slot.nodes),
            ]
        )
        out, kept = [], first.start
        for start, finish, text in pieces:
            out += [data[kept:start], text.encode()]
            kept = finish
        out.append(data[kept : first.end])
        mark = Group._decorator(slots, self.row.ids)
        return f"{mark}\n{self._head().decode()}{b''.join(out).decode()}"


class MergeTests(NodeOperation):
    """Merge tests that differ only in literals into the first, parametrized over the values of
    each test with its id, and delete the others with their decorators and the comment lines above.

    Each position where the literals differ becomes a parameter (see `Alignment` for its name),
    and the first test is renamed. The merged module collects the same cases the tests were.

    It refuses tests that differ in decorators or signature, tests already parametrized or taking
    positional-only parameters, a difference that is no literal or a literal where a name cannot
    stand (a docstring, a pattern, an annotation), a comment in a test it would delete, a `global`
    or `nonlocal`, a merged name the module already uses, a first test sharing its line with other
    code, and any test `DeleteNode` refuses.
    """

    def _deleted(self, group: Group) -> list[Edit] | None:
        """The edits deleting all but the first test, None (with why) when one is refused."""
        deleter, made = DeleteNode(), []
        for test in group.tests[1:]:
            if (edits := deleter.edits({"site": group.descended(test)})) is None:
                self.refused(deleter.why)
                return None
            made += edits
        return made

    def _group(self, site: Site, row: Mapping) -> Group | None:
        """The tests the row names, None (with the reason) when they are not Python functions."""
        try:
            group = Group(site, MergeRow.model_validate(row))
        except KeyError, ValidationError:
            self.refused("the row does not name tests of the node")
            return None
        functions = {test.type for test in group.tests} == {"function_definition"}
        if site.language != "python" or len(group.tests) < 2 or not functions:
            self.refused("the row does not name two or more Python functions")
            return None
        return group

    def lowered(self, site: Site, row: Mapping) -> list[Edit] | None:
        if (group := self._group(site, row)) is None:
            return None
        if why := group.refusal():
            self.refused(why)
            return None
        bodies = [group.part(test, "block") for test in group.tests]
        alignment = Alignment(group.tree, bodies, reserved=group.reserved())
        if (slots := alignment.slots()) is None:
            self.refused(alignment.why)
            return None
        if (deleted := self._deleted(group)) is None:
            return None
        replaced = Replace(span=group.first.span(site.path), text=group.template(slots))
        return [replaced, *deleted, Import(path=site.path, statement="import pytest")]
