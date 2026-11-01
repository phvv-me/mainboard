# The bookkeeping of a fixpoint run: the fixes applied, the files they changed together and what
# they say they did to the ids of the tests pytest collects.

import re
from collections.abc import Iterable, Mapping, Set

from patos import FrozenModel

# A test as a change line names it: a pytest node id, or the bare name of a test function with the
# cases it became (`test_f[one]`). `->` joins a test to the one it became, or to `gone`.
_TEST = r"[^\s,]+::[^\s,]+|\btest\w*(?:\[[^\s,]*\])?"
_ENTRY = re.compile(rf"(?P<old>{_TEST}) -> (?P<new>{_TEST}|gone)")


class Applied(FrozenModel):
    """A fix applied in the mirror: where its finding stood in the file as that pass began, the
    rule's words for what it changed, the files it changed, and the renames its pass made (old
    name, new name), which move the ids of the tests among the files it changed."""

    rule: str
    path: str
    line: int
    change: str
    files: frozenset[str]
    moves: tuple[tuple[str, str], ...] = ()

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.rule} {self.change}"


class Components:
    """Files changed since their last proven state, joined when one fix changed several."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def _root(self, name: str) -> str:
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def around(self, names: Set[str]) -> set[str]:
        """Every file in a component holding one of `names`."""
        roots = {self._root(name) for name in names if name in self.parent}
        return {name for name in self.parent if self._root(name) in roots}

    def drop(self, names: Set[str]) -> None:
        """Forget whole components."""
        self.parent = {name: root for name, root in self.parent.items() if name not in names}

    def groups(self) -> list[set[str]]:
        found: dict[str, set[str]] = {}
        for name in self.parent:
            found.setdefault(self._root(name), set()).add(name)
        return list(found.values())

    def join(self, names: Iterable[str]) -> None:
        first, *rest = names
        self.parent.setdefault(first, first)
        for name in rest:
            self.parent.setdefault(name, name)
            self.parent[self._root(name)] = self._root(first)


class Remap(FrozenModel):
    """What one kept fix did to the ids of the tests in the files it changed.

    The fix says it in its change line, with one entry for each test it renamed or deleted:
    `old -> new` for a test that took another id (a merge keeping each test as a case:
    `test_f_one -> test_f[one]`), `old[*] -> new` for every parametrized case of a test that
    became one test, and `old -> gone` for a test it deleted, with its cases. A test is named as
    pytest spells its id (`t.py::K::test_f_one`), or by its function name alone, which stands for
    that test in every file the fix changed. Every other id stays, apart from the files the fix
    moved. The ids the line gives its tests (`promised`) are the ones it vouches for exactly.

    ids: each id the change line names with the id it became, None for a deleted test.
    moves: each file or folder the fix renamed, with its new name.
    scope: the files the fix changed, which a name with no file speaks of.
    """

    ids: tuple[tuple[str, str | None], ...]
    moves: tuple[tuple[str, str], ...]
    scope: frozenset[str]

    @classmethod
    def of(cls, fix: Applied) -> Remap:
        """What `fix` says: the ids its change line names and the files it moved."""
        return cls(
            ids=tuple(
                (entry["old"], None if entry["new"] == "gone" else entry["new"])
                for entry in _ENTRY.finditer(fix.change)
            ),
            moves=fix.moves,
            scope=fix.files,
        )

    @property
    def files(self) -> set[str]:
        """The files the change line speaks of.

        Those of the ids it names, and the files of the fix when it names a test by its function
        name alone.
        """
        named = {node for entry in self.ids for node in entry if node and Remap._filed(node)}
        bare = any(not Remap._filed(old) for old, _ in self.ids)
        return {node.partition("::")[0] for node in named} | (set(self.scope) if bare else set())

    @staticmethod
    def _filed(node: str) -> bool:
        """Whether the id `node` starts with a file, as pytest spells an id."""
        return "::" in node

    @staticmethod
    def _renamed(node: str, named: Mapping[str, str | None]) -> str | None:
        """The id `node` becomes, None when the fix deleted it.

        An entry for the id itself wins, then `base[*]`, then a deleted base (it takes its cases).
        """
        base = Remap.function(node)
        if node in named:
            return named[node]
        if f"{base}[*]" in named:
            return named[f"{base}[*]"]
        return None if named.get(base, node) is None else node

    @staticmethod
    def function(node: str) -> str:
        """The test function of the id `node`: the id without its `[case]`."""
        return node.partition("[")[0]

    def _after(self, nodes: Iterable[str]) -> set[str | None]:
        """The ids `nodes` become under the names of the change line, None for a deleted one."""
        found = set(nodes)
        named = self._named(found)
        return {Remap._renamed(node, named) for node in found}

    def _moved(self, node: str) -> str:
        """`node` in the file or folder as the fix moved it."""
        path, separator, rest = node.partition("::")
        for old, new in self.moves:
            if path == old or path.startswith(f"{old}/"):
                return new + path.removeprefix(old) + separator + rest
        return node

    def _named(self, nodes: Iterable[str]) -> dict[str, str | None]:
        """The entries by full id.

        A name with no file is read in each module and class of `nodes` that lies in a file of
        the fix.
        """
        holders = {
            node.rpartition("::")[0] for node in nodes if node.partition("::")[0] in self.scope
        }
        named = {old: new for old, new in self.ids if Remap._filed(old)}
        for holder in holders:
            named |= {
                f"{holder}::{old}": None if new is None else f"{holder}::{new}"
                for old, new in self.ids
                if not Remap._filed(old)
            }
        return named

    def apply(self, nodes: Iterable[str]) -> set[str]:
        """The ids `nodes` become: the named ones renamed or gone, then the moved files' ids."""
        return {self._moved(node) for node in self._after(nodes) if node is not None}

    def promised(self, nodes: Set[str]) -> set[str]:
        """The ids the change line gives to tests among `nodes`, where the fix left them."""
        return {self._moved(node) for node in self._after(nodes) - nodes if node is not None}
