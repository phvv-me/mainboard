# The bookkeeping of a fixpoint run: the fixes applied and the files they changed together.

from collections.abc import Iterable, Set

from patos import FrozenModel


class Applied(FrozenModel):
    """A fix applied in the mirror: where its finding stood in the file as that pass began, the
    rule's words for what it changed, and the files it changed."""

    rule: str
    path: str
    line: int
    change: str
    files: frozenset[str]

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
