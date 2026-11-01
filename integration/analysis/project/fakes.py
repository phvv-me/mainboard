from collections.abc import Iterable
from pathlib import Path

from mainboard.analysis.project.consumers import Consumers


class Importers(Consumers):
    """The rest of the workspace: one file imports each of `names`, and nothing else is used."""

    def __init__(self, names: Iterable[str]) -> None:
        super().__init__(Path())
        self.names = frozenset(names)

    def used(self, rule: str, *, path: str, declared: str) -> str | None:
        return "consumer.py" if declared in self.names else None
