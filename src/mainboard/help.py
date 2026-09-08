"""Search the CLI's live command descriptions, without a second manual."""

from collections.abc import Iterator
from inspect import cleandoc

from cyclopts import App

from .core.errors import MissionError
from .render import rows


class Help:
    """Read command help from the same application that parses its arguments."""

    def __init__(self, app: App) -> None:
        self.app = app

    def show(self, query: str = "") -> None:
        """Show an exact command's help, or search names and docstrings for all words."""
        words = query.split()
        if not words:
            self.app.help_print([])
            return
        entries = tuple(self._entries(self.app))
        query = " ".join(words)
        if any(command == query for command, _ in entries):
            self.app.help_print(words)
            return
        terms = [word.casefold().lstrip("-").replace("_", "-") for word in words]
        matches = []
        for command, description in entries:
            searchable = f"{command}\n{description}".casefold().replace("_", "-")
            if all(term in searchable for term in terms):
                excerpt = next(
                    (
                        line.strip()
                        for line in description.splitlines()
                        if any(term in line.casefold().replace("_", "-") for term in terms)
                    ),
                    description.split("\n", 1)[0],
                )
                matches.append({"command": command, "description": excerpt})
        if not matches:
            raise MissionError(f"no command help matches {query!r}")
        rows(matches, mode=None, fields=(), title="help")

    def _entries(self, app: App, prefix: str = "") -> Iterator[tuple[str, str]]:
        """Walk public commands, including nested groups, without running them."""
        for name in app:
            child = app[name]
            if name.startswith("-") or not child.show:
                continue
            command = f"{prefix} {name}".strip()
            yield command, cleandoc(child.help)
            yield from self._entries(child, command)
