"""Search the commands, the README and the Python docstrings this tool ships, importing none of it.

It reads the command registry, the canonical README and the shipped source as text, so a search
never imports an API module, and it depends on no other part of the tool.
"""

import ast
from collections.abc import Iterable, Iterator, Sequence
from inspect import cleandoc
from pathlib import Path
from textwrap import shorten
from types import FunctionType

from cyclopts import App
from markdown_it import MarkdownIt
from markdown_it.token import Token

# How many matches a search shows before it only counts the rest.
_SHOWN = 20


class Help:
    """Read the command registry, canonical README, and Python source docstrings."""

    @staticmethod
    def _blocks(lines: list[str]) -> Iterator[tuple[Token, int, list[str]]]:
        """Each top-level block of Markdown `lines` with its first line index and its lines."""
        for block in MarkdownIt().enable("table").parse("\n".join(lines)):
            if not block.level and block.map:
                first, last = block.map
                yield block, first, lines[first:last]

    @staticmethod
    def _listed(matches: Sequence[tuple[str, str, str]]) -> str:
        """The first twenty matches as lines to print, and how many more there are."""
        listed = [f"{name}\n  {location}\n  {excerpt}\n\n" for name, location, excerpt in matches]
        more = f"Showing {_SHOWN} of {len(matches)} matching locations; add search words.\n"
        return "".join(listed[:_SHOWN]) + (more if len(matches) > _SHOWN else "")

    @staticmethod
    def _rows(block: Token, first: int, lines: list[str]) -> list[tuple[int, str]]:
        """The non-blank (line number, text) pieces of a block: code and table lines apart."""
        match block.type:
            case "fence":
                rows = enumerate(block.content.splitlines(), first + 2)
            case "table_open":
                rows = enumerate(lines, first + 1)
            case _:
                rows = enumerate(["\n".join(lines).rstrip()], first + 1)
        return [(number, text) for number, text in rows if text.strip()]

    def __init__(self, app: App) -> None:
        self.app = app
        self.package = Path(__file__).resolve().parents[1]

    def _definitions(
        self, node: ast.Module | ast.ClassDef, prefix: str
    ) -> Iterator[tuple[str, str, int]]:
        """Index declarations, not function bodies, imports, or executed descriptors."""
        if description := ast.get_docstring(node):
            yield prefix, description, node.body[0].lineno
        declared = (
            child
            for child in node.body
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and (not child.name.startswith("_") or child.name == "__init__")
        )
        for child in declared:
            name = f"{prefix}.{child.name}"
            if isinstance(child, ast.ClassDef):
                yield from self._definitions(child, name)
            elif description := ast.get_docstring(child):
                yield name, description, child.lineno

    def _docs(self) -> Iterator[tuple[str, str, str]]:
        """Search prose blocks and individual code/table lines in the canonical README.

        Each piece sits under the heading above it as (where, text, `path:line`).
        """
        path = self.package / "README.md"
        if not path.is_file():
            path = self.package.parent.parent / "README.md"
        heading = path.name
        for block, first, lines in Help._blocks(path.read_text(encoding="utf-8").splitlines()):
            if block.type == "heading_open":
                heading = lines[0].lstrip("# ")
            for number, text in Help._rows(block, first, lines):
                yield f"docs {heading}", text, f"{path}:{number}"

    def _entries(self, app: App, prefix: str = "") -> Iterator[tuple[str, str, str]]:
        """Walk public commands, including nested groups, without running them."""
        for name in app:
            child = app[name]
            if name.startswith("-") or not child.show:
                continue
            command = f"{prefix} {name}".strip()
            location = f"mb --help {command}"
            if isinstance(callback := child.default_command, FunctionType):
                location = f"{callback.__code__.co_filename}:{callback.__code__.co_firstlineno}"
            yield command, cleandoc(child.help), location
            yield from self._entries(child, command)

    def _excerpt(self, description: str, terms: Iterable[str]) -> str:
        """Prefer the short line containing the most query terms."""
        lines = [line.strip() for line in description.splitlines() if line.strip()]
        line = max(
            lines,
            key=lambda text: sum(term in text.casefold().replace("_", "-") for term in terms),
            default="",
        )
        return shorten(line, width=220, placeholder=" …")

    def _matches(
        self, entries: Sequence[tuple[str, str, str]], words: Sequence[str]
    ) -> list[tuple[str, str, str]]:
        """Every command, documentation piece and Python declaration holding all of `words`."""
        terms = [word.casefold().lstrip("-").replace("_", "-") for word in words]
        candidates = (
            *((f"command {name}", doc, location) for name, doc, location in entries),
            *self._docs(),
            *self._python(),
        )
        return [
            (name, location, self._excerpt(description, terms))
            for name, description, location in candidates
            if all(term in f"{name}\n{description}".casefold().replace("_", "-") for term in terms)
        ]

    def _python(self) -> Iterator[tuple[str, str, str]]:
        """Parse the shipped source; optional dependencies and GPU drivers stay unloaded."""
        for path in sorted(self.package.rglob("*.py")):
            parts = path.relative_to(self.package).with_suffix("").parts
            if any(part.startswith("_") and part != "__init__" for part in parts):
                continue
            module = ".".join((self.package.name, *(p for p in parts if p != "__init__")))
            source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name, description, line in self._definitions(source, module):
                yield f"python {name}", description, f"{path}:{line}"

    def show(self, query: str = "") -> str:
        """Print an exact command's help and answer nothing; answer what any other query found.

        Raises LookupError when nothing holds every word of `query`.
        """
        words = query.split()
        if not words:
            self.app.help_print([])
            return ""
        entries = list(self._entries(self.app))
        if any(command == " ".join(words) for command, _, _ in entries):
            self.app.help_print(words)
            return ""
        if not (matches := self._matches(entries, words)):
            raise LookupError(
                f"no help matches {' '.join(words)!r} in commands, documentation, or Python"
            )
        return Help._listed(matches)
