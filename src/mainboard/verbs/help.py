"""Search live commands and shipped documentation without importing API modules."""

import ast
from collections.abc import Iterable, Iterator, Sequence
from inspect import cleandoc
from pathlib import Path
from textwrap import shorten
from types import FunctionType

from cyclopts import App
from rich.console import Console

from ..core.errors import MissionError
from ..log import logger


class Help:
    """Read the command registry, canonical README, and Python source docstrings."""

    @staticmethod
    def _listed(matches: Sequence[tuple[str, str, str]]) -> str:
        """The first twenty matches as the terminal shows them, and how many more there are."""
        console = Console(markup=False)
        with console.capture() as captured:
            for name, location, excerpt in matches[:20]:
                console.print(f"{name}\n  {location}\n  {excerpt}\n", soft_wrap=True)
            if len(matches) > 20:
                console.print(
                    f"Showing 20 of {len(matches)} matching locations; add search words."
                )
        return captured.get()

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
        """Search prose paragraphs and individual code/table lines in the canonical README."""
        path = self.package / "README.md"
        if not path.is_file():
            path = self.package.parent.parent / "README.md"
        return _Readme(path).pieces()

    def _entries(self, app: App, prefix: str = "") -> Iterator[tuple[str, str, str]]:
        """Walk public commands, including nested groups, without running them."""
        for name in app:
            child = app[name]
            if name.startswith("-") or not child.show:
                continue
            command = f"{prefix} {name}".strip()
            location = f"mainboard help {command}"
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

    def show(self, query: str = "") -> None:
        """Exact commands keep their help; other queries search all three source surfaces."""
        words = query.split()
        if not words:
            self.app.help_print([])
            return
        entries = list(self._entries(self.app))
        if any(command == " ".join(words) for command, _, _ in entries):
            self.app.help_print(words)
            return
        if not (matches := self._matches(entries, words)):
            raise MissionError(
                f"no help matches {' '.join(words)!r} in commands, documentation, or Python"
            )
        logger.result(Help._listed(matches), end="")


class _Readme:
    """The searchable pieces of one README: its prose paragraphs, and its code and table lines
    one by one, each under the heading it sits beneath."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.heading = path.name
        self.fence = ""
        self.paragraph: list[str] = []
        self.start = 1

    def _closed(self, *, heading: str) -> Iterator[tuple[str, str, str]]:
        """The paragraph read so far, ended, before `heading` takes over."""
        if self.paragraph:
            yield f"docs {self.heading}", "\n".join(self.paragraph), f"{self.path}:{self.start}"
            self.paragraph = []
        self.heading = heading

    def _kept(self, number: int, line: str, *, separate: bool) -> Iterator[tuple[str, str, str]]:
        """A code or table line on its own, or a prose line added to its paragraph."""
        if separate:
            yield f"docs {self.heading}", line, f"{self.path}:{number}"
            return
        if not self.paragraph:
            self.start = number
        self.paragraph.append(line)

    def _read(self, number: int, line: str) -> Iterator[tuple[str, str, str]]:
        """What line `number` ends or adds."""
        marker = line.lstrip()[:3]
        fence_line = marker in ("```", "~~~")
        heading_line = not self.fence and line.startswith("#")
        separate = bool(self.fence) or line.startswith("|")
        if not line.strip() or heading_line or fence_line or separate:
            yield from self._closed(heading=line.lstrip("# ") if heading_line else self.heading)
        if fence_line:
            self.fence = "" if marker == self.fence else (self.fence or marker)
        elif line.strip():
            yield from self._kept(number, line, separate=separate)

    def pieces(self) -> Iterator[tuple[str, str, str]]:
        """Each piece as (where, text, `path:line`)."""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate([*lines, ""], start=1):
            yield from self._read(number, line)
