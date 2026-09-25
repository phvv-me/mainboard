"""Search live commands and shipped documentation without importing API modules."""

import ast
from collections.abc import Iterator
from inspect import cleandoc
from pathlib import Path
from textwrap import shorten
from types import FunctionType

from cyclopts import App
from rich.console import Console

from .core.errors import MissionError


class Help:
    """Read the command registry, canonical README, and Python source docstrings."""

    def __init__(self, app: App) -> None:
        self.app = app
        self.package = Path(__file__).resolve().parent

    def _definitions(
        self, node: ast.Module | ast.ClassDef, prefix: str
    ) -> Iterator[tuple[str, str, int]]:
        """Index declarations, not function bodies, imports, or executed descriptors."""
        if description := ast.get_docstring(node):
            yield prefix, description, node.body[0].lineno
        for child in node.body:
            if not isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if child.name.startswith("_") and child.name != "__init__":
                continue
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
        lines = path.read_text(encoding="utf-8").splitlines()
        paragraph: list[str] = []
        heading, start, fence = path.name, 1, ""
        for number, line in enumerate([*lines, ""], start=1):
            marker = line.lstrip()[:3]
            fence_line = marker in ("```", "~~~")
            heading_line = not fence and line.startswith("#")
            separate = bool(fence) or line.startswith("|")
            if not line.strip() or heading_line or fence_line or separate:
                if paragraph:
                    yield f"docs {heading}", "\n".join(paragraph), f"{path}:{start}"
                    paragraph = []
                if heading_line:
                    heading = line.lstrip("# ")
            if fence_line:
                fence = "" if marker == fence else (fence or marker)
                continue
            if separate and line.strip():
                yield f"docs {heading}", line, f"{path}:{number}"
                continue
            if line.strip():
                if not paragraph:
                    start = number
                paragraph.append(line)

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

    def _excerpt(self, description: str, terms: list[str]) -> str:
        """Prefer the short line containing the most query terms."""
        lines = [line.strip() for line in description.splitlines() if line.strip()]
        line = max(
            lines,
            key=lambda text: sum(term in text.casefold().replace("_", "-") for term in terms),
            default="",
        )
        return shorten(line, width=220, placeholder=" …")

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
        entries = tuple(self._entries(self.app))
        query = " ".join(words)
        if any(command == query for command, _, _ in entries):
            self.app.help_print(words)
            return
        terms = [word.casefold().lstrip("-").replace("_", "-") for word in words]
        candidates = (
            *((f"command {name}", doc, location) for name, doc, location in entries),
            *self._docs(),
            *self._python(),
        )
        matches = [
            (name, location, self._excerpt(description, terms))
            for name, description, location in candidates
            if all(term in f"{name}\n{description}".casefold().replace("_", "-") for term in terms)
        ]
        if not matches:
            raise MissionError(f"no help matches {query!r} in commands, documentation, or Python")
        console = Console(markup=False)
        for name, location, excerpt in matches[:20]:
            console.print(f"{name}\n  {location}\n  {excerpt}\n", soft_wrap=True)
        if len(matches) > 20:
            console.print(f"Showing 20 of {len(matches)} matching locations; add search words.")
