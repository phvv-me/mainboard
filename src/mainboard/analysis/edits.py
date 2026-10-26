"""Edits a fix returns: plain values the engine expands, applies and verifies (AUTOFIX.md).

Span edits change bytes inside a file; file edits create, rename or remove whole files; an import
names what a file needs and the plan adds it once.
"""

from abc import ABC, abstractmethod
from typing import Literal

from patos import FrozenModel

from .findings import Span


class Edit(FrozenModel, ABC):
    """One change to source, by kind."""

    kind: str

    @abstractmethod
    def file(self) -> str:
        """The project path the edit changes."""


class Replace(Edit):
    kind: Literal["replace"] = "replace"
    span: Span
    text: str

    def file(self) -> str:
        return self.span.path


class Insert(Edit):
    kind: Literal["insert"] = "insert"
    text: str
    before: Span | None = None
    after: Span | None = None

    def anchor(self) -> Span:
        """The span the text goes before, else the one it goes after."""
        if anchor := self.before or self.after:
            return anchor
        raise ValueError("an insert needs a span to sit before or after")

    def file(self) -> str:
        return self.anchor().path


class Delete(Edit):
    kind: Literal["delete"] = "delete"
    span: Span

    def file(self) -> str:
        return self.span.path


class Move(Edit):
    kind: Literal["move"] = "move"
    span: Span
    before: Span | None = None
    after: Span | None = None

    def file(self) -> str:
        return self.span.path


def replace(span: Span, text: str) -> Replace:
    return Replace(span=span, text=text)


class Append(Edit):
    """Add `text` at the end of the file at `path`, creating it when missing.

    Appends to one file stack in the order the plan takes them, so several fixes may each add a
    definition to one shared module (a `constants.py`) in a single pass.
    """

    kind: Literal["append"] = "append"
    path: str
    text: str

    def file(self) -> str:
        return self.path


class Create(Edit):
    """Write a new file at `path` holding `text`, refused when something already stands there."""

    kind: Literal["create"] = "create"
    path: str
    text: str

    def file(self) -> str:
        return self.path


class Remove(Edit):
    """Delete a file or an empty directory."""

    kind: Literal["remove"] = "remove"
    path: str

    def file(self) -> str:
        return self.path


class Rename(Edit):
    """Move a file or directory to `target`, a path not yet taken."""

    kind: Literal["rename"] = "rename"
    path: str
    target: str

    def file(self) -> str:
        return self.path


def rename(path: str, *, target: str) -> Rename:
    return Rename(path=path, target=target)


class Import(Edit):
    """Make `statement` (`from functools import cache`) available in a Python file: added once
    after its imports, however many fixes ask, and not at all when the file already holds it. A
    statement that is no import (`logger = logging.getLogger(__name__)`) is added the same way,
    after the imports it may use."""

    kind: Literal["import"] = "import"
    path: str
    statement: str

    def file(self) -> str:
        return self.path
