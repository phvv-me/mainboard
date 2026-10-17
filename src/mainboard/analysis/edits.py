"""Edits a fix returns: plain values the engine expands, applies and verifies (AUTOFIX.md).

This first slice holds the span tier; the structural (`rewrite`) and semantic (`rename`,
`move_module`, `extract`) tiers follow on the same base.
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


def insert(text: str, *, before: Span | None = None, after: Span | None = None) -> Insert:
    return Insert(text=text, before=before, after=after)


def delete(span: Span) -> Delete:
    return Delete(span=span)


def move(span: Span, *, before: Span | None = None, after: Span | None = None) -> Move:
    return Move(span=span, before=before, after=after)


class Remove(Edit):
    """Delete a file or an empty directory."""

    kind: Literal["remove"] = "remove"
    path: str

    def file(self) -> str:
        return self.path


def remove(path: str) -> Remove:
    return Remove(path=path)


class Rename(Edit):
    """Move a file or directory to `target`, a path not yet taken."""

    kind: Literal["rename"] = "rename"
    path: str
    target: str

    def file(self) -> str:
        return self.path


def rename(path: str, target: str) -> Rename:
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


def require(path: str, statement: str) -> Import:
    return Import(path=path, statement=statement)
