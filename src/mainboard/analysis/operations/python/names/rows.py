# The columns of a `rewrite-references` fix row, and the records of the reads of a module's name.

import keyword
from typing import Literal, Self

from patos import FrozenModel
from pydantic import ConfigDict, field_validator, model_validator

from ....findings import Span

# One record of `name_readers`, as the engine hands a struct over.
type Record = dict[str, str | int | list[str]]


class Reading(FrozenModel):
    """One file's read of the name, as `name_readers` quotes it, with the columns of its kind.

    content: the source of the file.
    """

    model_config = ConfigDict(extra="allow")

    content: str
    path: str


class Importer(Reading):
    """One `from ... import` of the name; `start_line` is the line the statement opens on."""

    start_line: int


class Reader(Reading):
    """One read of the name through the module (`module.name`, `a.b.name`).

    end_column, end_line: where the read ends, which is where the name does.
    """

    end_column: int
    end_line: int

    def tail(self, name: str) -> Span:
        """The span of the attribute `name` that ends the read."""
        return Span(
            path=self.path,
            start_line=self.end_line,
            start_column=self.end_column - len(name.encode()),
            end_line=self.end_line,
            end_column=self.end_column,
        )


class RenameRow(FrozenModel):
    """The columns of a `rewrite-references` fix row; the others the engine adds are left alone.

    content: the source of the file at `path`, which binds `name`.
    dotted: each `a.b.name` read, where the module is spelled in full.
    importers: each `from ... import` of `name`.
    line: the line the function binding `name` opens on (function scope).
    module: the dotted name of the module at `path` (module scope).
    reached: each `module.name` read through a name bound to the module.
    scanned: each Python file that imports the module or spells `module.name` in its text. The
        rename rewrites none the other columns do not list, so none of those may spell `name`.
    starred: whether some module imports every name of the module.
    to: the name `name` is given.
    """

    model_config = ConfigDict(extra="ignore")

    content: str
    dotted: tuple[Reader, ...] = ()
    importers: tuple[Importer, ...] = ()
    line: int | None = None
    module: str = ""
    name: str
    path: str
    reached: tuple[Reader, ...] = ()
    scanned: tuple[Reading, ...] = ()
    scope: Literal["function", "module"]
    starred: bool = False
    to: str

    @field_validator("dotted", "importers", "reached", "scanned", mode="before")
    @classmethod
    def _listed(cls, value: list[Record] | None) -> list[Record]:
        """A list SQL aggregated over no row is NULL; read it as empty."""
        return value or []

    @model_validator(mode="after")
    def _renaming(self) -> Self:
        """A rename gives a new identifier, and a function scope names the function."""
        if not self.to.isidentifier() or keyword.iskeyword(self.to) or self.to == self.name:
            raise ValueError(f"`{self.to}` is no new name for `{self.name}`")
        if self.scope == "function" and not self.line:
            raise ValueError("a function scope needs the line the function opens on")
        return self
