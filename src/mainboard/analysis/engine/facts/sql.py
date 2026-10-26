# The SQL a rule runs: its text, its settings as parameters, and the views and macros it names.

import atexit
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import ClassVar

import duckdb
from patos import FrozenModel

from ....state.storage.parsed import names

_CAMEL_CASE_BOUNDARY_PATTERN = r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"

# Sequences and mappings, which accept a `list[str]` where a list of settings is read.
type Setting = str | int | float | bool | None | Sequence[Setting] | Mapping[str, Setting]


class SqlNames:
    """The views and the table macros each SQL text names: the views without the text's own common
    table expressions, the macros as `name()` in a `FROM`.

    Parsing is the cost, and the answer depends only on the text and DuckDB's version, so every
    text is parsed once per process and once per machine: the answers are kept in a file keyed by
    the text's digest, which a changed text simply misses.
    """

    FILE: ClassVar[Path] = (
        Path.home() / f".cache/mainboard/analysis/sql-names-duckdb-{duckdb.__version__}.json"
    )

    @classmethod
    def _read(cls) -> dict[str, list[list[str]]]:
        try:
            return json.loads(cls.FILE.read_text())
        except OSError, json.JSONDecodeError:
            return {}

    @staticmethod
    def _parse(sql: str) -> list[list[str]]:
        tables, macros = names(sql)
        return [sorted(table for table in tables if "." not in table), sorted(macros)]

    def __init__(self) -> None:
        self.stored = self._read()
        self.parsed: dict[str, list[list[str]]] = {}

    def of(self, sql: str) -> tuple[frozenset[str], frozenset[str]]:
        """The views and the macros `sql` names."""
        key = hashlib.blake2b(sql.encode(), digest_size=16).hexdigest()
        if (names := self.stored.get(key)) is None:
            if not self.parsed:
                atexit.register(self.save)
            names = self.stored[key] = self.parsed[key] = self._parse(sql)
        views, macros = names
        return frozenset(views), frozenset(macros)

    def save(self) -> None:
        """Add this process's parses to the file, whole or not at all (processes may share it)."""
        self.FILE.parent.mkdir(parents=True, exist_ok=True)
        merged = self._read() | self.parsed
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.FILE.parent, suffix=".tmp", delete=False
        ) as handle:
            json.dump(merged, handle)
        Path(handle.name).replace(self.FILE)


@cache
def _names(sql: str) -> tuple[frozenset[str], frozenset[str]]:
    return _NAMES.of(sql)


_NAMES = SqlNames()


def snake(name: str) -> str:
    """`ImportBindingFact` -> `import_binding`, `symbols.reference` -> `symbols_reference`."""
    words = re.sub(_CAMEL_CASE_BOUNDARY_PATTERN, "_", name)
    return words.removesuffix("_Fact").replace(".", "_").lower()


class Query(FrozenModel):
    """A rule's SQL over the fact views, its settings bound as `$name` parameters."""

    sql: str
    parameters: dict[str, Setting] = {}

    def macros(self) -> frozenset[str]:
        """The table macros the SQL calls, `name()` in a `FROM`."""
        return _names(self.sql)[1]

    def views(self) -> frozenset[str]:
        """The views the SQL names, its own common table expressions and macro calls aside."""
        return _names(self.sql)[0]
