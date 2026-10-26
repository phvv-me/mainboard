"""Resumable, concurrency-safe measurement rows: one immutable part per writer."""

import os
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..log import logger
from ..state.relations import Relations, records, write_parquet

if TYPE_CHECKING:
    from pydantic import JsonValue

# The knee of the size against speed curve for zstd on tabular rows.
_ZSTD_LEVEL = 9


class RowLog:
    """Rows keyed by identity, appended by one writer, federated across every writer on resume.

    Each process writes its own `part-<host>-<pid>.parquet` inside the directory named for the
    table, so parallel jobs never rewrite a shared file. Resume reads the keys of every part,
    and an append whose key already exists anywhere is ignored, so a killed sweep replayed after
    a partial cell cannot double count. Writes are atomic through a temporary file and rename.

    path: the logical table path; parts live in that directory with any table suffix dropped.
    id_fields: the columns whose values together identify one row.
    """

    def __init__(self, path: Path, id_fields: Sequence[str]) -> None:
        self.dir = path if path.suffix == "" else path.with_suffix("")
        self.id_fields = tuple(id_fields)
        writer = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{socket.gethostname()}-{os.getpid()}").strip(
            "-"
        )
        self.part = self.dir / f"part-{writer}.parquet"
        self.rows: list[dict] = self._read(self.part)
        self._keys: set[tuple[str, ...]] = {
            self._key(row)
            for part in sorted(self.dir.glob("part-*.parquet"))
            for row in self._read(part)
        }

    def __len__(self) -> int:
        return len(self._keys)

    def _key(self, row: Mapping[str, JsonValue]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in self.id_fields)

    def _read(self, file: Path) -> list[dict]:
        return records(Relations().files([file])) if file.exists() else []

    def append(self, row: Mapping[str, JsonValue]) -> bool:
        """Append one row unless its identity is already recorded; return whether it was taken."""
        key = self._key(row)
        if key in self._keys:
            return False
        self.rows.append(dict(row))
        self._keys.add(key)
        return True

    def drop_where(self, predicate: Callable[[dict], bool]) -> None:
        """Remove this writer's rows satisfying `predicate`."""
        dropped = {self._key(row) for row in self.rows if predicate(row)}
        self.rows = [row for row in self.rows if not predicate(row)]
        self._keys -= dropped

    def extend(self, rows: Sequence[Mapping[str, JsonValue]]) -> int:
        """Append many rows, returning how many were new."""
        return sum(self.append(row) for row in rows)

    def flush(self) -> None:
        """Atomically rewrite this writer's part from its in-memory rows."""
        if not self.rows:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        temporary = self.part.with_suffix(".parquet.tmp")
        write_parquet(Relations().rows(self.rows), temporary, level=_ZSTD_LEVEL)
        temporary.replace(self.part)
        logger.info("wrote {} rows to {}", len(self.rows), self.part)

    def has(self, **ids: JsonValue) -> bool:
        """Whether a row with this identity already exists in any part."""
        return self._key(ids) in self._keys
