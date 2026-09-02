"""Resumable, concurrency-safe measurement rows: one immutable part per writer."""

import logging
import os
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import polars as pl

logger = logging.getLogger("mainboard.experiments")

# The knee of the size against speed curve for zstd on tabular rows.
_ZSTD_LEVEL = 9


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


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
        writer = _slug(f"{socket.gethostname()}-{os.getpid()}")
        self.part = self.dir / f"part-{writer}.parquet"
        self.rows: list[dict] = self._read(self.part)
        self._keys: set[tuple[str, ...]] = {self._key(row) for row in self._load_all()}

    def has(self, **ids: object) -> bool:
        """Whether a row with this identity already exists in any part."""
        return tuple(str(ids[field]) for field in self.id_fields) in self._keys

    def append(self, row: Mapping[str, object]) -> bool:
        """Append one row unless its identity is already recorded; return whether it was taken."""
        key = self._key(row)
        if key in self._keys:
            return False
        self.rows.append(dict(row))
        self._keys.add(key)
        return True

    def extend(self, rows: Sequence[Mapping[str, object]]) -> int:
        """Append many rows, returning how many were new."""
        return sum(self.append(row) for row in rows)

    def drop_where(self, predicate: Callable[[dict], bool]) -> None:
        """Remove this writer's rows satisfying `predicate`."""
        dropped = {self._key(row) for row in self.rows if predicate(row)}
        self.rows = [row for row in self.rows if not predicate(row)]
        self._keys -= dropped

    def flush(self) -> None:
        """Atomically rewrite this writer's part from its in-memory rows."""
        if not self.rows:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(self.rows, infer_schema_length=None)
        temporary = self.part.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd", compression_level=_ZSTD_LEVEL)
        temporary.replace(self.part)
        logger.info("wrote %d rows to %s", len(self.rows), self.part)

    def __len__(self) -> int:
        return len(self._keys)

    def _key(self, row: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in self.id_fields)

    def _read(self, file: Path) -> list[dict]:
        if not file.exists():
            return []
        return pl.read_parquet(file).to_dicts()

    def _load_all(self) -> list[dict]:
        rows: list[dict] = []
        for part in sorted(self.dir.glob("part-*.parquet")):
            rows += self._read(part)
        return rows
