# DuckDB relations, the one table type this tool hands around: what a trial writes as Parquet,
# what a results query answers and what a plot draws. One `Relations` is one in-memory database
# whose relations union and join with each other, and a relation keeps that database alive for as
# long as anyone holds it. Rows and Parquet bytes are read in whole, so a relation outlives the
# file it came through; `files` stays lazy, since the store's fragments outlive any read of them.
#
# Everything that opens Parquet files does so under the permitted file limit: matching columns by
# name opens every file at once, and a store of 3,625 fragments met macOS's default of 256.

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import CapsuleType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import duckdb

from ..runtime.tree import FileBudget
from .database import connect
from .lake import ndjson

if TYPE_CHECKING:
    from pydantic import JsonValue

type Relation = duckdb.DuckDBPyRelation


@runtime_checkable
class ArrowStream(Protocol):
    """Anything exporting an Arrow stream (a polars or pyarrow frame), which DuckDB reads
    through pyarrow, so the caller that holds one brings it."""

    def __arrow_c_stream__(self, requested_schema: CapsuleType | None = None) -> CapsuleType: ...


# The codec every Parquet file this tool writes uses, and its own default level.
_CODEC, LEVEL = "zstd", 3


class Relations:
    """One in-memory DuckDB database that the relations of one read share."""

    def __init__(self) -> None:
        self.connection = connect(config={"autoinstall_known_extensions": False})
        self._made = 0

    def rows(self, rows: Sequence[Mapping[str, JsonValue]]) -> Relation:
        """`rows` as one table, each column's type read from every row and a key a row lacks
        read as null.

        A key holding text stays text. Left to infer, DuckDB reads an instant, a date, a clock
        time or a UUID out of a string, and the value comes back as another object, an instant
        in the reader's time zone, rather than as written.
        """
        if not rows:
            raise ValueError("a table needs at least one row to read its columns from")
        texts = {key for row in rows for key, value in row.items() if isinstance(value, str)}
        with ndjson(rows, typed=True) as source:
            inferred = self.connection.sql(f"SELECT * FROM {source}")
            kinds = {
                name: "VARCHAR" if name in texts else str(kind)
                for name, kind in zip(inferred.columns, inferred.types, strict=True)
            }
        with ndjson(rows, typed=True, columns=kinds) as source:
            return self.kept(f"SELECT * FROM {source}")

    def arrow(self, table: ArrowStream) -> Relation:
        """An Arrow stream as one table, its column types kept."""
        self._made += 1
        name = f"t{self._made}"
        self.connection.from_arrow(table).to_table(name)
        return self.connection.table(name)

    def parquet(self, data: bytes) -> Relation:
        """Parquet bytes as one table."""
        descriptor, name = tempfile.mkstemp(prefix="mb-", suffix=".parquet")
        try:
            with os.fdopen(descriptor, "wb") as staged:
                staged.write(data)
            return self.kept("SELECT * FROM read_parquet(?)", [name])
        finally:
            Path(name).unlink(missing_ok=True)

    def files(self, paths: Sequence[Path], *, positions: bool = False) -> Relation:
        """Parquet files as one table, read lazily, their columns matched by name.

        positions: also carry each row's `filename` and `file_row_number`, the order it was
            written in.
        """
        located = ", filename = true, file_row_number = true" if positions else ""
        with FileBudget.permitted():
            return self.connection.sql(
                f"SELECT * FROM read_parquet(?, union_by_name = true, hive_partitioning = false"
                f"{located})",
                params=[[str(path) for path in paths]],
            )

    def sql(self, query: str, params: Sequence[object] = (), **named: Relation) -> Relation:
        """`query` over this database, each relation in `named` readable under its name."""
        for name, relation in named.items():
            relation.create_view(name, replace=True)
        return self.connection.sql(query, params=list(params) or None)

    def union(self, relations: Sequence[Relation], *, empty: str) -> Relation:
        """`relations`, built here, stacked with their columns matched by name.

        empty: the SELECT standing for no relation at all, naming the columns a reader expects.
        """
        if not relations:
            return self.connection.sql(empty)
        names = []
        for relation in relations:
            self._made += 1
            names.append(f"u{self._made}")
            relation.create_view(names[-1])
        return self.connection.sql(
            " UNION ALL BY NAME ".join(f"SELECT * FROM {name}" for name in names)
        )

    def kept(self, query: str, params: Sequence[object] = ()) -> Relation:
        """`query` materialized as a new table of this database."""
        self._made += 1
        name = f"t{self._made}"
        with FileBudget.permitted():
            self.connection.execute(f"CREATE TABLE {name} AS {query}", list(params) or None)
        return self.connection.table(name)


def write_parquet(relation: Relation, target: Path, *, level: int = LEVEL) -> None:
    """`relation`, from any database, as one zstd Parquet file at `target`."""
    quoted = str(target).replace("'", "''")
    with FileBudget.permitted():
        relation.query(
            "source",
            f"COPY source TO '{quoted}' (FORMAT parquet, COMPRESSION {_CODEC}, "
            f"COMPRESSION_LEVEL {level})",
        )


def parquet_bytes(relation: Relation, *, level: int = LEVEL) -> bytes:
    """`relation` as the bytes of one zstd Parquet file."""
    descriptor, name = tempfile.mkstemp(prefix="mb-", suffix=".parquet")
    os.close(descriptor)
    try:
        write_parquet(relation, Path(name), level=level)
        return Path(name).read_bytes()
    finally:
        Path(name).unlink(missing_ok=True)


def records(relation: Relation) -> list[dict[str, JsonValue]]:
    """`relation`'s rows as plain mappings, column name to value."""
    names = relation.columns
    with FileBudget.permitted():
        rows = relation.fetchall()
    return [dict(zip(names, row, strict=True)) for row in rows]
