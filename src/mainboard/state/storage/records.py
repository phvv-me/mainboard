"""Records staged as one NDJSON file that DuckDB reads itself."""

import json
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path

from .database import quoted, scratch

# DuckDB's own largest JSON object by default, the floor of a staged file's object limit. A
# reader buffers twice its limit, so the limit follows the longest staged line: a fixed 2 GiB,
# set so a 25 MB source file could be sealed (2026-09-29), reserved 4 GiB for every append and
# ran the 7 GB macOS CI runner out of memory (2026-10-07).
_JSON_OBJECT_BYTES = 16 << 20


@contextmanager
def ndjson(
    records: Iterable[Mapping[str, object] | str],
    *,
    typed: bool = False,
    columns: Mapping[str, str] | None = None,
) -> Generator[str]:
    """`records` as a temporary NDJSON file, answered as the call reading it.

    Records are mappings or lines already JSON, and the object limit is the longest line's. Each
    record is one `json` object, or with `typed` the columns `read_json` infers from every record,
    a key one lacks read as null, or with `columns` exactly those, each name to its DuckDB type.

    DuckDB reads a file like this in a fraction of a second, while this build binds a list
    parameter at two milliseconds an element: twenty thousand rows took forty seconds.
    """
    with scratch(".ndjson") as staging:
        longest = _written(records, staging)
        path, limit = quoted(staging.as_posix()), max(_JSON_OBJECT_BYTES, longest + 1)
        kinds = ", ".join(
            f"{quoted(column)}: {quoted(kind)}" for column, kind in (columns or {}).items()
        )
        pinned = f", columns = {{{kinds}}}" if columns else ""
        yield (
            f"read_json({path}, format = 'newline_delimited', sample_size = -1, "
            f"maximum_object_size = {limit}{pinned})"
            if typed
            else f"read_ndjson_objects({path}, maximum_object_size = {limit})"
        )


def _written(records: Iterable[Mapping[str, object] | str], staging: Path) -> int:
    """Write `records` one JSON line each to `staging`, answering the longest line's bytes."""
    longest = 0
    with staging.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            line = record if isinstance(record, str) else json.dumps(record, default=str)
            longest = max(longest, len(line.encode()))
            out.write(line + "\n")
    return longest
