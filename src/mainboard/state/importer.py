# The one-time import of a workspace's file state into its lake.
#
# Everything the state directory keeps as a file of records is read and appended to the lake in
# one transaction, so an import either lands whole or not at all, then read back and compared
# with the files: a row count per source and, for the batch logs, a SHA-256 over every byte
# rebuilt from the lake against the files themselves. The old files are only ever read, never
# moved, rewritten or deleted, and writers keep using them until they are switched over.
#
# A line that does not parse is not dropped: it lands in `strays` with where it came from, so
# the count still adds up and nothing a torn write left behind is lost. What is not state is left
# out on purpose: wandb run folders, pins, source archives, recovery and environments.
#
# The old `collection.digests.json` holds two entry shapes, the stamp an older release wrote
# (`[size, mtime_ns, sha256]`) beside today's (`[size, inode, mtime_ns, ctime_ns, sha256]`), and
# both map onto the one typed `digests` table.

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
from patos import FrozenModel

from ..core.errors import MissionError
from .lake import ALIAS, Lake, insert

# How many log bytes are staged before they are appended, bounding memory on a large workspace.
_CHUNK_BYTES = 64 << 20

# How many rows a verification read pulls from the lake at a time.
_FETCH = 100_000

# The two digest memories and the kind each is recorded under.
_DIGESTS = (("dispatch/digests.json", "mirror"), ("collection.digests.json", "collection"))

type Row = dict[str, object]

# How one parsed JSON line of a file becomes a row.
type _Shape = Callable[[Path, dict[str, object]], Row]


class Tally(FrozenModel):
    """One source compared with what the lake holds of it.

    source: the file or glob read, relative to the state directory.
    table: the lake table it went to.
    expected: records in the source (non-blank lines, rows, entries or files).
    imported: rows the lake holds of it, read back after the commit.
    strays: records that did not parse and were kept in `strays` instead.
    ok: whether imported and strays add up to expected, and any digest matched.
    detail: what else was compared, or what differed.
    """

    source: str
    table: str
    expected: int
    imported: int
    strays: int = 0
    ok: bool
    detail: str = ""


class _Load:
    """What one source contributed: its rows, its strays and how to count it in the lake."""

    def __init__(self, source: str, table: str, where: str = "true") -> None:
        self.source = source
        self.table = table
        self.where = where
        self.expected = 0
        self.rows: list[Row] = []
        self.strays: list[Row] = []

    def stray(self, path: str, number: int, line: str) -> None:
        self.strays.append({"source": path, "destination": self.table, "n": number, "line": line})


class _Logs(FrozenModel):
    """What the batch logs held: files, lines, lossy lines and the digest of the exact ones."""

    files: int
    lines: int
    lossy: tuple[str, ...]
    lossy_lines: int
    sha256: str


class Importer:
    """Imports one workspace's file state into its lake, once."""

    def __init__(self, lake: Lake) -> None:
        self.lake = lake
        self.out = lake.out
        self.instant = datetime.now(UTC)

    def run(self, *, again: bool = False) -> list[Tally]:
        """Import every source in one transaction and verify it, returning one tally per source.

        again: import into a fresh lake when this one already holds an import, the old lake set
            aside under `lake.aside/` rather than deleted.
        Raises MissionError when the lake already holds an import and `again` is not set.
        """
        if self.lake.exists():
            done = self.lake.query(f"SELECT count(*) AS n FROM {ALIAS}.imports")["n"][0]
            if done and not again:
                raise MissionError(
                    f"{self.lake.catalog} already holds an import; pass --again to set that "
                    "lake aside and import anew"
                )
            if done:
                _set_aside(self.lake)
        if not self.lake.exists():
            self.lake.create()
        loads: list[_Load] = []
        with self.lake.open(write=True) as connection:
            connection.execute("BEGIN")
            for load in self._loads():
                insert(connection, load.table, load.rows)
                insert(connection, "strays", self._stamped(load.strays))
                load.rows = []
                loads.append(load)
            logs = self._logs(connection)
            insert(
                connection,
                "imports",
                [
                    {
                        "ts": self.instant,
                        "source": load.source,
                        "destination": load.table,
                        "rows": load.expected,
                    }
                    for load in loads
                ]
                + [
                    {
                        "ts": self.instant,
                        "source": "batches/**/*.log",
                        "destination": "log_lines",
                        "rows": logs.lines,
                    }
                ],
            )
            connection.execute("COMMIT")
        return self._verified(loads, logs)

    def _loads(self) -> Iterator[_Load]:
        """Every record source but the logs, read one at a time."""
        yield from self._registry()
        yield self._lines("batches/*/events.ndjson", "events", self._event)
        yield self._receipts()
        yield self._lines("costs/*.ndjson", "costs", self._cost)
        yield self._lines("catalog.ndjson", "quotes", self._quote)
        yield self._holds()
        yield self._lines("studies/*.jsonl", "studies", self._study)
        yield self._pulse()
        for name, kind in _DIGESTS:
            yield self._digests(name, kind)
        yield self._scripts()
        yield self._closures()

    def _registry(self) -> Iterator[_Load]:
        """The dispatch registry's `runs` and `hosts` rows, their JSON blobs kept whole."""
        runs = _Load("dispatch/db.sqlite runs", "runs_log")
        hosts = _Load("dispatch/db.sqlite hosts", "host_facts")
        found, onboarded = _registered(self.out / "dispatch" / "db.sqlite")
        for number, (target, handle, data, submitted) in enumerate(found, 1):
            runs.expected += 1
            if not _parses(data):
                runs.stray("dispatch/db.sqlite", number, data)
                continue
            runs.rows.append(
                {
                    "ts": self.instant,
                    "target": target,
                    "handle": handle,
                    "submitted_at": submitted,
                    "record": data,
                    "dropped": False,
                }
            )
        for number, (alias, facts, probed) in enumerate(onboarded, 1):
            hosts.expected += 1
            if not _parses(facts):
                hosts.stray("dispatch/db.sqlite", number, facts)
                continue
            hosts.rows.append(
                {
                    "ts": self.instant,
                    "alias": alias,
                    "probed_at": probed,
                    "facts": facts,
                    "dropped": False,
                }
            )
        yield runs
        yield hosts

    def _lines(self, pattern: str, table: str, shape: _Shape) -> _Load:
        """Every non-blank JSON line of the files `pattern` matches, shaped into `table` rows."""
        load = _Load(pattern, table)
        for path in sorted(self.out.glob(pattern)):
            relative = path.relative_to(self.out).as_posix()
            for number, line in enumerate(_split(path), 1):
                if not line.strip():
                    continue
                load.expected += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                if isinstance(record, dict):
                    load.rows.append(shape(path, record))
                else:
                    load.stray(relative, number, line)
        return load

    def _event(self, path: Path, record: dict[str, object]) -> Row:
        """A batch event, its `at` renamed `ts`, a reserved word in DuckDB 2.0."""
        return {
            "ts": _instant(record.get("at")),
            "batch": record.get("batch", path.parent.name),
            "topic": record.get("topic"),
            "job": record.get("job"),
            "data": record.get("data"),
        }

    def _cost(self, path: Path, record: dict[str, object]) -> Row:
        return record

    def _quote(self, path: Path, record: dict[str, object]) -> Row:
        return {"ts": self.instant, **record}

    def _study(self, path: Path, record: dict[str, object]) -> Row:
        """A study ledger line, its study named by its file and `at` renamed `ts`."""
        fields = {key: value for key, value in record.items() if key != "at"}
        return {**fields, "ts": _instant(record.get("at")), "study": path.stem}

    def _receipts(self) -> _Load:
        """Every trial receipt line, kept verbatim beside the fields a query filters on.

        The monitor deduplicates receipts by their exact text, so the line itself is the
        record and a line that does not parse still lands, with its fields empty.
        """
        load = _Load("batches/*/receipts.ndjson", "receipts")
        for path in sorted(self.out.glob(load.source)):
            for number, line in enumerate(_split(path), 1):
                if not line.strip():
                    continue
                load.expected += 1
                record = _document(line)
                found = record.get("trial_receipt") if isinstance(record, dict) else None
                receipt = found if isinstance(found, dict) else {}
                load.rows.append(
                    {
                        "ts": _instant(receipt.get("at")),
                        "batch": path.parent.name,
                        "n": number,
                        "run": receipt.get("run"),
                        "trial": receipt.get("trial"),
                        "verdict": receipt.get("verdict"),
                        "host": receipt.get("host"),
                        "line": line,
                    }
                )
        return load

    def _holds(self) -> _Load:
        """The held machines, one row per alias."""
        load = _Load("dispatch/holds.json", "holds_log")
        path = self.out / load.source
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            held = _document(text)
            if isinstance(held, list) and all(isinstance(entry, dict) for entry in held):
                load.expected = len(held)
                load.rows = [
                    {
                        "ts": self.instant,
                        "alias": entry.get("alias"),
                        "held": entry,
                        "dropped": False,
                    }
                    for entry in held
                ]
            else:
                load.expected = 1
                load.stray(load.source, 0, text)
        return load

    def _pulse(self) -> _Load:
        """The pulse memory, one row per remembered job: its output length and last growth."""
        load = _Load("pulse.json", "pulse")
        path = self.out / load.source
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            held = _document(text)
            if not isinstance(held, dict):
                load.expected = 1
                load.stray(load.source, 0, text)
                return load
            for number, (key, kept) in enumerate(held.items(), 1):
                load.expected += 1
                if isinstance(kept, list) and len(kept) == 2:
                    load.rows.append(
                        {"ts": self.instant, "key": key, "size": kept[0], "grew": kept[1]}
                    )
                else:
                    load.stray(load.source, number, json.dumps({key: kept}))
        return load

    def _digests(self, name: str, kind: str) -> _Load:
        """One digest memory, both entry shapes typed into columns."""
        load = _Load(name, "digests", where=f"kind = '{kind}'")
        path = self.out / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            held = _document(text)
            if not isinstance(held, dict):
                load.expected = 1
                load.stray(name, 0, text)
                return load
            for number, (key, entry) in enumerate(held.items(), 1):
                load.expected += 1
                row = _digest(entry)
                if row is None:
                    load.stray(name, number, json.dumps({key: entry}))
                else:
                    load.rows.append({"ts": self.instant, "kind": kind, "path": key, **row})
        return load

    def _scripts(self) -> _Load:
        """Every generated job script, its text and the SHA-256 of its bytes."""
        load = _Load("dispatch/jobs/*.sh", "job_specs")
        for path in sorted(self.out.glob(load.source)):
            raw = path.read_bytes()
            load.expected += 1
            load.rows.append(
                {
                    "ts": self.instant,
                    "name": path.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "script": raw.decode("utf-8", "replace"),
                }
            )
        return load

    def _closures(self) -> _Load:
        """Every closure listing's rows: path, blob and status, tab-separated."""
        load = _Load("dispatch/jobs/closure-*.tsv", "closures")
        for path in sorted(self.out.glob(load.source)):
            relative = path.relative_to(self.out).as_posix()
            closure = path.stem.removeprefix("closure-")
            for number, line in enumerate(_split(path), 1):
                if not line.strip():
                    continue
                load.expected += 1
                fields = line.split("\t")
                if len(fields) == 3:
                    load.rows.append(
                        {
                            "ts": self.instant,
                            "closure": closure,
                            "path": fields[0],
                            "blob": fields[1],
                            "status": fields[2],
                        }
                    )
                else:
                    load.stray(relative, number, line)
        return load

    def _logs(self, connection: duckdb.DuckDBPyConnection) -> _Logs:
        """Append every batch log line by line, returning what the files held.

        A line is decoded as UTF-8; one that is not keeps the replacement character and is
        marked lossy. Lines split on `\\n` alone, so a `\\r` stays on its line and joining the
        lines with `\\n` gives back the file's bytes exactly. Symbolic links and wandb folders
        are not logs of this workspace's own.
        """
        hasher = hashlib.sha256()
        lossy: list[str] = []
        staged: list[Row] = []
        size = files = lines = lossy_lines = 0
        for path in _logfiles(self.out / "batches"):
            relative = path.relative_to(self.out / "batches").as_posix()
            raw = path.read_bytes()
            files += 1
            exact = True
            for number, piece in enumerate(raw.split(b"\n"), 1):
                try:
                    line, damaged = piece.decode("utf-8"), False
                except UnicodeDecodeError:
                    line, damaged = piece.decode("utf-8", "replace"), True
                exact = exact and not damaged
                lossy_lines += damaged
                lines += 1
                staged.append(
                    {
                        "ts": self.instant,
                        "batch": relative.split("/")[0],
                        "file": relative,
                        "n": number,
                        "line": line,
                        "lossy": damaged,
                    }
                )
            size += len(raw)
            if exact:
                hasher.update(raw)
            else:
                lossy.append(relative)
            if size >= _CHUNK_BYTES:
                insert(connection, "log_lines", staged)
                staged, size = [], 0
        insert(connection, "log_lines", staged)
        return _Logs(
            files=files,
            lines=lines,
            lossy=tuple(lossy),
            lossy_lines=lossy_lines,
            sha256=hasher.hexdigest(),
        )

    def _stamped(self, rows: Sequence[Row]) -> list[Row]:
        return [{"ts": self.instant, **row} for row in rows]

    def _verified(self, loads: Sequence[_Load], logs: _Logs) -> list[Tally]:
        """Every source's count read back from the lake, and the logs rebuilt byte for byte."""
        tallies: list[Tally] = []
        with self.lake.open() as connection:
            for load in loads:
                imported = _count(
                    connection, f"SELECT count(*) FROM {ALIAS}.{load.table} WHERE {load.where}"
                )
                strays = _count(
                    connection,
                    f"SELECT count(*) FROM {ALIAS}.strays WHERE destination = ? AND source GLOB ?",
                    [load.table, load.source.split(" ")[0]],
                )
                tallies.append(
                    Tally(
                        source=load.source,
                        table=load.table,
                        expected=load.expected,
                        imported=imported,
                        strays=strays,
                        ok=imported + strays == load.expected,
                    )
                )
            tallies.extend(self._rebuilt(connection, logs))
        return tallies

    def _rebuilt(self, connection: duckdb.DuckDBPyConnection, logs: _Logs) -> list[Tally]:
        """The log tallies: lines, files, lossy lines, and the exact files' bytes rebuilt."""
        lines = _count(connection, f"SELECT count(*) FROM {ALIAS}.log_lines")
        files = _count(connection, f"SELECT count(DISTINCT file) FROM {ALIAS}.log_lines")
        lossy = _count(connection, f"SELECT count(*) FROM {ALIAS}.log_lines WHERE lossy")
        skipped = set(logs.lossy)
        hasher = hashlib.sha256()
        result = connection.execute(
            f"SELECT file, n, line FROM {ALIAS}.log_lines ORDER BY file, n"
        )
        while batch := result.fetchmany(_FETCH):
            for file, number, line in batch:
                if file in skipped:
                    continue
                hasher.update(("\n" if number > 1 else "").encode() + line.encode())
        rebuilt = hasher.hexdigest()
        exact = logs.files - len(skipped)
        return [
            Tally(
                source="batches/**/*.log",
                table="log_lines",
                expected=logs.lines,
                imported=lines,
                ok=lines == logs.lines,
            ),
            Tally(
                source="batches/**/*.log files",
                table="log_lines",
                expected=logs.files,
                imported=files,
                ok=files == logs.files,
            ),
            Tally(
                source="batches/**/*.log lossy lines",
                table="log_lines",
                expected=logs.lossy_lines,
                imported=lossy,
                ok=lossy == logs.lossy_lines,
                detail=", ".join(logs.lossy[:3]),
            ),
            Tally(
                source="batches/**/*.log bytes",
                table="log_lines",
                expected=exact,
                imported=exact,
                ok=rebuilt == logs.sha256,
                detail=f"sha256 {logs.sha256[:16]} files, {rebuilt[:16]} lake",
            ),
        ]


def _registered(path: Path) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """The registry's `runs` and `hosts` rows, read from a copy so the original is untouched.

    Even a read-only SQLite open of a WAL database creates its `-wal` and `-shm` files, so the
    database and its WAL are copied aside and the copy is what gets opened; the WAL rides along
    so commits not yet checkpointed into the database are read too. Empty when there is none.
    """
    if not path.is_file():
        return [], []
    with TemporaryDirectory() as scratch:
        copy = Path(scratch) / path.name
        for suffix in ("", "-wal"):
            source = Path(f"{path}{suffix}")
            if source.is_file():
                shutil.copyfile(source, Path(f"{copy}{suffix}"))
        with closing(sqlite3.connect(copy)) as registry:
            runs = registry.execute("SELECT target, handle, data, submitted_at FROM runs")
            found = runs.fetchall()
            onboarded = registry.execute("SELECT alias, facts, probed_at FROM hosts").fetchall()
        return found, onboarded


def _logfiles(batches: Path) -> list[Path]:
    """Every regular `.log` file under `batches`, sorted by its UTF-8 path as the lake sorts.

    Symbolic links are left out, and so is anything inside a `wandb` folder.
    """
    found = [
        path
        for path in batches.rglob("*.log")
        if not path.is_symlink()
        and path.is_file()
        and "wandb" not in path.relative_to(batches).parts[:-1]
    ]
    return sorted(found, key=lambda path: path.relative_to(batches).as_posix().encode())


def _split(path: Path) -> list[str]:
    """`path`'s lines, split on a newline alone.

    A JSON string may hold a character `splitlines` also breaks at, which would tear one record
    in two; a carriage return before the newline is only a Windows editor's and is dropped.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.removesuffix("\r") for line in text.split("\n")]


def _count(
    connection: duckdb.DuckDBPyConnection, sql: str, parameters: Sequence[object] = ()
) -> int:
    row = connection.execute(sql, list(parameters)).fetchone()
    return int(row[0]) if row else 0


def _parses(text: object) -> bool:
    """Whether `text` is a JSON document."""
    try:
        json.loads(str(text))
    except json.JSONDecodeError:
        return False
    return True


def _document(text: str) -> object:
    """`text` parsed as JSON, None when it is not."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _instant(value: object) -> str | None:
    """`value` when it is an ISO-8601 instant, else None: one bad stamp cannot sink an import."""
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except ValueError:
        return None


def _digest(entry: object) -> Mapping[str, object] | None:
    """A digest memory's entry as columns, None for a shape neither release wrote."""
    if not isinstance(entry, list) or not entry or not isinstance(entry[-1], str):
        return None
    numbers = entry[:-1]
    if not all(isinstance(number, int) for number in numbers):
        return None
    if len(numbers) == 4:
        size, inode, mtime, ctime = numbers
        return {
            "size": size,
            "inode": inode,
            "mtime_ns": mtime,
            "ctime_ns": ctime,
            "sha256": entry[-1],
        }
    if len(numbers) == 2:
        size, mtime = numbers
        return {"size": size, "mtime_ns": mtime, "sha256": entry[-1]}
    return None


def _set_aside(lake: Lake) -> Path:
    """Move `lake`'s catalog and data under `<out>/lake.aside/<instant>/`, returning where.

    Nothing is deleted: the set-aside pair keeps its layout and is removed by hand if wanted.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    aside = lake.out / "lake.aside" / stamp
    aside.mkdir(parents=True)
    for source in sorted(lake.out.glob(f"{lake.catalog.name}*")):
        shutil.move(source, aside / source.name)
    shutil.move(lake.data, aside / lake.data.name)
    return aside
