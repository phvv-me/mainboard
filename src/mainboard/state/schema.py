# THE STATE LAKE'S SCHEMA, versioned, and the one place a table or view of it is spelled.
#
# Every table is append-only: a writer adds rows and never updates or deletes one, so two
# processes appending at once can only ever add, and DuckLake's snapshot retry settles which
# commits first. What used to be a row updated in place (a run's verdict, a host's facts, a held
# machine) is a log of whole records instead, and "the current one" is a view taking the last
# record appended per key. Last means commit order, the DuckLake `rowid`, never a wall clock,
# since two machines' clocks disagree and one process can append twice in a microsecond. Removal
# is one more record with `dropped` set, which the view reads as absence.
#
# No column may be a DuckDB keyword outside the unreserved class: DuckDB 2.0 reads `at` as a
# type-function keyword and refuses it as a bare column name, and a schema that needs quoting in
# one query needs it in every query anybody ever types against the lake.
#
# The tables are SQLAlchemy tables in the `lake` schema, the alias every attach uses, so a query
# built from them names `lake.runs`. Their DDL and the views are compiled from copies outside any
# schema, executed inside the attached catalog, so a view resolves however the catalog is attached.

from duckdb_sqlalchemy import Dialect
from duckdb_sqlalchemy.datatypes import UBigInteger
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    ColumnElement,
    DateTime,
    Double,
    Integer,
    LargeBinary,
    MetaData,
    Select,
    String,
    Table,
    func,
    literal_column,
    select,
)
from sqlalchemy.schema import CreateTable

from .constants import ALIAS

# The SQL every statement is compiled to: DuckDB's, binding `?` placeholders in order.
dialect = Dialect(paramstyle="qmark")


_BARE = MetaData()
_LAKE = MetaData()


def _table(name: str, *columns: Column) -> Table:
    """`name` declared outside any schema, answered as its copy in the `lake` schema.

    The declaration serves its DDL and the views, and a query names the copy.
    """
    return Table(name, _BARE, *columns).to_metadata(_LAKE, schema=ALIAS)


def _stamped(name: str, *columns: Column) -> Table:
    """A log table whose every record carries its append time, `ts`."""
    return _table(name, Column("ts", DateTime(timezone=True)), *columns)


def _dropping(name: str, *columns: Column) -> Table:
    """A log whose view keeps the last record per key, which `dropped` set removes."""
    return _stamped(name, *columns, Column("dropped", Boolean))


schema_log = _stamped(
    "schema_log", Column("version", Integer), Column("spec", String), Column("engine", String)
)
imports = _stamped(
    "imports", Column("source", String), Column("destination", String), Column("rows", BigInteger)
)
events = _stamped(
    "events",
    Column("batch", String),
    Column("topic", String),
    Column("job", String),
    Column("data", JSON),
)
runs_log = _dropping(
    "runs_log",
    Column("target", String),
    Column("handle", String),
    Column("submitted_at", String),
    Column("record", JSON),
)
host_facts = _dropping(
    "host_facts", Column("alias", String), Column("probed_at", String), Column("facts", JSON)
)
receipts = _stamped(
    "receipts",
    Column("batch", String),
    Column("n", BigInteger),
    Column("run", String),
    Column("trial", String),
    Column("verdict", String),
    Column("host", String),
    Column("line", String),
)
costs = _table(
    "costs",
    Column("provider", String),
    Column("gpu", String),
    Column("region", String),
    Column("t_submit", Double),
    Column("t_running", Double),
    Column("t_ended", Double),
    Column("billed_usd", Double),
)
quotes = _stamped(
    "quotes",
    Column("provider", String),
    Column("gpu", String),
    Column("gpu_count", Integer),
    Column("spot", Boolean),
    Column("region", String),
    Column("rate_usd_hr", Double),
    Column("granularity_s", Integer),
    Column("minimum_s", Integer),
    Column("fees_usd", Double),
    Column("available", Boolean),
    Column("source", String),
)
holds_log = _dropping("holds_log", Column("alias", String), Column("held", JSON))
pulse = _stamped(
    "pulse", Column("key", String), Column("size", BigInteger), Column("grew", Double)
)
digests = _dropping(
    "digests",
    Column("kind", String),
    Column("path", String),
    Column("size", BigInteger),
    Column("mtime_ns", BigInteger),
    Column("sha256", String),
    Column("inode", UBigInteger),
    Column("ctime_ns", BigInteger),
)
job_specs = _stamped(
    "job_specs", Column("name", String), Column("sha256", String), Column("script", String)
)
closures = _stamped(
    "closures",
    Column("closure", String),
    Column("path", String),
    Column("blob", String),
    Column("status", String),
)
# Content-addressed objects in ordered chunks (`state.blobs`); a row from before chunking has no
# ordinal and is the whole object.
blobs = _table(
    "blobs",
    Column("sha256", String),
    Column("bytes", LargeBinary),
    Column("ordinal", BigInteger),
)
# Each chunk's MD5, recorded from bytes whose SHA-256 was just verified (`state.blobs`), so `mb
# lake check` proves every chunk intact inside DuckDB. A side table rather than a column of
# `blobs`, so gaining it never rewrote a byte of the chunks it vouches for.
checksums = _table(
    "checksums",
    Column("sha256", String),
    Column("ordinal", BigInteger),
    Column("md5", String),
)
# Which workspace-relative evidence file held which object (`state.evidence`), so the file can
# leave the tree and every reader still finds its bytes; project, node and run are read off the
# path for querying.
evidence_log = _dropping(
    "evidence_log",
    Column("path", String),
    Column("sha256", String),
    Column("size", BigInteger),
    Column("media_type", String),
    Column("project", String),
    Column("node", String),
    Column("run", String),
)
# RE findings identify an analysis of artifact bytes, independently of the workspace-relative
# file archive above. The full record preserves artifact identity, profile and raw diagnostics;
# result_json and the other projections let SQL query findings without decoding that envelope.
re_evidence = _dropping(
    "re_evidence",
    Column("evidence_id", String),
    Column("artifact_sha256", String),
    Column("kind", String),
    Column("provider", String),
    Column("tool_version", String),
    Column("operation", String),
    Column("params_json", JSON),
    Column("analysis_profile_digest", String),
    Column("coverage", String),
    Column("limitations_json", JSON),
    Column("unresolved_json", JSON),
    Column("result_json", JSON),
    Column("record_json", JSON),
)
log_lines = _stamped(
    "log_lines",
    Column("batch", String),
    Column("file", String),
    Column("n", BigInteger),
    Column("line", String),
    Column("lossy", Boolean),
)

# The tables outside any schema, which the DDL and the views are compiled from.
_bare = _BARE.tables


def latest(log: Table, *key: str, where: tuple[ColumnElement[bool], ...] = ()) -> Select:
    """The last record appended per `key` in `log` among those `where` keeps.

    A key whose last record is a drop is left out.
    """
    ranked = (
        select(
            log,
            func.row_number()
            .over(
                partition_by=[log.c[name] for name in key],
                order_by=literal_column("rowid").desc(),
            )
            .label("rank"),
        )
        .where(*where)
        .subquery()
    )
    kept = [column for column in ranked.c if column.name not in {"dropped", "rank"}]
    drops = (ranked.c.dropped.is_not(True),) if "dropped" in ranked.c else ()
    return select(*kept).where(ranked.c.rank == 1, *drops)


def _runs() -> Select:
    """The current runs with the record's everyday fields as columns.

    A query then filters by project or verdict without reaching into its JSON.
    """
    current = latest(_bare["runs_log"], "target", "handle", "submitted_at").subquery()
    record = current.c.record
    return select(
        current,
        *(record[field].as_string().label(field) for field in ("project", "name", "verdict")),
    )


def _offers() -> Select:
    """The latest catalog of rental offers: every quote of the newest sweep."""
    quoted = _bare["quotes"]
    newest = select(func.max(quoted.c.ts)).scalar_subquery()
    return select(*(column for column in quoted.c if column.name != "ts")).where(
        quoted.c.ts == newest
    )


class View:
    """One view over the tables, the current state a log of appends adds up to, and the table a
    query reads it as."""

    @property
    def ddl(self) -> str:
        """The CREATE OR REPLACE VIEW statement, its constants written inline."""
        body = self.definition.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
        return f"CREATE OR REPLACE VIEW {self.name} AS {body}"

    def __init__(self, name: str, definition: Select) -> None:
        self.name = name
        self.definition = definition
        columns = (Column(column.name, column.type) for column in definition.selected_columns)
        self.table = Table(name, _LAKE, *columns, schema=ALIAS)


views = (
    View("runs", _runs()),
    View("hosts", latest(_bare["host_facts"], "alias")),
    View("holds", latest(_bare["holds_log"], "alias")),
    View("evidence", latest(_bare["evidence_log"], "path")),
    View("offers", _offers()),
    View("re_evidence_current", latest(_bare["re_evidence"], "evidence_id")),
)
runs, hosts, holds, evidence, offers, re_evidence_current = (view.table for view in views)

# Every table, in the order a new lake creates them.
tables = tuple(table for table in _LAKE.sorted_tables if table.name in _bare)


def ddl(table: Table) -> str:
    """The CREATE TABLE statement, outside any schema so it lands in the attached catalog."""
    return str(CreateTable(_bare[table.name]).compile(dialect=dialect)).strip()


def kind(column: Column) -> str:
    """`column`'s DuckDB type as SQL spells it."""
    return column.type.compile(dialect=dialect)
