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

from patos import FrozenModel

# The schema's own version, recorded in `schema_log` by `Lake.create` beside the DuckLake spec.
VERSION = 1


class Table(FrozenModel):
    """One append-only table: its name and its `(column, SQL type)` pairs in order."""

    name: str
    columns: tuple[tuple[str, str], ...]

    @property
    def ddl(self) -> str:
        """The CREATE TABLE statement, unqualified so it lands in the attached lake's schema."""
        body = ", ".join(f"{column} {kind}" for column, kind in self.columns)
        return f"CREATE TABLE {self.name} ({body})"

    @property
    def names(self) -> tuple[str, ...]:
        """The column names in order."""
        return tuple(column for column, _ in self.columns)


class View(FrozenModel):
    """One view over the tables, the current state a log of appends adds up to."""

    name: str
    select: str

    @property
    def ddl(self) -> str:
        """The CREATE VIEW statement."""
        return f"CREATE VIEW {self.name} AS {self.select}"


def _latest(table: str, *key: str) -> str:
    """The last record appended per `key` in `table`, unless that record is a drop."""
    columns = ", ".join(key)
    return (
        f"SELECT * EXCLUDE (dropped) FROM (SELECT * FROM {table} QUALIFY row_number() "
        f"OVER (PARTITION BY {columns} ORDER BY rowid DESC) = 1) WHERE dropped IS NOT TRUE"
    )


TABLES: tuple[Table, ...] = (
    Table(
        name="schema_log",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("version", "INTEGER"),
            ("spec", "VARCHAR"),
            ("engine", "VARCHAR"),
        ),
    ),
    Table(
        name="imports",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("source", "VARCHAR"),
            ("destination", "VARCHAR"),
            ("rows", "BIGINT"),
        ),
    ),
    Table(
        name="events",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("batch", "VARCHAR"),
            ("topic", "VARCHAR"),
            ("job", "VARCHAR"),
            ("data", "JSON"),
        ),
    ),
    Table(
        name="runs_log",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("target", "VARCHAR"),
            ("handle", "VARCHAR"),
            ("submitted_at", "VARCHAR"),
            ("record", "JSON"),
            ("dropped", "BOOLEAN"),
        ),
    ),
    Table(
        name="host_facts",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("alias", "VARCHAR"),
            ("probed_at", "VARCHAR"),
            ("facts", "JSON"),
            ("dropped", "BOOLEAN"),
        ),
    ),
    Table(
        name="receipts",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("batch", "VARCHAR"),
            ("n", "BIGINT"),
            ("run", "VARCHAR"),
            ("trial", "VARCHAR"),
            ("verdict", "VARCHAR"),
            ("host", "VARCHAR"),
            ("line", "VARCHAR"),
        ),
    ),
    Table(
        name="costs",
        columns=(
            ("provider", "VARCHAR"),
            ("gpu", "VARCHAR"),
            ("region", "VARCHAR"),
            ("t_submit", "DOUBLE"),
            ("t_running", "DOUBLE"),
            ("t_ended", "DOUBLE"),
            ("billed_usd", "DOUBLE"),
        ),
    ),
    Table(
        name="quotes",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("provider", "VARCHAR"),
            ("gpu", "VARCHAR"),
            ("gpu_count", "INTEGER"),
            ("spot", "BOOLEAN"),
            ("region", "VARCHAR"),
            ("rate_usd_hr", "DOUBLE"),
            ("granularity_s", "INTEGER"),
            ("minimum_s", "INTEGER"),
            ("fees_usd", "DOUBLE"),
            ("available", "BOOLEAN"),
            ("source", "VARCHAR"),
        ),
    ),
    Table(
        name="holds_log",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("alias", "VARCHAR"),
            ("held", "JSON"),
            ("dropped", "BOOLEAN"),
        ),
    ),
    Table(
        name="studies",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("study", "VARCHAR"),
            ("kind", "VARCHAR"),
            ("handle", "VARCHAR"),
            ("host", "VARCHAR"),
            ("state", "VARCHAR"),
            ("name", "VARCHAR"),
        ),
    ),
    Table(
        name="pulse",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("key", "VARCHAR"),
            ("size", "BIGINT"),
            ("grew", "DOUBLE"),
        ),
    ),
    Table(
        name="digests",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("kind", "VARCHAR"),
            ("path", "VARCHAR"),
            ("size", "BIGINT"),
            ("mtime_ns", "BIGINT"),
            ("sha256", "VARCHAR"),
            ("inode", "UBIGINT"),
            ("ctime_ns", "BIGINT"),
        ),
    ),
    Table(
        name="job_specs",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("name", "VARCHAR"),
            ("sha256", "VARCHAR"),
            ("script", "VARCHAR"),
        ),
    ),
    Table(
        name="closures",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("closure", "VARCHAR"),
            ("path", "VARCHAR"),
            ("blob", "VARCHAR"),
            ("status", "VARCHAR"),
        ),
    ),
    Table(
        name="log_lines",
        columns=(
            ("batch", "VARCHAR"),
            ("file", "VARCHAR"),
            ("n", "BIGINT"),
            ("line", "VARCHAR"),
            ("lossy", "BOOLEAN"),
        ),
    ),
    Table(
        name="strays",
        columns=(
            ("ts", "TIMESTAMPTZ"),
            ("source", "VARCHAR"),
            ("destination", "VARCHAR"),
            ("n", "BIGINT"),
            ("line", "VARCHAR"),
        ),
    ),
)

VIEWS: tuple[View, ...] = (
    View(name="runs", select=_latest("runs_log", "target", "handle", "submitted_at")),
    View(name="hosts", select=_latest("host_facts", "alias")),
    View(name="holds", select=_latest("holds_log", "alias")),
    View(
        name="offers",
        select="SELECT * EXCLUDE (ts) FROM quotes WHERE ts = (SELECT max(ts) FROM quotes)",
    ),
)

# Each table by name, the lookup `insert` types a batch of rows against.
BY_NAME: dict[str, Table] = {table.name: table for table in TABLES}
