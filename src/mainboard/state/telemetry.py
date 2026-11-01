"""The telemetry lake: what Claude Code, Codex and mb's spool send the sink, and the views over it.

It is a DuckLake of its own under `.mb/telemetry/`, a SQLite catalog and Parquet data on the
internal disk, apart from the research lake: the sink sets a whole catalog's expiry to 900 s and
compacts it on its own schedule, which the research lake must never undergo. The sink is its one
writer. mb attaches it READ_ONLY and nothing else, since the sink's DuckDB (1.5.6 in duckdb-otlp
0.7.3) writes DuckLake 1.0 and refuses a catalog that a newer DuckDB migrated, so mb never creates,
migrates or maintains it.

Telemetry is untrusted and lossy by rule: any harness holding the sink's token can append to it,
and a hard kill of the sink drops the rows of its last few seconds. Nothing reads it to decide,
grant or spend. The raw table keeps what the harnesses send, `user.email` and Codex's tool
arguments included; the views name every column they show, so neither ever reaches one.

A turn is one model response: a Claude `api_request`, or a Codex `response.completed` that produced
output. Codex opens a turn with a warm-up response that reads the prompt and writes nothing (12,438
input tokens and none out, 2026-10-29), which its own turn total leaves out, so the views do too.
Token columns mean the same for both harnesses: `input` is the input read uncached, `cache_read`
and `cache_write` the cached input, `output` what was written. `cost_usd` is Claude's
API-equivalent estimate, not a bill; Codex reports no cost.
"""

from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import duckdb
from patos import FrozenModel
from pydantic import AwareDatetime, TypeAdapter

from ..constants import ASK_ACT, ASK_EVENT, ASK_TARGET, EVENT_ID
from ..core import MissionError, write_atomic
from ..core.events import Event, EventSpool, Key
from ..core.project import Project
from ..enums import JobFrame, JobReading
from ..render.values import Cell
from .enums import Access
from .lake.attachment import Attachment

# The trailing windows `standings` sums, as a subscription meters use: five hours and a week.
_WINDOWS = "('5h', INTERVAL 5 HOUR), ('7d', INTERVAL 7 DAY)"
_JOB_FRAMES = ", ".join(f"'{frame}'" for frame in JobFrame)
_SPEC_SQL = "SELECT value FROM lake.options() WHERE option_name = 'version'"
# Which of some events' ids mb's rows carry, looked for no earlier than the oldest event.
_HELD_SQL = f"""
    SELECT DISTINCT attribute(log_attributes, '{EVENT_ID}') FROM lake.main.otlp_logs
    WHERE service_name = 'mb' AND time_unix_nano >= ?
        AND list_contains(?, attribute(log_attributes, '{EVENT_ID}'))
"""
_LEDGER = TypeAdapter(dict[str, int])

_VIEWS = (
    """
    CREATE OR REPLACE TEMP MACRO attribute(attributes, key) AS
        json_extract_string(attributes, '$."' || key || '"')
    """,
    """
    CREATE OR REPLACE TEMP MACRO counted(attributes, key) AS
        coalesce(try_cast(attribute(attributes, key) AS BIGINT), 0)
    """,
    """
    CREATE OR REPLACE TEMP VIEW turns AS
    SELECT 'claude' AS harness, 'claude:' || attribute(log_attributes, 'session.id') AS session,
        time_unix_nano AS answered, attribute(log_attributes, 'model') AS model,
        attribute(log_attributes, 'prompt.id') AS prompt,
        counted(log_attributes, 'input_tokens') AS input,
        counted(log_attributes, 'cache_read_tokens') AS cache_read,
        counted(log_attributes, 'cache_creation_tokens') AS cache_write,
        counted(log_attributes, 'output_tokens') AS output, 0 AS reasoning,
        try_cast(attribute(log_attributes, 'cost_usd') AS DOUBLE) AS cost_usd,
        try_cast(attribute(log_attributes, 'duration_ms') AS BIGINT) AS duration_ms
    FROM lake.main.otlp_logs
    WHERE service_name = 'claude-code' AND attribute(log_attributes, 'event.name') = 'api_request'
    UNION ALL
    SELECT 'codex', 'codex:' || attribute(log_attributes, 'conversation.id'),
        coalesce(time_unix_nano,
            try_cast(attribute(log_attributes, 'event.timestamp') AS TIMESTAMP),
            observed_time_unix_nano),
        attribute(log_attributes, 'model'), NULL,
        counted(log_attributes, 'input_token_count')
            - counted(log_attributes, 'cached_token_count'),
        counted(log_attributes, 'cached_token_count'),
        counted(log_attributes, 'cache_write_token_count'),
        counted(log_attributes, 'output_token_count'),
        counted(log_attributes, 'reasoning_token_count'), NULL, NULL
    FROM lake.main.otlp_logs
    WHERE service_name LIKE 'codex%'
        AND attribute(log_attributes, 'event.name') = 'codex.sse_event'
        AND attribute(log_attributes, 'event.kind') = 'response.completed'
        AND counted(log_attributes, 'output_token_count') > 0
    """,
    """
    CREATE OR REPLACE TEMP VIEW sessions AS
    SELECT session, harness, min(answered) AS started, max(answered) AS latest,
        string_agg(DISTINCT model, ',' ORDER BY model) AS models, count(*) AS turns,
        sum(input) AS input, sum(cache_read) AS cache_read, sum(cache_write) AS cache_write,
        sum(output) AS output, round(sum(cost_usd), 6) AS cost_usd,
        coalesce(any_value(compacted.compactions), 0) AS compactions
    FROM turns LEFT JOIN (
        SELECT 'claude:' || attribute(log_attributes, 'session.id') AS session,
            count(*) AS compactions
        FROM lake.main.otlp_logs
        WHERE service_name = 'claude-code'
            AND attribute(log_attributes, 'event.name') = 'compaction'
        GROUP BY ALL
    ) AS compacted USING (session)
    GROUP BY session, harness
    """,
    """
    CREATE OR REPLACE TEMP VIEW spend AS
    SELECT CAST(answered AS DATE) AS day, harness, model, count(*) AS turns, sum(input) AS input,
        sum(cache_read) AS cache_read, sum(cache_write) AS cache_write, sum(output) AS output,
        round(sum(cost_usd), 6) AS cost_usd
    FROM turns GROUP BY ALL
    """,
    f"""
    CREATE OR REPLACE TEMP VIEW standings AS
    SELECT harness, span, count(*) AS turns, sum(input) AS input, sum(cache_read) AS cache_read,
        sum(cache_write) AS cache_write, sum(output) AS output,
        round(sum(cost_usd), 6) AS cost_usd,
        min(answered) AS started, max(answered) AS latest
    FROM turns, (VALUES {_WINDOWS}) AS windows(span, width)
    WHERE answered > now()::TIMESTAMP - width
    GROUP BY ALL
    """,
    f"""
    CREATE OR REPLACE TEMP VIEW asks AS
    SELECT time_unix_nano AS asked, attribute(log_attributes, '{ASK_ACT}') AS act,
        attribute(log_attributes, '{ASK_TARGET}') AS target, body AS why
    FROM lake.main.otlp_logs WHERE service_name = 'mb' AND event_name = '{ASK_EVENT}'
    """,
    f"""
    CREATE OR REPLACE TEMP VIEW jobs AS
    SELECT attribute(log_attributes, '{Key.job}') AS job,
        any_value(attribute(log_attributes, '{Key.host}')) AS host,
        min(time_unix_nano) FILTER (WHERE event_name = '{JobFrame.started}') AS started,
        max(time_unix_nano) FILTER (WHERE event_name = '{JobFrame.heartbeat}') AS last_beat,
        count(*) FILTER (WHERE event_name = '{JobFrame.heartbeat}') AS beats,
        max(counted(log_attributes, '{JobReading.rss_bytes}')) AS peak_rss_bytes,
        max(time_unix_nano) FILTER (WHERE event_name = '{JobFrame.settled}') AS settled,
        max(try_cast(attribute(log_attributes, '{JobReading.exit_code}') AS BIGINT))
            FILTER (WHERE event_name = '{JobFrame.settled}') AS exit_code,
        any_value(attribute(log_attributes, '{JobReading.stop}'))
            FILTER (WHERE event_name = '{JobFrame.settled}') AS stop
    FROM lake.main.otlp_logs
    WHERE service_name = 'mb' AND event_name IN ({_JOB_FRAMES})
        AND attribute(log_attributes, '{Key.job}') IS NOT NULL
    GROUP BY job
    """,
)


class Waiting(FrozenModel):
    """One spool file's lines that no pass has delivered yet.

    oldest: when the first of them happened; None for a file holding a line that is no spool
        line, which the collect pass sets aside unsent and never retries.
    """

    file: str
    lines: int
    oldest: AwareDatetime | None

    def row(self, *, at: datetime) -> dict[str, Cell]:
        """This file as a printed row: its state, its oldest event's instant and age at `at`."""
        if self.oldest is None:
            return {"file": self.file, "state": "refused", "lines": self.lines}
        return {
            "file": self.file,
            "state": "waiting",
            "lines": self.lines,
            "oldest": self.oldest.astimezone().isoformat(timespec="seconds"),
            "age_s": round((at - self.oldest).total_seconds()),
        }


class TelemetryLake(FrozenModel):
    """One workspace's telemetry lake, read through its views and never written by mb.

    root: the workspace whose `.mb/telemetry/` holds it.

    `turns` has a row per model response; `sessions` one per session, `<harness>:<id>` as
    `Key.session` spells it; `spend` one per UTC day, harness and model; `standings` one per
    harness and trailing window (`5h`, `7d`), an estimate from telemetry that the vendor's own
    meter overrules; `asks` one per ask the policy filed and a pass delivered, which
    `policy.asks` reads beside the spool; `jobs` one per `mb.job`, its start, beats, peak memory
    and settle as the host's runner framed them, with no settle for a job killed outright.
    What still waits in the spool is `pending`, read from
    its files, since no lake holds it yet.
    """

    root: Path

    @contextmanager
    def read(self) -> Generator[duckdb.DuckDBPyConnection]:
        """A connection with the lake attached READ_ONLY as `lake` and the views defined over it.

        Refused, with what to run, while the sink has not yet created the lake.
        """
        if not self.attachment.exists():
            raise MissionError(
                f"no telemetry lake at {self.attachment.catalog}; the sink creates it once "
                f"`{Project().name} serve` runs it"
            )
        with self.attachment.attached(access=Access.READ) as connection:
            for statement in _VIEWS:
                connection.execute(statement)
            yield connection

    @property
    def attachment(self) -> Attachment:
        """The lake's files, never a served lake whatever `MB_LAKE` says."""
        return Attachment(root=self.root, home=self.home, served="")

    @property
    def home(self) -> Path:
        """The lake's directory: its catalog `lake.sqlite`, its data `lake/`, the sink's files."""
        return Project().out(self.root) / "telemetry"

    @property
    def ledger(self) -> Path:
        """Where the collect pass keeps the byte offset it delivered each spool file up to."""
        return self.home / "collected.json"

    @property
    def spool(self) -> Path:
        """Where mb's own events wait as OTLP lines until the collect pass delivers them."""
        return self.home / "spool"

    @staticmethod
    def _cell(value: Cell | datetime | date) -> Cell:
        """A value as a printed cell: an instant in this machine's zone, a day as its ISO date."""
        if isinstance(value, datetime):
            return value.replace(tzinfo=UTC).astimezone().isoformat(timespec="seconds")
        if isinstance(value, date):
            return value.isoformat()
        return value

    @staticmethod
    def _waiting(path: Path, offset: int) -> Waiting:
        """What `path` holds past `offset`, every line of it when one is no spool line."""
        try:
            events = EventSpool.read(path, offset).events
        except ValueError:
            lines = path.read_bytes()[offset:].count(b"\n")
            return Waiting(file=path.name, lines=lines, oldest=None)
        oldest = min((event.at for event in events), default=None)
        return Waiting(file=path.name, lines=len(events), oldest=oldest)

    def held(self, events: Sequence[Event]) -> frozenset[UUID]:
        """The ids of `events` the lake already holds; none before the sink has made it."""
        if not events or not self.attachment.exists():
            return frozenset()
        since = min(event.at for event in events).astimezone(UTC).replace(tzinfo=None)
        with self.read() as connection:
            rows = connection.execute(_HELD_SQL, [since, [str(e.id) for e in events]]).fetchall()
        return frozenset(UUID(held) for (held,) in rows)

    def keep(self, offsets: Mapping[str, int]) -> None:
        """Write the delivered offsets whole, replacing the old ledger in one rename."""
        write_atomic(self.ledger, _LEDGER.dump_json(dict(offsets)), mode=0o600)

    def offsets(self) -> dict[str, int]:
        """The byte offset delivered up to, by spool file name; none before the first pass."""
        try:
            return _LEDGER.validate_json(self.ledger.read_bytes())
        except FileNotFoundError:
            return {}

    def pending(self) -> list[Waiting]:
        """Each spool file's lines not yet delivered by file name, then the files set aside."""
        offsets = self.offsets()
        spooled = sorted(self.spool.glob("*.jsonl"))
        files = [TelemetryLake._waiting(path, offsets.get(path.name, 0)) for path in spooled]
        files += [TelemetryLake._waiting(path, 0) for path in sorted(self.spool.glob("*.refused"))]
        return [file for file in files if file.lines]

    def rows(self, sql: str, parameters: Iterable[Cell] = ()) -> list[dict[str, Cell]]:
        """The rows of one query over the views, each a printable record by column name."""
        with self.read() as connection:
            result = connection.execute(sql, list(parameters))
            names = [column for column, *_ in result.description or ()]
            return [
                {name: TelemetryLake._cell(value) for name, value in zip(names, row, strict=True)}
                for row in result.fetchall()
            ]

    def spec(self) -> str:
        """The DuckLake spec the catalog is at, which the sink's binary must still write."""
        with self.read() as connection:
            row = connection.execute(_SPEC_SQL).fetchone()
        return str(row[0]) if row else ""
