"""Telemetry lakes holding the records the sink stores, written by this tool's DuckDB."""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from patos import FrozenModel

from mainboard.constants import EVENT_ID
from mainboard.core.events import Event, JobFrame, Key
from mainboard.state import TelemetryLake
from mainboard.state.lake import Access

from .constants import EMAIL

_MICROS_PER_USD = 1_000_000
# The sink's own `otlp_logs` columns, as duckdb-otlp 0.7.3 creates them.
_TABLE = """
CREATE OR REPLACE TABLE lake.main.otlp_logs (
    time_unix_nano TIMESTAMP, observed_time_unix_nano TIMESTAMP, trace_id VARCHAR,
    span_id VARCHAR, service_name VARCHAR, service_namespace VARCHAR, service_instance_id VARCHAR,
    severity_number INTEGER, severity_text VARCHAR, event_name VARCHAR, body VARCHAR,
    resource_attributes VARCHAR, scope_name VARCHAR, scope_version VARCHAR,
    scope_attributes VARCHAR, log_attributes VARCHAR, dropped_attributes_count INTEGER,
    flags INTEGER
)
"""
_INSERT = (
    "INSERT INTO lake.main.otlp_logs "
    "(time_unix_nano, observed_time_unix_nano, service_name, log_attributes, event_name) "
    "VALUES (?, ?, ?, ?, ?)"
)


class Record(FrozenModel):
    """One log record as the sink stores it: instants naive in UTC, attributes as JSON text.

    at: the record's own time, None as Codex sends it.
    event: the record's `eventName`, as mb's spool lines carry one; empty as the harnesses send.
    """

    at: datetime | None
    attributes: dict[str, str | int | float | bool]
    observed: datetime
    service: str
    event: str = ""


def claude_turn(
    session: str,
    at: datetime,
    *,
    model: str = "claude-haiku-5-5",
    tokens: int = 1,
    micros: int = 1,
) -> Record:
    """A Claude `api_request` reading and writing `tokens` of each kind, costing `micros` µUSD."""
    attributes: dict[str, str | int | float | bool] = {
        "event.name": "api_request",
        "session.id": session,
        "model": model,
        "input_tokens": str(tokens),
        "cache_read_tokens": str(tokens),
        "cache_creation_tokens": str(tokens),
        "output_tokens": str(tokens),
        "cost_usd": str(micros / _MICROS_PER_USD),
        "user.email": EMAIL,
    }
    return Record(at=at, attributes=attributes, observed=at, service="claude-code")


def codex_turn(
    conversation: str, at: datetime, *, read: int = 2, cached: int = 1, output: int = 1
) -> Record:
    """A Codex `response.completed` reading `read` tokens, `cached` of them from its cache.

    Its counts come in Codex's own mixed types, and it carries no record time.
    """
    attributes: dict[str, str | int | float | bool] = {
        "event.name": "codex.sse_event",
        "event.kind": "response.completed",
        "event.timestamp": f"{at.isoformat()}Z",
        "conversation.id": conversation,
        "model": "gpt-6.1-sol",
        "input_token_count": str(read),
        "cached_token_count": cached,
        "output_token_count": str(output),
        "reasoning_token_count": 0,
        "user.email": EMAIL,
    }
    return Record(at=None, attributes=attributes, observed=at, service="codex_exec")


def job_frame(
    job: str, frame: JobFrame, at: datetime, readings: dict[str, int | str] | None = None
) -> Record:
    """A frame of `job` as the sink stores mb's spool line: its event name, host and `readings`."""
    attributes: dict[str, str | int | float | bool] = {
        Key.host: "gold",
        Key.job: job,
        **(readings or {}),
    }
    return Record(at=at, attributes=attributes, observed=at, service="mb", event=frame)


def stored(events: Sequence[Event]) -> list[Record]:
    """The rows the sink keeps of spool lines `events`, which is how a pass finds them again."""
    return [
        Record(
            at=event.at.replace(tzinfo=None),
            attributes={EVENT_ID: str(event.id)},
            observed=event.at.replace(tzinfo=None),
            service="mb",
            event=event.name,
        )
        for event in events
    ]


def lake_with(root: Path, records: Sequence[Record]) -> TelemetryLake:
    """The telemetry lake of the workspace at `root`, holding `records` and nothing else."""
    lake = TelemetryLake(root=root)
    rows = [
        [
            record.at,
            record.observed,
            record.service,
            json.dumps(record.attributes),
            record.event or None,
        ]
        for record in records
    ]
    lake.home.mkdir(parents=True, exist_ok=True)
    with lake.attachment.attached(access=Access.CREATE) as connection:
        connection.execute(_TABLE)
        if rows:
            connection.executemany(_INSERT, rows)
    return lake
