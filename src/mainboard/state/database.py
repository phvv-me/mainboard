# Every DuckDB connection this tool opens takes the same settings, by DuckDB's own names: the
# workspace's `[duckdb]` table, each overridden on one machine by `MB_DUCKDB_<NAME>` from the
# workspace `.env` or this process's environment. Instants read in UTC unless a setting says
# otherwise, so text a receipt wrote as a UTC instant reads back the same on every host.
#
# Settings are applied with SET after connecting, since DuckDB refuses a session option (the
# progress bar) in the startup config; what a call site requires at startup (where extensions
# load from) it passes as `config`. An unknown name is DuckDB's own error, never skipped.

import os
from collections.abc import Mapping
from functools import cache

import duckdb

from ..core.errors import MissionError
from ..core.project import Project
from ..manifest.loading import composition

type Setting = str | int | float | bool

# What every connection holds unless the workspace or the machine says otherwise.
DEFAULTS: dict[str, Setting] = {"timezone": "UTC", "enable_progress_bar": False}
# What every database starts with. DuckDB pins one worker thread to each core on a machine of more
# than 64 cores, and a process this tool hosts must keep the affinity its own work sets: a
# spawned worker of a cutok CPU baseline held 72 threads pinned one per core on the GH200 (Oct 10).
STARTUP: dict[str, Setting] = {"pin_threads": "off"}
PREFIX = "MB_DUCKDB_"


def quoted(text: str) -> str:
    """`text` as a SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def connect(
    database: str = ":memory:", *, config: Mapping[str, Setting] | None = None
) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection to `database` holding the workspace's settings.

    config: the startup options the caller requires, which DuckDB reads before any setting.
    """
    connection = duckdb.connect(database, config={**STARTUP, **(config or {})})
    for name, value in settings().items():
        literal = str(value).lower() if isinstance(value, bool) else str(value)
        connection.execute(f"SET {name} = {quoted(literal)}")
    return connection


@cache
def settings() -> dict[str, Setting]:
    """The defaults, then the workspace's `[duckdb]`, then `MB_DUCKDB_*` from `.env` and the
    environment, names folded to lower case as DuckDB reads them."""
    root = Project().workspace()
    manifest = Project().manifest(root)
    declared = composition(manifest).manifest.duckdb if manifest.is_file() else {}
    machine = {
        name.removeprefix(PREFIX): value
        for name, value in (Project().dotenv(root) | dict(os.environ)).items()
        if name.startswith(PREFIX)
    }
    merged = {name.lower(): value for name, value in (DEFAULTS | declared | machine).items()}
    if wrong := sorted(name for name in merged if not name.isidentifier()):
        raise MissionError(f"DuckDB settings are plain names, not {', '.join(wrong)}")
    return merged
