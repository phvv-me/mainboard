# Every DuckDB connection this tool opens takes the same settings, by DuckDB's own names: the
# workspace's `[duckdb]` table, each overridden on one machine by `MB_DUCKDB_<NAME>` from the
# workspace `.env` or this process's environment. Instants read in UTC unless a setting says
# otherwise, so text a receipt wrote as a UTC instant reads back the same on every host.
#
# Settings are applied with SET after connecting, since DuckDB refuses a session option (the
# progress bar) in the startup config; what a call site requires at startup (where extensions
# load from) it passes as `config`. An unknown name is DuckDB's own error, never skipped.
#
# Extensions load from a folder this tool owns under the user's cache, never the workspace and
# never DuckDB's default shared with every other program. Loading comes first and installing only
# when loading fails, so a machine with no network still works once the extensions are there.

import os
import sys
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

import duckdb

from ..core.errors import MissionError
from ..core.project import Project
from ..manifest.loading import composition

type Setting = str | int | float | bool

# What every connection holds unless the workspace or the machine says otherwise.
DEFAULTS: dict[str, Setting] = {"timezone": "UTC", "enable_progress_bar": False}
# What every database starts with. DuckDB pins one worker thread to each core on a machine of more
# than 64 cores, from `import duckdb` on, and a process this tool is part of must keep the
# affinity its own work sets: a spawned worker of a cutok CPU baseline held 71 threads pinned one
# per core on the GH200 (2026-10-10). `_unpinned` releases the instance the import opened.
STARTUP: dict[str, Setting] = {"pin_threads": "off"}
PREFIX = "MB_DUCKDB_"
# The repository an extension installs from when it is not one of DuckDB's core extensions.
COMMUNITY = {"sitting_duck"}


def quoted(text: str) -> str:
    """`text` as a SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def cache_home(
    platform: str = sys.platform,
    environ: Mapping[str, str] = os.environ,
    home: Path | None = None,
) -> Path:
    """The user's cache directory on `platform`: `~/Library/Caches` on macOS, else
    `XDG_CACHE_HOME` or `~/.cache`."""
    base = home or Path.home()
    if platform == "darwin":
        return base / "Library" / "Caches"
    return Path(environ.get("XDG_CACHE_HOME") or base / ".cache")


def extension_home() -> Path:
    """Where this tool keeps DuckDB's extensions: its own folder under the user's cache."""
    return cache_home() / Project().name / "duckdb"


def load(
    connection: duckdb.DuckDBPyConnection, names: Iterable[str], folder: Path | None = None
) -> None:
    """Load extensions `names` from `folder` (`extension_home()` by default), installing only
    what fails to load.

    Raises MissionError naming the folder when an extension is neither there nor installable, as
    on a machine that never went online.
    """
    folder = folder or extension_home()
    connection.execute(f"SET extension_directory = {quoted(folder.as_posix())}")
    for name in names:
        try:
            connection.load_extension(name)
        except duckdb.IOException:
            try:
                connection.install_extension(
                    name, repository="community" if name in COMMUNITY else None
                )
                connection.load_extension(name)
            except duckdb.Error as fault:
                raise MissionError(
                    f"DuckDB extension {name} is not in {folder} and could not be installed: "
                    f"{first_line(fault)}"
                ) from fault


def first_line(fault: BaseException) -> str:
    """The first line of what `fault` says."""
    return (str(fault).splitlines() or [""])[0]


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


def _unpinned() -> None:
    """Relaunch the threads of the instance `import duckdb` opened, none pinned to a core.

    A thread count change relaunches them, so it goes to one and back.
    """
    if (os.cpu_count() or 0) <= 64:
        return
    [(threads,)] = duckdb.execute("SELECT current_setting('threads')").fetchall()
    duckdb.execute("SET pin_threads = 'off'")
    duckdb.execute("SET threads = 1")
    duckdb.execute(f"SET threads = {threads}")


_unpinned()
