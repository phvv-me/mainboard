# The workspace's state lake: one DuckLake, its catalog a SQLite file and its data Parquet.
#
# Every rule here was learned by a rehearsal against a real workspace's state, and each is
# encoded rather than remembered:
#
# LAYOUT. The catalog is `<out>/lake.sqlite` and the data `<out>/lake/`, the catalog outside the
# data folder so a sweep of the data can never take the catalog with it. The catalog records an
# absolute data path, so a workspace moved or cloned elsewhere would read, or refuse to read, the
# old folder; every attach therefore derives the data path from the workspace root and overrides
# the recorded one.
#
# NO LONG-LIVED CONNECTIONS. Each operation opens a fresh in-memory DuckDB, attaches, works and
# detaches, the way a CLI command lives. Two processes appending at once each commit a snapshot,
# and the retry budget with the catalog's busy timeout is what makes the later one wait and retry
# rather than fail. Only maintenance takes a lock, an application-level file lock so two
# maintainers never compact at once; appends never take it and stay safe beside it.
#
# NEVER CREATED BY ACCIDENT. An ATTACH of a missing catalog creates an empty lake, and even one
# told not to leaves an empty SQLite file behind, so the catalog's existence is checked before
# any attach and `create` is the only way a lake comes to be.
#
# THE SPEC. A lake is created at the extension's latest spec, never a pinned one, and records it.
# Opening never migrates a catalog: an older one keeps working at its spec until `upgrade` is
# asked for, since a migration is a one-way step every other reader must be ready for.
#
# EXTENSIONS. ducklake and sqlite load from this tool's own extension folder (`database.load`),
# so a machine with no network still opens a lake once they are there.
#
# SERVED. `serving` opens DuckDB on the lake itself and serves it over Quack, DuckDB's own
# client-server protocol, on localhost only; another machine reaches it through an ssh tunnel.
# With `MB_LAKE=quack:host:port` (and `MB_LAKE_TOKEN`) set, writes and the session's reads attach
# the served writer instead of the workspace's files, under the same `lake` alias, and with
# `MB_LAKE_READ`/`MB_LAKE_READ_TOKEN` a read-only attach (`mb query`, `Results`) reaches the second
# endpoint on port+1, where DuckLake itself is opened READ_ONLY, so no reader can write, drop or
# maintain. A host with only the writer pair cannot `mb query`.
# Learned against DuckDB 2.0: a Quack client resolves names in the server's
# default database only (hence DuckDB opened on the lake, not attached to memory); bound
# parameters are dropped on the way to the server (duckdb-quack#323, still so at 2.0.0, hence
# `inlined`); a remote `query()` runs a statement twice (duckdb-quack#282), so writes go
# through tables only; and
# both ends must run the same DuckDB release. Maintenance stays with the machine holding the
# files.

from ..storage.database import quoted as quoted
from .attachment import Access
from .core import Lake, Serving
from .maintenance import Finding, Health
from .session import Session
from .staging import inlined, insert, staged

__all__ = [
    "Access",
    "Finding",
    "Health",
    "Lake",
    "Serving",
    "Session",
    "inlined",
    "insert",
    "quoted",
    "staged",
]
