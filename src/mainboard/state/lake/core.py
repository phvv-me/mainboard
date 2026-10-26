"""One workspace's state lake: located from its manifest, appended to, transacted on, shared by
one session per catalog, and served over Quack."""

import math
import os
import secrets
import sqlite3
import weakref
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from threading import RLock

import duckdb
from cyclopts import Parameter
from filelock import FileLock
from patos import FrozenModel
from pydantic import ConfigDict, Field
from sqlalchemy import Table

from ...core.errors import MissionError
from ...core.project import Project
from ...manifest.composing import composition
from ...manifest.schema.project.workspace import Header
from ...runtime.tree import FileBudget
from ..constants import QUACK, RETRIES
from ..storage.database import connect, first_line, quoted
from .maintenance import Maintained
from .session import Session, patiently
from .staging import insert

# Quack's own default port.
_PORT = 9494

# The live session on each catalog, gone when its last holder is, and the lock creating one.
_SESSIONS: weakref.WeakValueDictionary[tuple[str, int], Session] = weakref.WeakValueDictionary()
_SHARING = RLock()


@Parameter(name="*")
class Serving(FrozenModel):
    """How a lake is served over Quack: the local port, the token clients present, the limit."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    port: int = _PORT
    """the local port the writer listens on; the read-only endpoint takes the one after it."""
    token: str = ""
    """what clients must present to write; the one kept in the state directory (else a fresh
    one, then kept) when omitted. The read-only endpoint always keeps its own."""
    timeout: float = Field(default=10.0, ge=0, allow_inf_nan=False)
    """execution limit for each server query in seconds, including blocking Quack operators;
    10 by default, 0 disables it. Client limits are not forwarded."""

    @property
    def read_port(self) -> int:
        """The port of the read-only endpoint, the one after the writer's."""
        return self.port + 1


# A lake served on Quack's own port with the kept token.
_SERVED = Serving()


class Lake(Maintained):
    """One workspace's state lake (see the package for the rules it keeps)."""

    @classmethod
    def at(cls, root: Path) -> Lake:
        """The lake of the workspace at `root`, where its manifest's `[workspace] lake` says."""
        root = root.absolute()
        manifest = Project().manifest(root)
        header = (
            composition(manifest).manifest.workspace if manifest.is_file() else Header(name="")
        )
        return cls(root=root, home=header.lake_home(root))

    @contextmanager
    def _endpoint(
        self, catalog: str, *, port: int, token: str, timeout: float, read_only: bool = False
    ) -> Generator[tuple[str, str]]:
        """One Quack endpoint on the catalog at `catalog`, yielding its URI and token.

        read_only: open DuckLake itself READ_ONLY, which refuses every write, DROP and
            maintenance call a client sends; a client's own READ_ONLY attach is not enforced.
        """
        uri = f"quack:localhost:{port}"
        config: dict[str, str | bool] = {
            "extension_directory": self.extensions.as_posix(),
            "autoinstall_known_extensions": False,
            "temp_directory": (self.out / "run" / f"lake.tmp.{port}").as_posix(),
        }
        if read_only:
            config["access_mode"] = "READ_ONLY"
        with (
            closing(connect(f"ducklake:sqlite:{catalog}", config=config)) as served,
            _refused(uri),
        ):
            self._started(served, uri=uri, token=token, timeout=timeout)
            yield uri, token
            served.execute(f"CALL quack_stop({quoted(uri)})")

    @contextmanager
    def serving(self, asked: Serving = _SERVED) -> Generator[tuple[str, str]]:
        """This lake served over Quack on localhost as `asked` while the block runs.

        Two endpoints on the catalog itself, the one database a Quack client resolves names in:
        the writer on `asked.port`, and the read-only one on the next port, which no client can
        write, drop or maintain through. Both are opened after an ordinary attach brought the
        schema up to date. It yields the writer's URI and token. Appends from this machine keep
        going straight to the files beside it, as they always do.
        """
        self._held("serve")
        with self.ready().open(write=True) as connection:
            self.evolve(connection)
            self._load(connection, QUACK)
        self._rehome()
        folder, name = self.catalog.parent.as_posix(), self.catalog.name
        with (
            FileBudget.permitted(),
            self._endpoint(
                f"{folder}/{name}",
                port=asked.port,
                token=asked.token or self._kept_token(self.token),
                timeout=asked.timeout,
            ) as writer,
            # DuckDB refuses a second configuration of one path in a process; "/./" spells it anew.
            self._endpoint(
                f"{folder}/./{name}",
                port=asked.read_port,
                token=self._kept_token(self.read_token),
                timeout=asked.timeout,
                read_only=True,
            ),
        ):
            yield writer

    @staticmethod
    def _inode(path: Path) -> int:
        """`path`'s inode, 0 for a file not there yet."""
        try:
            return path.stat().st_ino
        except FileNotFoundError:
            return 0

    @staticmethod
    def _kept_token(path: Path) -> str:
        """The token kept at `path`, made and kept on first use, readable by its owner only."""
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            with os.fdopen(
                os.open(path, flags, 0o600), "w", encoding="utf-8", newline="\n"
            ) as made:
                made.write(secrets.token_urlsafe(24))
        path.chmod(0o600)
        return path.read_text(encoding="utf-8").strip()

    def _rehome(self) -> None:
        """Point the catalog's recorded data path at this workspace's data folder.

        Serving opens the catalog as it is, and a catalog carried from another machine records
        that machine's folder, which an ordinary attach overrides and serving cannot. Files are
        recorded relative to the data path, so this one row is all that moves.
        """
        here = self.data.as_posix() + "/"
        with FileLock(self.lock), sqlite3.connect(self.catalog) as catalog:
            catalog.execute(
                "UPDATE ducklake_metadata SET value = ? "
                "WHERE key = 'data_path' AND scope IS NULL AND value <> ?",
                (here, here),
            )
        catalog.close()

    def _started(
        self, served: duckdb.DuckDBPyConnection, *, uri: str, token: str, timeout: float
    ) -> None:
        """Start Quack serving the lake on `served` at `uri`, each query bounded by `timeout`."""
        served.execute(f"SET GLOBAL ducklake_max_retry_count = {RETRIES}")
        served.execute("SET GLOBAL TimeZone = 'UTC'")
        served.execute(f"SET GLOBAL max_execution_time = {math.ceil(timeout * 1000)}")
        self._load(served, QUACK)
        served.execute(f"CALL quack_serve({quoted(uri)}, token => {quoted(token)})")
        # A client may send the server any statement, so file access is confined to the data
        # folder and the settings are locked; the order matters, as external access cannot return.
        served.execute(f"SET GLOBAL allowed_directories = [{quoted(self.data.as_posix() + '/')}]")
        served.execute("SET GLOBAL enable_external_access = false")
        served.execute("SET GLOBAL lock_configuration = true")

    def _transacted[T](self, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        with self.open(write=True) as connection:
            self.evolve(connection)
            connection.execute("BEGIN")
            done = work(connection)
            connection.execute("COMMIT")
        return done

    def append(self, table: Table, rows: Iterable[Mapping[str, object]]) -> int:
        """Append `rows` to `table` in one commit, returning how many were appended.

        Retried with a fresh attach while SQLite answers that the catalog is locked.
        """
        staged = list(rows)
        for attempt in patiently():
            with attempt, self.open(write=True) as connection:
                insert(connection, table, staged)
        return len(staged)

    def session(self) -> Session:
        """This process's one session on this lake, shared by every holder while any holds it.

        An attach costs a tenth of a second, and a command reads the lake from several places
        (the registry, a manifest's held machines, a verdict), so they share one. A catalog file
        replaced under the same name (a restore, a fresh import) is a different lake and gets a
        session of its own, since an attach keeps reading the file it opened.
        """
        identity = (self.served or str(self.catalog), Lake._inode(self.catalog))
        with _SHARING:
            shared = _SESSIONS.get(identity)
            if shared is None:
                shared = _SESSIONS[identity] = Session(self)
            return shared

    def transact[T](self, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """`work` in one transaction on a fresh write attach, committed whole or not at all.

        The lake is brought to this release's schema first. Retried with a fresh attach while
        SQLite answers that the catalog is locked; a failure commits nothing, so a retry cannot
        double what `work` appends.
        """
        return patiently()(lambda: self._transacted(work))


@contextmanager
def _refused(uri: str) -> Generator[None]:
    """Say why the lake could not be served at `uri` when DuckDB refuses."""
    try:
        yield
    except duckdb.Error as fault:
        raise MissionError(f"could not serve the lake at {uri}: {first_line(fault)}") from fault
