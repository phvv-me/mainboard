"""How a lake is reached: where its files live, and a DuckDB connection with it attached."""

from collections.abc import Generator, Iterable
from contextlib import closing, contextmanager, suppress
from enum import StrEnum, auto
from pathlib import Path
from threading import Event, Thread
from typing import Any

import duckdb
from patos import FrozenModel
from pydantic import Field
from sqlalchemy import ClauseElement

from ...core.errors import MissionError
from ...core.project import Project
from ...runtime.tree import FileBudget
from ..constants import ALIAS, QUACK, RETRIES
from ..storage.database import connect, extension_home, first_line, load, quoted
from .staging import compiled, inlined

# The DuckDB extensions an attach needs.
_EXTENSIONS = ("ducklake", "sqlite")

# How long, in milliseconds, the SQLite catalog waits on another writer's lock.
_BUSY_MS = 30000

# The homes a read was seen to answer from, once per process, and how long one may take.
_ANSWERED: set[Path] = set()
_ANSWER_SECONDS = 10.0


class Access(StrEnum):
    """What an attach may do: read, write, create a lake that is not there, or migrate one."""

    READ = auto()
    WRITE = auto()
    CREATE = auto()
    MIGRATE = auto()


# The attach options each access adds: a write journals the catalog in WAL mode, creation and
# migration are writes too, and a read refuses every write.
_ACCESS_OPTIONS = {
    Access.READ: ("CREATE_IF_NOT_EXISTS false", "READ_ONLY"),
    Access.WRITE: ("CREATE_IF_NOT_EXISTS false", "META_JOURNAL_MODE 'WAL'"),
    Access.CREATE: ("CREATE_IF_NOT_EXISTS true", "META_JOURNAL_MODE 'WAL'"),
    Access.MIGRATE: (
        "CREATE_IF_NOT_EXISTS false",
        "AUTOMATIC_MIGRATION true",
        "META_JOURNAL_MODE 'WAL'",
    ),
}


class Attachment(FrozenModel):
    """One workspace's state lake as files and the attach that reaches them.

    root: the workspace root; the lock and token paths derive from it.
    home: where the catalog and data live, the workspace's `[workspace] lake` (see `Lake.at`).
    extensions: where DuckDB's extensions are loaded from and installed to.
    served: the `quack:host:port` a lake's writer is served at, attached instead of the workspace's
        own files when set; `MB_LAKE` by default, else `MB_LAKE_READ` on a host that only reads.
        Writes attach `MB_LAKE` with `MB_LAKE_TOKEN`, reads the read-only endpoint `MB_LAKE_READ`
        with `MB_LAKE_READ_TOKEN`.
    """

    root: Path
    home: Path
    extensions: Path = Field(default_factory=extension_home)
    served: str = Field(
        default_factory=lambda: (
            Project().variable("LAKE").read() or Project().variable("LAKE_READ").read()
        )
    )

    @contextmanager
    def attached(self, *, access: Access) -> Generator[duckdb.DuckDBPyConnection]:
        """A fresh in-memory DuckDB with the lake attached, detached and closed on the way out.

        A failure skips the detach: closing the connection drops the attachment and rolls back
        whatever transaction the failure left open. Raises MissionError before any attach when
        the lake is not being created and was never created.
        """
        if access is not Access.CREATE and not self.exists():
            raise MissionError(
                f"no state lake at {self.catalog}; the first command recording state creates it"
            )
        with closing(
            connect(
                config={"autoinstall_known_extensions": False, "autoload_known_extensions": False}
            )
        ) as connection:
            self.attach(connection, access=access)
            yield connection
            connection.execute("USE memory")
            connection.execute(f"DETACH {ALIAS}")

    @contextmanager
    def open(self, *, write: bool = False) -> Generator[duckdb.DuckDBPyConnection]:
        """A fresh connection with the lake attached as `lake`, detached and closed after.

        write: attach read-write; a read-only attach refuses every write.
        Raises MissionError when the lake was never created.
        """
        access = Access.WRITE if write else Access.READ
        with FileBudget.permitted(), self.attached(access=access) as connection:
            yield connection

    @property
    def catalog(self) -> Path:
        return self.home / "lake.sqlite"

    @property
    def central(self) -> bool:
        """Whether this is the center's lake, the home of the workspace's evidence.

        It is when its checkout holds the repository, which a dispatch mirror ships without (see
        `Header.lake_home`).
        """
        return (self.root / ".git").exists()

    @property
    def data(self) -> Path:
        return self.home / "lake"

    @property
    def lock(self) -> Path:
        """The maintenance lock, which creation also takes and an append never does."""
        return self.out / "run" / "lake.maint.lock"

    @property
    def out(self) -> Path:
        """The workspace's generated-state directory, `.mb` or a legacy name."""
        return Project().out(self.root)

    @property
    def read_token(self) -> Path:
        """Where `serving` keeps the token of the read-only endpoint."""
        return self.out / "run" / "lake.read.token"

    @property
    def token(self) -> Path:
        """Where `serving` keeps the token it hands clients, so a restart keeps theirs valid."""
        return self.out / "run" / "lake.token"

    @staticmethod
    def _mismatched(fault: duckdb.InvalidInputException) -> Exception:
        """Why an attach was refused: a catalog newer than this DuckDB says how to upgrade."""
        if "catalog version mismatch" not in str(fault):
            return fault
        said = first_line(fault).split(": ", 1)[-1].split(" To automatically", 1)[0]
        return MissionError(
            f"{said} Run `mb lake upgrade` on the machine holding the lake, after which "
            "older DuckDB releases cannot open it."
        )

    @staticmethod
    def _unreached(fault: duckdb.Error) -> str:
        """Why a served lake could not be attached, in the words that tell what to do."""
        said = first_line(fault)
        if "deserialize" in said.lower():
            return (
                f"it answers in a protocol DuckDB {duckdb.__version__} here cannot read; "
                "both ends must run the same DuckDB release"
            )
        if "connect" in said.lower():
            return "nothing answers there; start `mb lake serve` and the ssh tunnel to it"
        return said

    def _answering(self) -> None:
        """Raise MissionError unless a read of the lake's home returns, once per home and process.

        macOS privacy leaves a read on an external volume waiting forever, never refusing it, in
        a process whose responsible app lacks Full Disk Access (a tmux server, a launchd job),
        while `stat` still answers.
        """
        if self.home in _ANSWERED:
            return
        probe = self.catalog if self.catalog.is_file() else self.home
        answered = Event()

        def read() -> None:
            with suppress(OSError):
                if probe.is_dir():
                    next(probe.iterdir(), None)
                else:
                    with probe.open("rb") as held:
                        held.read(1)
            answered.set()

        Thread(target=read, daemon=True).start()
        if not answered.wait(_ANSWER_SECONDS):
            raise MissionError(
                f"the lake at {self.home} gave no answer in {_ANSWER_SECONDS:.0f} s; on macOS, "
                "run this through `ssh localhost`, whose sessions hold Full Disk Access when "
                "Remote Login allows it, or grant it to the app this process runs under (under "
                "tmux, the tmux binary) and restart that app"
            )
        _ANSWERED.add(self.home)

    def _held(self, verb: str) -> None:
        """Refuse `verb` on a served lake: it works on the files, on the machine holding them."""
        if self.served:
            raise MissionError(
                f"the lake is served from {self.served}; `{verb}` works on its files, so run it "
                "on the machine holding them"
            )

    def _load(self, connection: duckdb.DuckDBPyConnection, names: Iterable[str]) -> None:
        load(connection, list(names), self.extensions)

    def _reach(self, connection: duckdb.DuckDBPyConnection, access: Access) -> None:
        """Attach the served lake as `lake`: reads at the read-only endpoint, writes at the writer.

        Saying plainly why when it cannot be reached.
        """
        reading = access is Access.READ
        uri = Project().variable("LAKE_READ" if reading else "LAKE").read()
        if not uri:
            only = "for writing only" if reading else "for reading only"
            need = (
                "reading needs MB_LAKE_READ and MB_LAKE_READ_TOKEN"
                if reading
                else "writing needs MB_LAKE and MB_LAKE_TOKEN (`mb lake serve` names them)"
            )
            raise MissionError(f"the lake is served from {self.served} {only}; {need}")
        self._load(connection, QUACK)
        token = Project().variable("LAKE_READ_TOKEN" if reading else "LAKE_TOKEN").read()
        options = f" (TOKEN {quoted(token)})" if token else ""
        try:
            connection.execute(f"ATTACH {quoted(uri)} AS {ALIAS}{options}")
        except duckdb.Error as fault:
            raise MissionError(
                f"cannot reach the lake at {uri}: {Attachment._unreached(fault)}"
            ) from fault

    def attach(
        self, connection: duckdb.DuckDBPyConnection, *, access: Access = Access.READ
    ) -> None:
        """Attach this lake to `connection` as `lake`, read-only unless `access` writes.

        A query that already has its own views (`mb query`) then reaches every table beside them.

        A served lake is attached over Quack instead: a read at the read-only endpoint, any other
        access at the writer.
        """
        connection.execute("SET enable_progress_bar = false")
        connection.execute("SET TimeZone = 'UTC'")
        if self.served:
            self._reach(connection, access)
            return
        self._answering()
        self._load(connection, _EXTENSIONS)
        connection.execute(f"SET ducklake_max_retry_count = {RETRIES}")
        options = (
            f"DATA_PATH {quoted(self.data.as_posix() + '/')}",
            "OVERRIDE_DATA_PATH true",
            f"META_BUSY_TIMEOUT {_BUSY_MS}",
            *_ACCESS_OPTIONS[access],
        )
        target = quoted(f"ducklake:sqlite:{self.catalog.as_posix()}")
        try:
            connection.execute(f"ATTACH {target} AS {ALIAS} ({', '.join(options)})")
        except duckdb.InvalidInputException as fault:
            raise Attachment._mismatched(fault) from fault

    def execute(
        self,
        connection: duckdb.DuckDBPyConnection,
        statement: ClauseElement | str,
        parameters: Iterable[object] = (),
    ) -> duckdb.DuckDBPyConnection:
        """`statement` run on `connection`, inlined for a served lake.

        It is a built statement compiled, or SQL with `parameters` bound.
        """
        sql, bound = (
            (statement, list(parameters)) if isinstance(statement, str) else compiled(statement)
        )
        if self.served and bound:
            return connection.execute(inlined(sql, bound))
        return connection.execute(sql, bound)

    def exists(self) -> bool:
        return bool(self.served) or self.catalog.is_file()

    def query(
        self, statement: ClauseElement | str, parameters: Iterable[object] = ()
    ) -> list[tuple[Any, ...]]:
        """One read-only query's rows; tables and views are named `lake.<name>`."""
        with self.open() as connection:
            return self.execute(connection, statement, parameters).fetchall()
