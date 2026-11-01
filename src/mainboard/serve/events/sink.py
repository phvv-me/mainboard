"""The `sink` child: duckdb-otlp's standalone binary, the one writer of the telemetry lake.

`mb serve process sink` writes the binary's start-up SQL and becomes the binary, so the process the
supervisor watches and stops is the sink itself, and SIGTERM lets it flush what it holds. It
listens on loopback alone, at the OTLP/HTTP port every exporter defaults to, with gRPC off. It
attaches only the telemetry lake (`state.telemetry`), creating it on the first start at the
DuckLake spec its own DuckDB writes, and its data, control database, extensions and spill all live
in that lake's directory, so it reads no DuckDB secret and leaves nothing in the home directory.

Its token allows appends and nothing else: the binary serves `POST /v1/{logs,traces,metrics}` and
an unauthenticated `GET /healthz`, no SQL. The token is made once, kept in `.mb/serve/otlp.token`
at 0600 and handed to the binary through its environment, never its arguments, and the sink never
starts without it: the binary runs anonymous on loopback when no token is given. The environment
is the supervisor's six-name allowlist and the sink's own settings, so no `.env` secret reaches
it even when it is started from an activated shell.

Memory is capped where the binary spends it, its buffered rows and seal sizes (see `_CAPS`), with
DuckDB's own limit beside them, which alone does not bound it.

Two fallbacks stand behind this seam if the binary cannot be kept: ccusage-style reading of the
agents' own transcripts (`~/.claude/projects`, `~/.codex/sessions`), which recovers token usage
and nothing else, or otelcol's file exporter with DuckDB's `read_json` over what it writes. The
doctor's `telemetry` row names the first while no pairing is proven.
"""

import http.client
import os
import re
import secrets
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import duckdb
from packaging.version import Version
from patos import FrozenModel

from ...core.errors import MissionError
from ...core.execution.shell import become
from ...core.execution.tool import ToolRunner
from ...core.project import Project
from ...core.section import Section
from ...enums import Source, Verdict
from ...state.storage.database import quoted
from ...state.telemetry import TelemetryLake
from ..compose import Compose
from ..keeper import Layout

if TYPE_CHECKING:
    from ...engines.compile.ecosystems import Binary
    from ...engines.compile.ecosystems.archive import Binaries, Found

_PROGRAM = "duckdb-otlp"
_HOST = "127.0.0.1"
_PORT = 4318
# The binary refuses a shorter token; this one is 43 URL-safe characters.
_SHORTEST = 16
_TOKEN_BYTES = 32
# The binary's own memory caps. Its resident memory follows the rows it buffers and never shrinks,
# so these bound what a flood can take: 4 sending threads at 907,000 rows a second peaked at
# 1.55 GiB with them and at 4.7 GiB with spike-otlp's 16 and 8 MiB (2026-10-29); past the buffer
# the sink answers 503, which an exporter retries.
_CAPS = {
    "DUCKDB_OTLP_MAX_BUFFERED_BYTES": str(4 << 20),
    "DUCKDB_OTLP_SEAL_TARGET_BYTES": str(2 << 20),
    "DUCKDB_OTLP_HTTP_THREADS": "2",
}
_MEMORY = "512MB"
_SPILL = "1GB"
_THREADS = 2
_EXTENSIONS_SQL = "INSTALL ducklake; LOAD ducklake; INSTALL sqlite; LOAD sqlite;"
# How long a look at the sink's health may take, and how often a wait looks again.
_ANSWER_SECONDS = 2.0
_POLL = 0.25
_EMBEDDED = re.compile(r"DuckDB v(\d+\.\d+\.\d+)")
_FALLBACK = (
    "or read usage from the transcripts (~/.claude/projects, ~/.codex/sessions; ccusage-style) "
    "until a pairing is proven"
)


class Pairing(FrozenModel):
    """A sink release proven against this tool's DuckDB.

    sink: the DuckDB the binary embeds, as its `version` prints it.
    spec: the DuckLake spec that DuckDB writes the lake at.
    reader: the newest DuckDB release line (`2.0`) proven to read that lake READ_ONLY.
    """

    reader: str
    sink: str
    spec: str


# Each one proven by a READ_ONLY attach of a lake the binary wrote: spike-otlp (2026-10-28, DuckDB
# 2.0.0 reading duckdb-otlp 0.7.3's lake while it ingested) and v2-events (2026-10-29). A sink
# release or a DuckDB release line outside this table fails the doctor until it is proven here.
_PAIRINGS = (Pairing(sink="1.5.6", spec="1.0", reader="2.0"),)


class Token:
    """The sink's bearer token, in a 0600 file of its own."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def ensure(self) -> str:
        """The token, made first when there is none: anyone may make it, but only once."""
        try:
            return self.read()
        except FileNotFoundError:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self.read()
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(_TOKEN_BYTES))
        return self.read()

    def read(self) -> str:
        """The token, refused when another user may read it or it is too short to be one."""
        if (mode := self.path.stat().st_mode & 0o777) & 0o077:
            raise MissionError(
                f"{self.path} is readable beyond its owner (mode {mode:o}); delete it, and "
                "the sink makes a new one"
            )
        token = self.path.read_text(encoding="utf-8").strip()
        if len(token) < _SHORTEST:
            raise MissionError(
                f"{self.path} holds {len(token)} characters, under the {_SHORTEST} a token needs"
            )
        return token


class Sink:
    """One workspace's sink: how it starts, where it answers, and whether its lake stays readable.

    root: the workspace served.
    port: its loopback OTLP/HTTP port.
    """

    @property
    def endpoint(self) -> str:
        """Where an exporter sends its logs."""
        return f"http://{_HOST}:{self.port}/v1/logs"

    @staticmethod
    def _asked(connection: http.client.HTTPConnection, path: str) -> int:
        """The status `connection` answers a GET of `path` with."""
        connection.request("GET", path)
        return connection.getresponse().status

    @staticmethod
    def _embedded(found: Found) -> str:
        """The DuckDB the binary at `found` embeds, empty when its `version` names none."""
        said = ToolRunner(timeout=10, max_bytes=4 << 10).run(str(found.path), ["version"])
        matched = _EMBEDDED.search(said.stdout + said.stderr)
        return matched[1] if matched else ""

    def __init__(self, root: Path, *, port: int = _PORT) -> None:
        self.root = root
        self.port = port
        self.lake = TelemetryLake(root=root)
        self.token = Token(Layout(Project().out(root) / "serve").state / "otlp.token")

    def _paired(self, found: Found, pairing: Pairing) -> Section:
        """The row once the binary is paired: the lake still at its spec and readable here."""
        versions = f"{_PROGRAM} {found.version} (DuckDB {pairing.sink})"
        reader = f"read by DuckDB {duckdb.__version__}"
        if not self.lake.attachment.exists():
            return Section(
                section="telemetry",
                verdict=Verdict.PASS,
                detail=f"{versions}, {reader}; no lake yet, the sink creates it",
            )
        try:
            spec = self.lake.spec()
        except (duckdb.Error, MissionError) as fault:
            return Section(
                section="telemetry",
                verdict=Verdict.FAIL,
                detail=f"DuckDB {duckdb.__version__} cannot read the telemetry lake: {fault}",
                fix=f"move {_PROGRAM} to a release on a DuckDB this one reads, {_FALLBACK}",
            )
        if spec != pairing.spec:
            return Section(
                section="telemetry",
                verdict=Verdict.FAIL,
                detail=(
                    f"the telemetry lake is at DuckLake {spec}, but {versions} writes "
                    f"{pairing.spec} and refuses it: a newer DuckDB migrated it"
                ),
                fix=f"move {self.lake.home} aside; the sink starts a new lake",
            )
        detail = f"{versions}, DuckLake {spec}, {reader}"
        return Section(section="telemetry", verdict=Verdict.PASS, detail=detail)

    def answering(self) -> bool:
        """Whether the sink answers its health check now."""
        with closing(self.connection(_ANSWER_SECONDS)) as connection:
            try:
                status = Sink._asked(connection, "/healthz")
            except OSError:
                return False
        return status == http.client.OK

    def awaited(self, seconds: float) -> bool:
        """Whether the sink answers within `seconds`, looking again every quarter second."""
        deadline = time.monotonic() + seconds
        while not self.answering():
            if time.monotonic() > deadline:
                return False
            time.sleep(_POLL)
        return True

    def command(self, program: Path) -> list[str]:
        """The binary's argv: loopback, HTTP alone, its own files in the lake's directory."""
        home = self.lake.home
        return [
            str(program),
            "serve",
            *("--mode", "none", "--init-sql", str(home / "sink.sql")),
            *("--catalog", "lake", "--schema", "main"),
            *("--host", _HOST, "--http", str(self.port), "--grpc", "0"),
            *("--data-dir", str(home), "--secret-dir", str(home / "secrets")),
            *("--database", str(home / "control.duckdb")),
        ]

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        """An HTTP connection to the sink's port, opened by its first request."""
        return http.client.HTTPConnection(_HOST, self.port, timeout=timeout)

    def environment(self, token: str) -> dict[str, str]:
        """The binary's whole environment: the supervisor's allowlist, its caps and its token."""
        return Compose.allowed() | _CAPS | {"DUCKDB_OTLP_TOKEN": token}

    def script(self) -> str:
        """The SQL the binary runs before it listens: its budget, then the lake attached.

        Its extensions are installed into the lake's directory on the first start, which needs
        the network once, and loaded from there after.
        """
        home = self.lake.home
        catalog = quoted(f"ducklake:sqlite:{self.lake.attachment.catalog}")
        options = ", ".join(
            (
                f"DATA_PATH {quoted(f'{self.lake.attachment.data}/')}",
                "OVERRIDE_DATA_PATH true",
                "META_JOURNAL_MODE 'WAL'",
                "META_BUSY_TIMEOUT 30000",
            )
        )
        return "\n".join(
            (
                f"SET extension_directory = {quoted(str(home / 'extensions'))};",
                f"SET memory_limit = '{_MEMORY}';",
                f"SET threads = {_THREADS};",
                f"SET temp_directory = {quoted(str(home / 'spill'))};",
                f"SET max_temp_directory_size = '{_SPILL}';",
                _EXTENSIONS_SQL,
                f"ATTACH {catalog} AS lake ({options});",
                "CREATE SCHEMA IF NOT EXISTS lake.main;",
                "",
            )
        )

    def section(self, binaries: Binaries) -> Section:
        """The doctor's `telemetry` row: whether this DuckDB still reads what the sink writes.

        It fails loudly when a DuckDB upgrade outran the sink: this environment's DuckDB is a
        release line newer than any proven against the binary's, the binary embeds a DuckDB no
        pairing names, or the lake is at a spec the binary no longer writes. It warns while the
        only duckdb-otlp here is not the install of its pinned archive, which `start` refuses.
        """
        found = binaries.find(_PROGRAM)
        if found is None:
            return Section(
                section="telemetry",
                verdict=Verdict.WARN,
                detail=f"{_PROGRAM} is not installed here, so the sink cannot start",
                fix=f"{Project().name} dep binary",
            )
        if found.source is not Source.INSTALLED:
            return Section(
                section="telemetry",
                verdict=Verdict.WARN,
                detail=(
                    f"{_PROGRAM} at {found.path} ({found.source}) is not the install of its "
                    "pinned archive, so the sink will not start it"
                ),
                fix=f"{Project().name} dep binary",
            )
        embedded = Sink._embedded(found)
        pairing = next((pairing for pairing in _PAIRINGS if pairing.sink == embedded), None)
        if pairing is None:
            return Section(
                section="telemetry",
                verdict=Verdict.FAIL,
                detail=(
                    f"{_PROGRAM} at {found.path} embeds DuckDB {embedded or '(unnamed)'}, which "
                    "no proven pairing names"
                ),
                fix=f"prove a READ_ONLY read of its lake, add it to _PAIRINGS, {_FALLBACK}",
            )
        if Version(duckdb.__version__).release[:2] > Version(pairing.reader).release[:2]:
            return Section(
                section="telemetry",
                verdict=Verdict.FAIL,
                detail=(
                    f"DuckDB {duckdb.__version__} here is newer than {pairing.reader}, the newest "
                    f"line proven to read the lake {_PROGRAM} writes (DuckDB {pairing.sink}, "
                    f"DuckLake {pairing.spec}): a DuckDB upgrade outran the sink"
                ),
                fix=f"prove a READ_ONLY read of its lake, or move the sink on, {_FALLBACK}",
            )
        return self._paired(found, pairing)

    def start(self, binary: Binary) -> NoReturn:
        """Become the sink: its pinned install only, its token made if missing, its SQL written."""
        program = binary.resolver().pinned(_PROGRAM).path
        token = self.token.ensure()
        home = self.lake.home
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        home.chmod(0o700)
        script = home / "sink.sql"
        script.write_text(self.script(), encoding="utf-8", newline="\n")
        script.chmod(0o600)
        os.umask(0o077)
        become(str(program), self.command(program), env=self.environment(token))
