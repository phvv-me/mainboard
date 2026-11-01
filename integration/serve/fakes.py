import json
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingUnixStreamServer
from types import TracebackType
from typing import Self

from mainboard.engines.compile.ecosystems import Binaries
from mainboard.engines.compile.ecosystems.archive import Receipt, sha256_of
from mainboard.manifest.render.interpolate import Json
from mainboard.serve.compose import Programs

# One request the fake sink received: its path, its Authorization header and its body.
type Posted = tuple[str, str, str]

_ARCHIVE = "https://github.com/example/duckdb-otlp/releases/duckdb-otlp.tar.gz"
# What the sink answers a GET with: its health check, which is read by status alone.
_HEALTH: Mapping[str, Json] = {"/healthz": {}}


def _executable(directory: Path, name: str, *, body: str) -> Path:
    """A shell script `name` in `directory` that runs `body`, standing in for a real program."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def programs(root: Path, *, supervisor: str = "") -> Programs:
    """Programs over a stand-in `mainboard` in an environment `bin/` under `root`.

    A process-compose running `supervisor` joins it when one is given.
    """
    environment = root / "env" / "bin"
    tool = _executable(environment, "mainboard", body="#!/bin/sh\n")
    if supervisor:
        _executable(environment, "process-compose", body=supervisor)
    return Programs(Binaries(root / "installed", environment=environment), tool=tool)


def binaries(root: Path, duckdb: str = "1.5.6", *, pinned: bool = True) -> Binaries:
    """A tool resolver whose duckdb-otlp prints its version as 0.7.3 does, embedding `duckdb`.

    It is the install of its pinned archive, receipt beside it, when `pinned`, else a same-version
    twin in the environment's `bin/`. An empty `duckdb` leaves no duckdb-otlp at all.
    """
    installed = root / "installed"
    environment = root / "env" / "bin"
    environment.mkdir(parents=True, exist_ok=True)
    if duckdb:
        said = f"duckdb-otlp eb83de3 (DuckDB v{duckdb})"
        body = f"#!/bin/sh\necho '{said}'\n"
        script = _executable(
            installed / "bin" if pinned else environment, "duckdb-otlp", body=body
        )
        if pinned:
            receipt = Receipt(
                version="0.7.3",
                url=_ARCHIVE,
                sha256="a" * 64,
                member="duckdb-otlp",
                executable_sha256=sha256_of(script),
            )
            (installed / "duckdb-otlp.json").write_text(
                receipt.model_dump_json(), encoding="utf-8"
            )
    return Binaries(installed, environment=environment)


@contextmanager
def answering(path: Path, routes: Mapping[str, Json]) -> Generator[None]:
    """process-compose's REST API as a double while the block runs, on a unix socket at `path`.

    routes: a path to the body it answers 200 with; any other path answers 404.
    """
    with FakeSink(path, routes=routes):
        yield


class FakeSink(ThreadingUnixStreamServer):
    """The sink's HTTP face: its health check, and every POST recorded, on a unix socket.

    The gate refuses every TCP endpoint on macOS, so the double carries the same HTTP over a
    unix socket at `path` and the `reach` fixture's sink reaches it there. Each POST is answered
    with `status`, 202 as the real sink accepts a record, until `answer` changes it.

    routes: a path to the JSON body a GET of it answers 200 with; any other path answers 404.
        Other read-only APIs, such as process-compose's, are this double with their routes.
    """

    class _Answer(BaseHTTPRequestHandler):
        def address_string(self) -> str:
            return "unix"

        def do_GET(self) -> None:
            assert isinstance(self.server, FakeSink)
            found = self.server.routes.get(self.path)
            body = b"" if found is None else json.dumps(found).encode()
            self.send_response(404 if found is None else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            assert isinstance(self.server, FakeSink)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
            self.server.posted.append((self.path, self.headers.get("Authorization") or "", body))
            self.send_response(self.server.status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: str) -> None:
            del format, args

    @property
    def bodies(self) -> list[str]:
        """The body of every POST, in the order they came."""
        return [body for _, _, body in self.posted]

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.shutdown()
        self.server_close()
        self.path.unlink(missing_ok=True)

    def __init__(self, path: Path, *, routes: Mapping[str, Json] = _HEALTH) -> None:
        super().__init__(str(path), FakeSink._Answer)
        self.path = path
        self.routes = routes
        self.posted: list[Posted] = []
        self.status = 202
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    def answer(self, status: int) -> None:
        """Answer every later POST with `status`."""
        self.status = status
