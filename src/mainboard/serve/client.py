"""process-compose's REST API, read over its unix socket: whether it answers, what it runs.

Only reads: `serve stop` goes through the service manager, never through this API, which can also
start processes. Every request is bounded, so a wedged supervisor is an answer, not a hang.
"""

import socket
from datetime import datetime
from http.client import HTTPConnection, HTTPException
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict, Field, ValidationError

from ..core.errors import MissionError

if TYPE_CHECKING:
    from pathlib import Path

# How long one request may take before the supervisor counts as not answering.
_TIMEOUT = 2.0


class Process(FrozenModel):
    """One child as process-compose reports it, the fields `serve status` reads.

    exit_code: the last run's exit status. A resident child past its restarts reads `Completed`
        with a nonzero one, and a pass waiting for its next run reads `Scheduled` after a failed
        one, so this, never `status`, says whether it failed.
    next_run_time: when a pass runs next, None for a resident child.
    """

    model_config = ConfigDict(extra="ignore")

    exit_code: int = 0
    is_running: bool = False
    name: str
    next_run_time: datetime | None = None
    pid: int = 0
    restarts: int = 0
    status: str


class ProjectState(FrozenModel):
    """The supervisor itself as it reports itself at `/project/state`.

    up_time: nanoseconds since it started.
    """

    model_config = ConfigDict(extra="ignore")

    up_time: int = Field(alias="upTime")
    version: str


class _Listing(FrozenModel):
    model_config = ConfigDict(extra="ignore")

    data: list[Process]


class _Unix(HTTPConnection):
    """An HTTP connection carried by a unix socket instead of TCP."""

    def __init__(self, path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))

    def get(self, route: str) -> tuple[int, bytes]:
        """The status and the whole body of one GET `route`."""
        self.request("GET", route)
        response = self.getresponse()
        return response.status, response.read()


class Unanswered(MissionError):
    """The supervisor did not answer: no socket, a refused or timed-out connection, a bad reply."""


class Client:
    """Reads process-compose's API at `socket`, each request bounded by `timeout` seconds."""

    def __init__(self, socket: Path, timeout: float = _TIMEOUT) -> None:
        self.socket = socket
        self.timeout = timeout

    def _get(self, route: str) -> bytes:
        """The body of a 200 answer to GET `route`; Unanswered for anything else."""
        connection = _Unix(self.socket, self.timeout)
        try:
            status, body = connection.get(route)
        except (OSError, HTTPException) as fault:
            raise Unanswered(f"{self.socket}: {fault}") from fault
        finally:
            connection.close()
        if status != 200:
            raise Unanswered(f"{self.socket}: GET {route} answered {status}")
        return body

    def alive(self) -> bool:
        """Whether the supervisor answers its liveness route."""
        try:
            self._get("/live")
        except Unanswered:
            return False
        return True

    def processes(self) -> list[Process]:
        """Every child the supervisor holds, as it reports them."""
        try:
            return _Listing.model_validate_json(self._get("/processes")).data
        except ValidationError as fault:
            raise Unanswered(f"{self.socket}: /processes is not what it was: {fault}") from fault

    def project(self) -> ProjectState:
        """The supervisor's own version and age."""
        try:
            return ProjectState.model_validate_json(self._get("/project/state"))
        except ValidationError as fault:
            raise Unanswered(
                f"{self.socket}: /project/state is not what it was: {fault}"
            ) from fault
