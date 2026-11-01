import http.client
import socket
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mainboard.serve.events.sink import Sink

from ..fakes import FakeSink


class UnixSink(Sink):
    """The workspace's sink, reached at the fake's unix socket instead of its loopback port."""

    class _Connection(http.client.HTTPConnection):
        def __init__(self, path: Path, timeout: float) -> None:
            super().__init__("localhost", timeout=timeout)
            self.path = path

        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(str(self.path))

    def __init__(self, root: Path, unix: Path) -> None:
        super().__init__(root)
        self.unix = unix

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        return UnixSink._Connection(self.unix, timeout)


@pytest.fixture
def fake(tmp_path: Path) -> Iterator[FakeSink]:
    """A fake sink answering on a unix socket of its own."""
    with FakeSink(tmp_path / "sink.sock") as answering:
        yield answering


@pytest.fixture
def reach() -> Callable[[Path, Path], Sink]:
    """The workspace's sink at a root, reached at a unix socket instead of its loopback port."""
    return UnixSink


@pytest.fixture
def sink(workspace: Path, fake: FakeSink) -> Sink:
    """The workspace's sink, reached at the fake, its token made."""
    served = UnixSink(workspace, fake.path)
    served.token.ensure()
    return served
