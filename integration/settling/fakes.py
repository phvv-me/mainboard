from collections.abc import Generator
from contextlib import contextmanager

from plumbum import local
from plumbum.machines.local import LocalMachine

from mainboard.dispatch.schedulers import HostUnreachable


class Connections:
    """The ssh seam of the relay: every host is this machine, and each ask is noted.

    answering: whether a host answers; one that does not raises what a dead link raises.
    asked: the hosts the relay connected to, in order.
    """

    @contextmanager
    def __call__(self, host: str) -> Generator[LocalMachine]:
        self.asked.append(host)
        if not self.answering:
            raise HostUnreachable(f"no answer from {host}")
        yield local

    def __init__(self, *, answering: bool = True) -> None:
        self.answering = answering
        self.asked: list[str] = []
