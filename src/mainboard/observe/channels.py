# How a caller fetches a job's frames off a host: a cheap batched poll or a long-lived stream,
# picked per `Observe.channel` and cascaded through patos `Strategy` when it says `auto`. Every
# channel takes its transport as an injected runner, so nothing here opens ssh itself.

import json
from typing import TYPE_CHECKING, Protocol, cast

from patos import Strategy

from .frames import Frame, decode, next_offset, parse_tail

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from patos import Resolution, Shared

    from ..dispatch.transport import Machine

_AUTO = "auto"
_POLL = "ssh-poll"
_STREAM = "stream"


class PollRunner(Protocol):
    """One blocking shell round-trip against a host, injected so no channel opens ssh itself."""

    def __call__(self, host: str, *, command: str) -> str:
        """Run `command` on `host` and return its captured stdout."""


class StreamRunner(Protocol):
    """One long-lived shell exec against a host, injected so no channel opens ssh itself."""

    def __call__(self, host: str, *, command: str) -> Iterator[str]:
        """Run `command` on `host`, yielding each stdout line as it arrives."""


class Channel(Protocol):
    """One way to fetch a job's frames from a host, polled or long-lived streamed."""

    def available(self) -> bool:
        """Whether this channel can be selected right now."""

    def fetch(self, host: str, *, job: str, offset: int) -> tuple[list[Frame], int]:
        """New frames from `offset` onward, plus the offset to resume from next."""


class PollChannel:
    """Fetches new frames with two batched round-trips: `cat status.json` then `tail -c +N`.

    Only ever sees the live segment, so a caller far enough behind to have fallen past a roll
    resyncs from the live segment's own start; the durable store is what never loses history,
    a shell-only poll is deliberately just a cheap live tail.
    """

    def __init__(self, root: str, runner: PollRunner) -> None:
        self.root = root
        self.runner = runner

    def available(self) -> bool:
        return bool(self.root)

    def fetch(self, host: str, *, job: str, offset: int) -> tuple[list[Frame], int]:
        job_dir = f"{self.root}/{job}"
        status = json.loads(self.runner(host, command=f"cat {job_dir}/status.json"))
        base = int(status["base"])
        local = max(0, offset - base)
        text = self.runner(host, command=f"tail -c +{local + 1} {job_dir}/live.ndjson")
        frames = parse_tail(text)
        return frames, next_offset(frames, default=max(offset, base))


class StreamChannel:
    """Fetches new frames from one long-lived `agentmain --follow` exec on the host."""

    def __init__(self, root: str, runner: StreamRunner) -> None:
        self.root = root
        self.runner = runner

    def available(self) -> bool:
        return bool(self.root)

    def fetch(self, host: str, *, job: str, offset: int) -> tuple[list[Frame], int]:
        command = (
            f"python -m mainboard.observe.agentmain --root {self.root} --job {job} "
            f"--follow --from-offset {offset}"
        )
        frames = [decode(line) for line in self.runner(host, command=command) if line.strip()]
        return frames, next_offset(frames, default=offset)


class Channels:
    """The `ssh-poll` and `stream` channels, selected by name or cascaded on `auto`.

    `resolve` returns the winner; a cascaded resolution also lands on `last_resolution`, so a
    caller can render why a host fell to its fallback channel instead of degrading silently.
    """

    def __init__(self, root: str, poll: PollRunner, stream: StreamRunner) -> None:
        self.strategy: Strategy[Channel] = Strategy("observe channel")
        self.strategy.register(_POLL, PollChannel(root, poll))
        self.strategy.register(_STREAM, StreamChannel(root, stream))
        self.last_resolution: Resolution[Channel] | None = None

    def resolve(self, channel: str) -> Channel:
        """The channel implementation for `channel`, cascading `ssh-poll` then `stream` on `auto`.

        channel: an `Observe.channel` value, a registered name or `auto`.
        """
        if channel == _AUTO:
            self.last_resolution = self.strategy.cascade()
            return cast("Channel", self.last_resolution.implementation)
        return self.strategy.select(channel)


def cached(shared: Shared[str, Machine], execute: Callable[[Machine, str], str]) -> PollRunner:
    """A `PollRunner` reusing one connection per host via `shared`, run through `execute`.

    shared: caches one live connection per host, closing it once every holder releases it.
    execute: runs one command over an already-open connection and returns its stdout.
    """

    def run(host: str, *, command: str) -> str:
        with shared.acquire(host) as machine:
            return execute(machine, command)

    return run


def cached_stream(
    shared: Shared[str, Machine], execute: Callable[[Machine, str], Iterator[str]]
) -> StreamRunner:
    """A `StreamRunner` reusing one connection per host via `shared`, run through `execute`.

    shared: caches one live connection per host, closing it once every holder releases it.
    execute: execs one long-lived command over an already-open connection, yielding its lines.
    """

    def run(host: str, *, command: str) -> Iterator[str]:
        with shared.acquire(host) as machine:
            yield from execute(machine, command)

    return run
