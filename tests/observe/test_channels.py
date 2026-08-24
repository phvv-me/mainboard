import json
from collections.abc import Iterator, Sequence

import pytest
from patos import Shared, StrategyError

from mainboard.observe import Channels, PollChannel, StreamChannel, encode, encoded_length
from mainboard.observe.channels import cached, cached_stream

from .support import line

_BODY = encode(line(0, "a")) + encode(line(50, "b"))
_FOLLOW = "python -m mainboard.observe.agentmain --root /spool --job job1 --follow --from-offset"


class Polling:
    """A poll runner answering `cat` with a scripted heartbeat and `tail` with a fixed body."""

    def __init__(self, base: int, body: str = "") -> None:
        self.base = base
        self.body = body
        self.commands: list[str] = []

    def __call__(self, host: str, *, command: str) -> str:
        self.commands.append(command)
        if command.startswith("cat"):
            return json.dumps(
                {"state": "running", "offset": 100, "base": self.base, "updated_at": "t"}
            )
        return self.body


def _channels() -> Channels:
    """The two-channel roster over a configured root, with both transports stubbed out."""
    return Channels("/spool", lambda host, command: "", lambda host, command: iter(()))


@pytest.mark.parametrize(("root", "available"), [("/spool", True), ("", False)])
def test_a_channel_is_available_only_with_a_configured_root(root: str, available: bool) -> None:
    """A host with no spool root has nowhere to read from, whichever transport would carry it."""
    assert PollChannel(root, lambda host, command: "").available() is available
    assert StreamChannel(root, lambda host, command: iter(())).available() is available


@pytest.mark.parametrize(
    ("base", "offset", "body", "start", "texts", "resume"),
    [
        (0, 0, _BODY, 1, ["a", "b"], 50 + encoded_length(line(50, "b"))),
        (40, 50, "", 11, [], 50),
        (40, 0, "", 1, [], 40),
    ],
)
def test_poll_reads_the_status_then_a_tail_translated_against_the_live_base(
    base: int, offset: int, body: str, start: int, texts: list[str], resume: int
) -> None:
    """A caller behind the live base resyncs from it, which is all a rolled segment costs here."""
    runner = Polling(base, body)
    frames, next_at = PollChannel("/spool", runner).fetch("gold", job="job1", offset=offset)
    assert [frame.payload["text"] for frame in frames] == texts
    assert next_at == resume
    assert runner.commands == [
        "cat /spool/job1/status.json",
        f"tail -c +{start} /spool/job1/live.ndjson",
    ]


@pytest.mark.parametrize(
    ("offset", "lines", "texts", "resume"),
    [
        (
            0,
            (encode(line(0, "a")), "\n", encode(line(30, "b"))),
            ["a", "b"],
            30 + encoded_length(line(30, "b")),
        ),
        (7, (), [], 7),
    ],
)
def test_stream_runs_one_follow_exec_and_skips_a_blank_line(
    offset: int, lines: Sequence[str], texts: list[str], resume: int
) -> None:
    """One long-lived exec carries the whole batch, and a keepalive line is not a frame."""
    seen: list[str] = []

    def runner(host: str, *, command: str) -> Iterator[str]:
        seen.append(command)
        yield from lines

    frames, next_at = StreamChannel("/spool", runner).fetch("gold", job="job1", offset=offset)
    assert [frame.payload["text"] for frame in frames] == texts
    assert next_at == resume
    assert seen == [f"{_FOLLOW} {offset}"]


def test_a_channel_resolves_by_name_and_cascades_only_when_asked_for_auto() -> None:
    """A cascade records why it landed where it did, so a fallback is never silent."""
    channels = _channels()
    assert isinstance(channels.resolve("stream"), StreamChannel)
    assert channels.last_resolution is None
    assert isinstance(channels.resolve("auto"), PollChannel)
    assert channels.last_resolution is not None
    assert channels.last_resolution.winner == "ssh-poll"
    assert channels.last_resolution.rejected == ()


def test_resolving_refuses_an_unknown_name_and_a_roster_with_nothing_available() -> None:
    """Neither refusal leaves a resolution behind, since neither one resolved anything."""
    named = _channels()
    with pytest.raises(StrategyError):
        named.resolve("carrier-pigeon")
    assert named.last_resolution is None
    bare = Channels("", lambda host, command: "", lambda host, command: iter(()))
    with pytest.raises(StrategyError):
        bare.resolve("auto")
    assert bare.last_resolution is None


def test_a_cached_runner_reuses_one_connection_per_host_through_shared() -> None:
    """Both transports go through the same pool, so a poll and a stream share a connection."""
    built: list[str] = []
    shared: Shared[str, str] = Shared(
        build=lambda host: built.append(host) or f"machine:{host}", idle_seconds=30.0
    )
    runner = cached(shared, lambda machine, command: f"{machine} ran {command}")
    assert runner("gold", command="echo hi") == "machine:gold ran echo hi"
    assert runner("gold", command="echo bye") == "machine:gold ran echo bye"
    assert built == ["gold"]

    def execute(machine: str, command: str) -> Iterator[str]:
        yield machine
        yield command

    streaming = cached_stream(shared, execute)
    assert list(streaming("gold", command="tail -f")) == ["machine:gold", "tail -f"]
    assert built == ["gold"]
