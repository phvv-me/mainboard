from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from patos import Shared, StrategyError

from mainboard.observe import Channels, Frame, Kind, PollChannel, StreamChannel, encode
from mainboard.observe.channels import cached, cached_stream

if TYPE_CHECKING:
    from collections.abc import Iterator

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _line(offset: int, text: str) -> Frame:
    return Frame(job="job1", kind=Kind.line, offset=offset, at=_AT, payload={"text": text})


def test_poll_channel_availability_needs_a_configured_root() -> None:
    assert PollChannel("/root", lambda host, command: "").available()
    assert not PollChannel("", lambda host, command: "").available()


def test_poll_channel_fetch_batches_status_then_a_translated_tail() -> None:
    calls: list[str] = []
    body = encode(_line(0, "a")) + encode(_line(50, "b"))

    def runner(host: str, *, command: str) -> str:
        calls.append(command)
        if command.startswith("cat"):
            return '{"state": "running", "offset": 100, "base": 0, "updated_at": "t"}'
        return body

    channel = PollChannel("/spool", runner)
    frames, offset = channel.fetch("gold", job="job1", offset=0)
    assert [frame.payload["text"] for frame in frames] == ["a", "b"]
    assert offset == 50 + len(encode(_line(50, "b")).encode())
    assert calls[0] == "cat /spool/job1/status.json"
    assert calls[1] == "tail -c +1 /spool/job1/live.ndjson"


def test_poll_channel_fetch_translates_the_offset_against_the_live_base() -> None:
    def runner(host: str, *, command: str) -> str:
        if command.startswith("cat"):
            return '{"state": "running", "offset": 100, "base": 40, "updated_at": "t"}'
        assert command == "tail -c +11 /spool/job1/live.ndjson"
        return ""

    frames, offset = PollChannel("/spool", runner).fetch("gold", job="job1", offset=50)
    assert frames == []
    assert offset == 50


def test_poll_channel_fetch_clamps_an_offset_behind_the_live_base() -> None:
    def runner(host: str, *, command: str) -> str:
        if command.startswith("cat"):
            return '{"state": "running", "offset": 100, "base": 40, "updated_at": "t"}'
        assert command == "tail -c +1 /spool/job1/live.ndjson"
        return ""

    frames, offset = PollChannel("/spool", runner).fetch("gold", job="job1", offset=0)
    assert frames == []
    assert offset == 40


def test_stream_channel_availability_needs_a_configured_root() -> None:
    assert StreamChannel("/root", lambda host, command: iter(())).available()
    assert not StreamChannel("", lambda host, command: iter(())).available()


def test_stream_channel_fetch_runs_a_follow_exec_and_skips_blank_lines() -> None:
    seen: list[str] = []

    def runner(host: str, *, command: str) -> Iterator[str]:
        seen.append(command)
        yield encode(_line(0, "a"))
        yield "\n"
        yield encode(_line(30, "b"))

    frames, offset = StreamChannel("/spool", runner).fetch("gold", job="job1", offset=0)
    assert [frame.payload["text"] for frame in frames] == ["a", "b"]
    assert offset == 30 + len(encode(_line(30, "b")).encode())
    assert seen == [
        "python -m mainboard.observe.agentmain --root /spool --job job1 --follow --from-offset 0"
    ]


def test_stream_channel_fetch_falls_back_to_the_given_offset_when_nothing_arrives() -> None:
    frames, offset = StreamChannel("/spool", lambda host, command: iter(())).fetch(
        "gold", job="job1", offset=7
    )
    assert frames == []
    assert offset == 7


def test_channels_auto_cascades_to_the_first_available_and_records_the_resolution() -> None:
    channels = Channels("/spool", lambda host, command: "", lambda host, command: iter(()))
    channel = channels.resolve("auto")
    assert isinstance(channel, PollChannel)
    assert channels.last_resolution is not None
    assert channels.last_resolution.winner == "ssh-poll"
    assert channels.last_resolution.rejected == ()


def test_channels_select_by_name_does_not_touch_last_resolution() -> None:
    channels = Channels("/spool", lambda host, command: "", lambda host, command: iter(()))
    channel = channels.resolve("stream")
    assert isinstance(channel, StreamChannel)
    assert channels.last_resolution is None


def test_channels_select_an_unknown_name_raises() -> None:
    channels = Channels("/spool", lambda host, command: "", lambda host, command: iter(()))
    with pytest.raises(StrategyError):
        channels.resolve("carrier-pigeon")


def test_channels_auto_raises_when_every_channel_is_unavailable() -> None:
    channels = Channels("", lambda host, command: "", lambda host, command: iter(()))
    with pytest.raises(StrategyError):
        channels.resolve("auto")
    assert channels.last_resolution is None


def test_cached_reuses_one_connection_per_host_through_shared() -> None:
    built: list[str] = []
    shared: Shared[str, str] = Shared(
        build=lambda host: built.append(host) or f"machine:{host}", idle_seconds=30.0
    )
    runner = cached(shared, lambda machine, command: f"{machine} ran {command}")
    assert runner("gold", command="echo hi") == "machine:gold ran echo hi"
    assert runner("gold", command="echo bye") == "machine:gold ran echo bye"
    assert built == ["gold"]


def test_cached_stream_reuses_one_connection_per_host_through_shared() -> None:
    shared: Shared[str, str] = Shared(build=lambda host: f"machine:{host}")

    def execute(machine: str, command: str) -> Iterator[str]:
        yield machine
        yield command

    runner = cached_stream(shared, execute)
    assert list(runner("gold", command="tail -f")) == ["machine:gold", "tail -f"]
