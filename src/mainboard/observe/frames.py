# The one wire frame every observability path shares (spool, poll or stream channel, store), so a
# consumer is agnostic to which channel served it.

from datetime import (
    datetime,  # ruff:ignore[typing-only-standard-library-import]  reason=pydantic validates Frame.at with this class at runtime since=2026-08-17
)
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from ..core.base import Declared

if TYPE_CHECKING:
    from collections.abc import Sequence

_SCHEMA_VERSION = 1

type JSONValue = str | int | float | bool | dict[str, JSONValue] | list[JSONValue] | None


class Kind(StrEnum):
    """The four events a job's observability stream can carry."""

    started = auto()
    line = auto()
    sample = auto()
    ended = auto()


class Frame(Declared):
    """One line of a job's stream, however it was fetched.

    schema_version: the wire format revision, so an incompatible change is detected, not misparsed.
    offset: the byte position this frame starts at in the job's cumulative stream, the
        resumable checkpoint a follow-up fetch passes back in.
    payload: `text` for a `line`, `rss` for a `sample`, `exit_code` for `ended`.
    """

    schema_version: int = _SCHEMA_VERSION
    job: str
    kind: Kind
    offset: int = 0
    at: datetime
    payload: dict[str, JSONValue] = {}


def encode(frame: Frame) -> str:
    """One NDJSON line for `frame`, newline-terminated."""
    return f"{frame.model_dump_json()}\n"


decode = Frame.model_validate_json
"""Parse one NDJSON line back into its frame."""


def parse_tail(text: str) -> list[Frame]:
    """Every complete frame in NDJSON `text`, tolerating a final line a concurrent writer or a
    partial network read cut short."""
    # The literal "\n" `encode` writes, never `str.splitlines`, which also breaks on unicode line
    # separators (NEL, U+2028...) a payload string may legally embed.
    complete = text if text.endswith("\n") else text.rpartition("\n")[0]
    return [decode(line) for line in complete.split("\n") if line.strip()]


def encoded_length(frame: Frame) -> int:
    """The exact byte length `frame` occupies once encoded on the wire."""
    return len(encode(frame).encode())


def next_offset(frames: Sequence[Frame], *, default: int) -> int:
    """The stream position to resume from after `frames` (oldest first), `default` if empty."""
    return frames[-1].offset + encoded_length(frames[-1]) if frames else default
