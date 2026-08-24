from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.observe import Frame, decode, encode, encoded_length, next_offset, parse_tail

from .support import FRAMES, line


@given(frame=FRAMES)
def test_a_frame_round_trips_through_its_wire_encoding(frame: Frame) -> None:
    """One newline-terminated line, whose byte length is what a reader seeks by."""
    encoded = encode(frame)
    assert decode(encoded) == frame
    assert encoded.endswith("\n")
    assert encoded.count("\n") == 1
    assert encoded_length(frame) == len(encoded.encode())
    assert decode(encoded).schema_version == 1


@given(frames=st.lists(FRAMES, max_size=5))
@example(frames=[line(text="before\u2028after")])
def test_parse_tail_keeps_every_complete_line_and_drops_a_truncated_one(
    frames: list[Frame],
) -> None:
    """A payload may legally embed a unicode line separator, which `str.splitlines` would cut."""
    body = "".join(encode(frame) for frame in frames)
    assert parse_tail(body) == frames
    assert parse_tail(f"\n{body}\n") == frames
    assert parse_tail(body + '{"job": "job1", "kind": "line", "trunc') == frames


@given(frames=st.lists(FRAMES, min_size=1, max_size=3))
def test_next_offset_advances_past_the_last_frame_or_reports_the_default(
    frames: list[Frame],
) -> None:
    """An empty batch advanced nothing, so the caller's own checkpoint comes straight back."""
    last = frames[-1]
    assert next_offset(frames, default=42) == last.offset + encoded_length(last)
    assert next_offset([], default=42) == 42
