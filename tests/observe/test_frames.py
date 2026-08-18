from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from mainboard.observe import Frame, Kind, decode, encode, encoded_length, next_offset, parse_tail

_AT = datetime(2026, 1, 1, tzinfo=UTC)
_kinds = st.sampled_from(list(Kind))
_texts = st.text(min_size=0, max_size=40)


def _frame(kind: Kind = Kind.line, *, offset: int = 0, text: str = "hi") -> Frame:
    return Frame(job="job1", kind=kind, offset=offset, at=_AT, payload={"text": text})


def test_frame_defaults_offset_and_payload() -> None:
    frame = Frame(job="job1", kind=Kind.started, at=_AT)
    assert frame.offset == 0
    assert frame.payload == {}
    assert frame.schema_version == 1


@given(kind=_kinds, text=_texts)
def test_encode_decode_round_trips(kind: Kind, text: str) -> None:
    frame = _frame(kind, text=text)
    assert decode(encode(frame)) == frame


def test_encode_is_one_newline_terminated_line() -> None:
    line = encode(_frame())
    assert line.endswith("\n")
    assert line.count("\n") == 1


def test_parse_tail_keeps_every_complete_line_when_newline_terminated() -> None:
    text = encode(_frame(offset=0)) + encode(_frame(offset=10))
    frames = parse_tail(text)
    assert [frame.offset for frame in frames] == [0, 10]


def test_parse_tail_drops_a_truncated_final_line() -> None:
    complete = encode(_frame(offset=0))
    text = complete + '{"job": "job1", "kind": "line", "offset": 10, "trunc'
    assert parse_tail(text) == [decode(complete)]


def test_parse_tail_on_a_single_incomplete_line_is_empty() -> None:
    assert parse_tail('{"job": "trunc') == []


def test_parse_tail_on_empty_text_is_empty() -> None:
    assert parse_tail("") == []


def test_parse_tail_skips_blank_lines() -> None:
    text = f"\n{encode(_frame())}\n"
    assert len(parse_tail(text)) == 1


def test_encoded_length_matches_the_actual_encoded_bytes() -> None:
    frame = _frame()
    assert encoded_length(frame) == len(encode(frame).encode())


def test_next_offset_returns_the_default_for_an_empty_batch() -> None:
    assert next_offset([], default=42) == 42


def test_next_offset_advances_past_the_last_frame() -> None:
    frame = _frame(offset=5)
    assert next_offset([frame], default=0) == 5 + encoded_length(frame)
