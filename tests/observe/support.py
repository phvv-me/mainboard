from datetime import UTC, datetime

from hypothesis import strategies as st

from mainboard.observe import Frame, Kind

from ..strategies import TEXT

# One fixed instant, since every ordering law here is about byte offsets and never about time.
AT = datetime(2026, 1, 1, tzinfo=UTC)

KINDS = st.sampled_from(list(Kind))

FRAMES = st.builds(
    Frame,
    job=st.just("job1"),
    kind=KINDS,
    offset=st.integers(min_value=0, max_value=1_000_000),
    at=st.just(AT),
    payload=st.builds(dict, text=TEXT),
)


def line(offset: int = 0, text: str = "hi") -> Frame:
    """One `line` frame for `job1`, the frame a spool, a channel and a store all carry."""
    return Frame(job="job1", kind=Kind.line, offset=offset, at=AT, payload={"text": text})


def ended(offset: int = 0) -> Frame:
    """The `ended` frame that closes a job's stream."""
    return Frame(job="job1", kind=Kind.ended, offset=offset, at=AT, payload={"exit_code": 0})
