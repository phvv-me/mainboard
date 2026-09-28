import pytest

from mainboard.tracking import is_batched, streamed


@pytest.mark.parametrize(
    ("name", "expected", "batched"),
    [
        ("batch:smoke-1/gold-1", ("smoke-1", "gold-1"), True),
        ("batch:smoke-1", ("smoke-1", "smoke-1"), True),
        ("study:abc/trial-3", ("abc", "trial-3"), False),
        ("study:abc", ("abc", "abc"), False),
        ("nightly", ("nightly", "nightly"), False),
        ("", ("run-7", "7"), False),
    ],
    ids=[
        "a batch job",
        "a whole batch",
        "a trial",
        "a whole study",
        "a named run",
        "an unnamed run",
    ],
)
def test_every_dispatch_label_routes_to_the_stream_and_job_it_names(
    name: str, expected: tuple[str, str], batched: bool
) -> None:
    """One router, so a plain submit and a study trial reach the run a batch job reaches, and
    only a batch's own flow publishes for a batch job."""
    assert streamed(name, handle="7") == expected
    assert is_batched(name) is batched
