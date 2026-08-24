from statistics import fmean

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard.profile import BenchSample, benchmark, compare


# Every example times a real loop, so the budget is trimmed from the shared default to keep
# the suite's sub-second inner loop.
@settings(max_examples=10)
@given(
    iters=st.integers(min_value=1, max_value=6),
    warmup=st.integers(min_value=0, max_value=4),
    barrier=st.booleans(),
)
def test_benchmark_keeps_every_run_and_derives_its_aggregates(
    iters: int, warmup: int, barrier: bool
) -> None:
    """One call yields the per-iteration rows, and mean/min/runs are read off those rows.

    The sync barrier, when given, fires once after the warmup and once after every timed
    run, so async device work lands inside the sample rather than after it.
    """
    calls: list[int] = []
    synced: list[int] = []
    sample = benchmark(
        lambda: calls.append(1),
        label="work",
        iters=iters,
        warmup=warmup,
        sync=(lambda: synced.append(1)) if barrier else None,
    )
    assert isinstance(sample, BenchSample)
    assert sample.label == "work"
    assert sample.runs == len(sample.samples) == iters
    assert sample.mean_us == fmean(sample.samples)
    assert sample.min_us == min(sample.samples)
    assert len(calls) == warmup + iters
    assert len(synced) == (1 + iters if barrier else 0)


def test_compare_tabulates_fastest_first(capsys: pytest.CaptureFixture[str]) -> None:
    """`compare` benchmarks each case and prints them fastest-mean first."""
    samples = compare({"a": lambda: None, "b": lambda: None}, iters=2, warmup=0)
    assert {s.label for s in samples} == {"a", "b"}
    assert capsys.readouterr().out
    assert samples == sorted(samples, key=lambda s: s.mean_us)
