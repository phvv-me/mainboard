from statistics import fmean

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard.profile import BenchSample, benchmark, compare


# Every example times a real loop, so the example budget stays small.
@settings(max_examples=10)
@given(
    iters=st.integers(min_value=1, max_value=6),
    warmup=st.integers(min_value=0, max_value=4),
    barrier=st.booleans(),
)
def test_benchmark_keeps_every_run_and_derives_its_aggregates(
    iters: int, warmup: int, barrier: bool
) -> None:
    """The sync barrier fires after the warmup and after every timed run, so async device work
    lands inside the sample."""
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
    samples = compare({"a": lambda: None, "b": lambda: None}, iters=2, warmup=0)
    assert {s.label for s in samples} == {"a", "b"}
    assert capsys.readouterr().out
    assert samples == sorted(samples, key=lambda s: s.mean_us)


@pytest.mark.parametrize(("iters", "warmup"), [(0, 1), (-1, 1), (1, -1)])
def test_invalid_benchmark_counts_do_not_run_work(iters: int, warmup: int) -> None:
    """Invalid counts fail before warmup or an empty sample reaches an aggregate."""
    calls: list[int] = []
    with pytest.raises(ValueError, match="iters >= 1 and warmup >= 0"):
        benchmark(lambda: calls.append(1), iters=iters, warmup=warmup)
    assert not calls
