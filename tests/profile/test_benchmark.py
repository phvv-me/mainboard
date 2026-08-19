from statistics import fmean

import pytest

from mainboard.profile import BenchSample, benchmark, compare


def test_benchmark_keeps_every_run_and_derives_its_aggregates() -> None:
    """One call yields the per-iteration rows, and mean/min/runs are read off those rows."""
    calls = {"n": 0}

    def work() -> None:
        calls["n"] += 1

    sample = benchmark(work, label="work", iters=5, warmup=2)
    assert isinstance(sample, BenchSample)
    assert sample.label == "work"
    assert len(sample.samples) == 5
    assert sample.runs == 5
    assert sample.mean_us == fmean(sample.samples)
    assert sample.min_us == min(sample.samples)
    assert calls["n"] == 7  # warmup 2 + iters 5


def test_benchmark_calls_sync_after_warmup_and_every_run() -> None:
    synced: list[int] = []
    benchmark(lambda: None, iters=3, warmup=1, sync=lambda: synced.append(1))
    assert len(synced) == 4  # one post-warmup call + one per timed iter


def test_compare_tabulates_fastest_first(capsys: pytest.CaptureFixture[str]) -> None:
    """`compare` benchmarks each case and prints them fastest-mean first."""
    samples = compare({"a": lambda: None, "b": lambda: None}, iters=2, warmup=0)
    assert {s.label for s in samples} == {"a", "b"}
    assert capsys.readouterr().out
    assert samples == sorted(samples, key=lambda s: s.mean_us)
