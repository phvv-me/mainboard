"""The run registry in a real lake: built queries, and verdicts moved along `VERDICTS`."""

from collections.abc import Generator

import pytest

from mainboard.dispatch import vocabulary
from mainboard.dispatch.state.cache import Cache, RunRecord


@pytest.fixture
def cache() -> Generator[Cache]:
    registry = Cache.private()
    yield registry
    registry.close()


def _run(handle: str, *, project: str = "p") -> RunRecord:
    return RunRecord(
        handle=handle,
        target="gold",
        kind="ssh",
        script="train.py",
        args="",
        submitted_at=f"2026-10-07T00:00:0{handle}",
        verdict=vocabulary.QUEUED,
        project=project,
    )


def test_a_verdict_moves_only_along_the_table(cache: Cache) -> None:
    run = _run("1")
    cache.record(run)

    finished = cache.resolve(run, "C", 0, vocabulary.OK)
    stale = cache.resolve(run, "R", None, vocabulary.RUNNING)

    assert finished.verdict == vocabulary.OK, "a job finished between polls is never seen running"
    assert stale.verdict == vocabulary.OK, "a stale report never reopens a settled run"


def test_a_requeued_run_goes_back_to_the_queue(cache: Cache) -> None:
    run = _run("1")
    cache.record(run)
    cache.resolve(run, "R", None, vocabulary.RUNNING)

    assert cache.resolve(run, "Q", None, vocabulary.QUEUED).verdict == vocabulary.QUEUED


def test_listings_filter_in_the_lake(cache: Cache) -> None:
    for handle, project in (("1", "p"), ("2", "q"), ("3", "p")):
        cache.record(_run(handle, project=project))
    cache.resolve(_run("1"), "C", 0, vocabulary.OK)

    assert [run.handle for run in cache.live()] == ["3", "2"]
    assert [run.handle for run in cache.live(project="p")] == ["3"]
    assert [run.handle for run in cache.settled(5)] == ["1"]
    assert [run.handle for run in cache.tracked()] == ["3", "2", "1"]
    assert cache.total() == 3
    assert cache.run("2").project == "q"
