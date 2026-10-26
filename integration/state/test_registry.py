"""The run registry in a real lake: built queries, and verdicts moved along `VERDICTS`."""

from collections.abc import Generator
from pathlib import Path

import pytest

from mainboard.dispatch.jobs import vocabulary
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.state import Lake


@pytest.fixture
def cache(tmp_path: Path) -> Generator[Cache]:
    registry = Cache(Lake.at(tmp_path))
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
        verdict=vocabulary.Word.QUEUED,
        project=project,
    )


def test_a_verdict_moves_only_along_the_table(cache: Cache) -> None:
    run = _run("1")
    cache.record(run)

    finished = cache.resolve(run, "C", 0, vocabulary.Word.OK)
    stale = cache.resolve(run, "R", None, vocabulary.Word.RUNNING)

    assert finished.verdict == vocabulary.Word.OK, (
        "a job finished between polls is never seen running"
    )
    assert stale.verdict == vocabulary.Word.OK, "a stale report never reopens a settled run"


def test_a_requeued_run_goes_back_to_the_queue(cache: Cache) -> None:
    run = _run("1")
    cache.record(run)
    cache.resolve(run, "R", None, vocabulary.Word.RUNNING)

    assert cache.resolve(run, "Q", None, vocabulary.Word.QUEUED).verdict == vocabulary.Word.QUEUED


def test_listings_filter_in_the_lake(cache: Cache) -> None:
    for handle, project in (("1", "p"), ("2", "q"), ("3", "p")):
        cache.record(_run(handle, project=project))
    cache.resolve(_run("1"), "C", 0, vocabulary.Word.OK)

    assert [run.handle for run in cache.listed("live")] == ["3", "2"]
    assert [run.handle for run in cache.listed("live", project="p")] == ["3"]
    assert [run.handle for run in cache.listed("settled", limit=5)] == ["1"]
    assert [run.handle for run in cache.listed("tracked")] == ["3", "2", "1"]
    assert cache.count() == 3
    assert cache.run("2").project == "q"
