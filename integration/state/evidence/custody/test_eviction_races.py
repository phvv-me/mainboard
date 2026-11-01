import os
from collections.abc import Callable, Collection
from pathlib import Path

import duckdb
import pytest

from mainboard import MissionError
from mainboard.state import Evidence
from mainboard.state.evidence import core
from mainboard.state.storage.blobs import Audit
from mainboard.state.storage.owned import OwnedFile

_CLOSED = b"finalized"
_SUCCESSOR = b"successor"


@pytest.fixture
def source(stored: Callable[[str, bytes], Path]) -> Path:
    """A file ingested and then closed by its producer."""
    return stored("result.bin", _CLOSED)


def test_a_file_replaced_after_it_was_closed_is_put_back(
    kept: Evidence, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = OwnedFile.digest

    def replace(self: OwnedFile) -> str:
        seen = digest(self)
        (kept.root / "replacement").write_bytes(_SUCCESSOR)
        (kept.root / "replacement").replace(source)
        return seen

    monkeypatch.setattr(OwnedFile, "digest", replace)
    assert kept.evict(kept.finalize([source])) == []
    assert source.read_bytes() == _SUCCESSOR and not list(kept.root.rglob(".mb-*"))


def test_a_successor_written_while_the_lake_is_checked_is_never_unlinked(
    kept: Evidence,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest_of: Callable[[bytes], str],
) -> None:
    audit, asked = kept.blobs.audit, []

    def check(connection: duckdb.DuckDBPyConnection, digests: Collection[str]) -> dict[str, Audit]:
        audits = audit(connection, digests)
        if asked:
            source.write_bytes(_SUCCESSOR)
        asked.append(digests)
        return audits

    monkeypatch.setattr(kept.blobs, "audit", check)
    assert kept.evict(kept.finalize([source])) == [source]
    assert source.read_bytes() == _SUCCESSOR
    assert kept.recall(digest_of(_CLOSED)) == _CLOSED


def test_a_failure_with_a_successor_in_place_keeps_both_versions(
    kept: Evidence,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest_of: Callable[[bytes], str],
) -> None:
    audit, asked = kept.blobs.audit, []

    def fail(connection: duckdb.DuckDBPyConnection, digests: Collection[str]) -> dict[str, Audit]:
        if not asked:
            asked.append(digests)
            return audit(connection, digests)
        source.write_bytes(_SUCCESSOR)
        raise RuntimeError("custody interrupted")

    monkeypatch.setattr(kept.blobs, "audit", fail)
    with pytest.raises(MissionError) as refused:
        kept.evict(kept.finalize([source]))
    [candidate] = kept.root.glob(".mb-custody-*/result.bin")
    assert candidate.read_bytes() == _CLOSED and source.read_bytes() == _SUCCESSOR
    assert all(digest_of(text) in str(refused.value) for text in [_CLOSED, _SUCCESSOR])


def test_a_failure_before_the_unlink_puts_the_copy_back(
    kept: Evidence, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unlink = os.unlink
    interrupted: list[str] = []

    def interrupt(path: str, *, dir_fd: int | None = None) -> None:
        if path == source.name and dir_fd is not None and not interrupted:
            interrupted.append(path)
            raise RuntimeError("unlink interrupted")
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", interrupt)
    with pytest.raises(RuntimeError, match="unlink interrupted"):
        kept.evict(kept.finalize([source]))
    assert source.read_bytes() == _CLOSED and not list(kept.root.rglob(".mb-custody-*"))


def test_an_eviction_asks_the_lake_once_for_each_batch_it_moves_aside(
    kept: Evidence, stored: Callable[[str, bytes], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [stored(f"part-{n}.bin", f"part {n}".encode()) for n in range(5)]
    audit, asked = kept.blobs.audit, []

    def counted(
        connection: duckdb.DuckDBPyConnection, digests: Collection[str]
    ) -> dict[str, Audit]:
        asked.append(len(digests))
        return audit(connection, digests)

    monkeypatch.setattr(kept.blobs, "audit", counted)
    monkeypatch.setattr(core, "EVICTED_TOGETHER", 2)
    assert sorted(kept.evict(kept.finalize(sources))) == sorted(sources)
    assert asked == [5, 2, 2, 1] and not list(kept.root.rglob(".mb-custody-*"))
