from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO

import duckdb
import pytest

from mainboard import MissionError
from mainboard.dispatch import SourceTree
from mainboard.dispatch.provenance import listing
from mainboard.state import Evidence
from mainboard.state.storage.blobs import Audit


@pytest.fixture
def raced(kept: Evidence, monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[[], None]], None]:
    """Have an action happen once an object is read out of the lake, as a racing process would."""
    inspect = kept.blobs.inspect

    def after_the_lake_is_read(act: Callable[[], None]) -> None:
        def read(connection: duckdb.DuckDBPyConnection, digest: str, sink: IO[bytes]) -> Audit:
            audit = inspect(connection, digest, sink)
            act()
            return audit

        monkeypatch.setattr(kept.blobs, "inspect", read)

    return after_the_lake_is_read


@pytest.fixture
def sources(workspace: Path, keeper: Evidence) -> dict[str, Path]:
    """Three source files, the first ingested, the second its twin and the third other bytes."""
    code = workspace / "src" / "code.py"
    code.parent.mkdir()
    code.write_text("print('kept')\n", encoding="utf-8")
    keeper.ingest([code])
    twin, added = code.with_name("copy.py"), code.with_name("added.py")
    twin.write_bytes(code.read_bytes())
    added.write_text("value = 1\n", encoding="utf-8")
    return {"code.py": code, "copy.py": twin, "added.py": added}


def _archive(tree: SourceTree, names: list[str]) -> str:
    """The digest of `names` archived, which archiving them again must give too."""
    identity, rows = tree.seal(names)
    digest = tree.archive(listing(rows))
    assert digest == identity.digest
    assert tree.archive(listing(rows)) == digest
    return digest


def test_materialize_refuses_a_changed_file_lists_digests_and_writes_nothing(
    kept: Evidence, stored: Callable[[str, bytes], Path], digest_of: Callable[[bytes], str]
) -> None:
    changed = stored("t/changed.txt", b"old settled result")
    gone = stored("t/gone.txt", b"settled and then evicted")
    gone.unlink()
    changed.write_bytes(b"new uncollected result")
    with pytest.raises(MissionError) as refused:
        kept.materialize([kept.root / "t"])
    digests = [digest_of(b"old settled result"), digest_of(b"new uncollected result")]
    assert all(text in str(refused.value) for text in [str(changed), *digests])
    assert changed.read_bytes() == b"new uncollected result" and not gone.exists()


def test_materialize_writes_only_what_is_missing_into_any_root(
    kept: Evidence, stored: Callable[[str, bytes], Path], tmp_path: Path
) -> None:
    same, gone = stored("t/same.txt", b"same"), stored("t/gone.txt", b"gone")
    gone.unlink()
    before = same.stat()
    assert kept.materialize([kept.root / "t"]) == [gone] and gone.read_bytes() == b"gone"
    assert (same.stat().st_ino, same.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    elsewhere = tmp_path / "restore" / "here"
    assert kept.materialize([same], into=elsewhere) == [elsewhere / "t/same.txt"]
    assert (elsewhere / "t/same.txt").read_bytes() == b"same"


@pytest.mark.parametrize("link", ["folder", "file"])
def test_materialize_refuses_a_link_in_the_way(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    tmp_path_factory: pytest.TempPathFactory,
    link: str,
) -> None:
    source = stored("sub/result.txt", b"archived")
    restore, outside = tmp_path_factory.mktemp("restore"), tmp_path_factory.mktemp("outside")
    (outside / "result.txt").write_bytes(b"unrelated file")
    if link == "folder":
        (restore / "sub").symlink_to(outside, target_is_directory=True)
    else:
        (restore / "sub").mkdir()
        (restore / "sub" / "result.txt").symlink_to(outside / "result.txt")
    with pytest.raises(MissionError, match="unsafe evidence"):
        kept.materialize([source], into=restore)
    assert (outside / "result.txt").read_bytes() == b"unrelated file"


def test_a_folder_swapped_for_a_link_during_a_restore_is_never_followed(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    raced: Callable[[Callable[[], None]], None],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    source = stored("sub/result.bin", b"archived")
    restore, outside = tmp_path_factory.mktemp("restore"), tmp_path_factory.mktemp("outside")
    (restore / "sub").mkdir()

    def swap() -> None:
        (restore / "sub").rename(restore / "original")
        (restore / "sub").symlink_to(outside, target_is_directory=True)

    raced(swap)
    kept.materialize([source], into=restore)
    assert list(outside.iterdir()) == []
    assert (restore / "original" / "result.bin").read_bytes() == b"archived"


def test_a_file_appearing_during_a_restore_is_not_overwritten(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    raced: Callable[[Callable[[], None]], None],
) -> None:
    source = stored("result.bin", b"archived")
    source.unlink()

    def meanwhile() -> None:
        source.write_bytes(b"written meanwhile")

    raced(meanwhile)
    with pytest.raises(MissionError, match="refusing overwrite"):
        kept.materialize([source])
    assert source.read_bytes() == b"written meanwhile" and not list(kept.root.glob(".mb-*"))


def test_the_cache_replaces_a_stale_file_in_place(
    kept: Evidence, stored: Callable[[str, bytes], Path]
) -> None:
    source = stored("tree/result.bin", b"first")
    first = kept.cached(kept.indexed("tree")[0])
    assert first == kept.cache / "tree/result.bin" and first.read_bytes() == b"first"
    source.write_bytes(b"second")
    kept.ingest([source])
    assert kept.cached(kept.indexed("tree")[0]) == first and first.read_bytes() == b"second"


def test_the_cache_follows_no_link(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    stored("tree/result.bin", b"cached")
    outside = tmp_path_factory.mktemp("outside")
    kept.cache.mkdir(parents=True)
    (kept.cache / "tree").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MissionError, match="unsafe evidence path"):
        kept.cached(kept.indexed("tree")[0])
    assert list(outside.iterdir()) == []


def test_source_archives_restore_through_the_same_blobs(
    workspace: Path,
    keeper: Evidence,
    sources: Mapping[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    tree = SourceTree(workspace)
    first = _archive(tree, ["src/code.py"])
    restored = tree.restore(first, tmp_path_factory.mktemp("restored"))
    assert [path.read_bytes() for path in restored] == [sources["code.py"].read_bytes()]
    newer = _archive(tree, ["src/code.py", "src/copy.py", "src/added.py"])
    counts = "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.closures)"
    assert keeper.lake.query(counts) == [(2, 4)]
    restored = tree.restore(newer, tmp_path_factory.mktemp("newer"))
    assert {p.name: p.read_bytes() for p in restored} == {
        name: path.read_bytes() for name, path in sources.items()
    }
