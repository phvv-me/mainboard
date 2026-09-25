"""Archive round trips preserve pinned bytes and reject incomplete evidence."""

import hashlib
import random
from pathlib import Path

import polars as pl
import pytest

from mainboard.trials.archive import ArtifactChunk, ParquetArtifacts
from mainboard.trials.artifacts import Artifact, Artifacts


def test_archive_deduplicates_and_bounds_shards_without_changing_receipts(tmp_path: Path) -> None:
    payload = random.Random(42).randbytes(1500000)
    references = [
        Artifacts(tmp_path, tmp_path / run).write(data, media_type="application/octet-stream")
        for run, data in (("first", payload), ("second", payload), ("empty", b""))
    ]
    paths = [tmp_path / reference.path for reference in references]
    archive = ParquetArtifacts(tmp_path)
    archive.pack(paths)
    assert all(path.is_file() for path in paths)
    for path in paths:
        path.unlink()
    assert [reference.read(tmp_path) for reference in references] == [payload, payload, b""]
    shards = sorted(archive.directory.glob("*/*.parquet"))
    assert all(path.stat().st_size < 1000000 for path in shards)
    frame = pl.read_parquet(shards)
    assert frame["sha256"].n_unique() == 2
    assert sum(len(value) for value in frame["payload"]) == len(payload)
    missing = references[0].model_copy(update={"path": "other/objects/" + references[0].sha256})
    with pytest.raises(FileNotFoundError):
        missing.read(tmp_path)
    with pytest.raises(FileExistsError):
        archive.pack([])


def test_archive_checks_bytes_and_chunk_order(tmp_path: Path) -> None:
    reference = Artifacts(tmp_path, tmp_path / "run").write(
        b"evidence", media_type="application/octet-stream"
    )
    source = tmp_path / reference.path
    archive = ParquetArtifacts(tmp_path)
    archive.pack([source])
    source.unlink()
    shard = next(archive.directory.glob("*/*.parquet"))
    original = pl.read_parquet(shard)
    original.with_columns(pl.lit(b"changed").alias("payload")).write_parquet(shard)
    with pytest.raises(ValueError, match="content changed"):
        reference.read(tmp_path)
    original.with_columns(pl.lit(1).alias("ordinal")).write_parquet(shard)
    with pytest.raises(ValueError, match="incomplete"):
        reference.read(tmp_path)


def test_archive_rejects_changed_content_addresses_and_external_paths(tmp_path: Path) -> None:
    reference = Artifacts(tmp_path, tmp_path / "run").write(
        b"evidence", media_type="application/octet-stream"
    )
    source = tmp_path / reference.path
    source.write_bytes(b"modified")
    archive = ParquetArtifacts(tmp_path)
    with pytest.raises(ValueError, match="content-addressed artifact changed"):
        archive.pack([source])
    escaped = Artifact(path="../outside", sha256=hashlib.sha256(b"").hexdigest(), size=0)
    with pytest.raises(ValueError, match="canonical"):
        escaped.read(tmp_path)


def test_a_bucket_sharing_only_a_prefix_is_passed_over_up_to_the_boundary(tmp_path: Path) -> None:
    """A bucket holds every digest of one prefix, so finding it is not finding the bytes.

    The boundary is the filesystem root, the widest one a caller can declare, so the search
    climbs through every ancestor and still refuses rather than running off the top.
    """
    reference = Artifacts(tmp_path, tmp_path / "run").write(
        b"evidence", media_type="application/octet-stream"
    )
    ParquetArtifacts(tmp_path).pack([tmp_path / reference.path])
    sibling = reference.sha256[:2] + "f" * 62
    with pytest.raises(FileNotFoundError):
        ParquetArtifacts.read(
            tmp_path / reference.path, boundary=Path(tmp_path.anchor), digest=sibling
        )


def test_an_object_that_fills_a_shard_exactly_leaves_no_empty_shard_behind(tmp_path: Path) -> None:
    payload = random.Random(7).randbytes(8 * 65536)
    reference = Artifacts(tmp_path, tmp_path / "run").write(
        payload, media_type="application/octet-stream"
    )
    source = tmp_path / reference.path
    archive = ParquetArtifacts(tmp_path)
    archive.pack([source])
    source.unlink()
    assert len(list(archive.directory.glob("*/*.parquet"))) == 1
    assert reference.read(tmp_path) == payload


def test_a_shard_over_the_size_budget_is_refused_rather_than_kept(tmp_path: Path) -> None:
    """Packing bounds a shard by what it counts, and the written file is checked on top of that."""
    chunk = ArtifactChunk(
        sha256="0" * 64, ordinal=0, paths=[], payload=random.Random(1).randbytes(1100000)
    )
    with pytest.raises(ValueError, match="size budget"):
        ParquetArtifacts._write_part(tmp_path, 0, [chunk])
