"""Evidence kept in the lake: ingested byte for byte, read back once the files are gone from the
tree, written back on request and verified by `mb lake check`."""

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from mainboard.dispatch.provenance import SourceTree, listing
from mainboard.nodes import evidence_of
from mainboard.state import DirectoryReplica, Evidence, EvidenceTree, Lake, schema
from mainboard.state import blobs as blobs_module
from mainboard.state import evidence as evidence_module
from mainboard.state.blobs import Blobs
from mainboard.trials.artifacts import Artifact
from mainboard.trials.dataset import Dataset

pl = pytest.importorskip("polars")

NODE = "research/lab/datasets/experiments/law"


@pytest.fixture
def evidence(workspace: Path) -> Path:
    """A node's evidence: a receipt partition, a content-addressed object the receipt pins, an
    empty file and a file of random bytes."""
    root = workspace / NODE / "evidence"
    table = pl.DataFrame({"bits": [1, 2, 3]})
    payload = root / "artifacts" / "run1" / "objects" / "table"
    payload.parent.mkdir(parents=True)
    table.write_parquet(payload)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    reference = Artifact(
        path=f"datasets/experiments/law/evidence/artifacts/run1/objects/{digest}",
        sha256=digest,
        size=payload.stat().st_size,
        media_type="application/vnd.apache.parquet",
    )
    payload.rename(payload.with_name(digest))
    part = root / "receipts" / "run=run1" / "part-00000.parquet"
    part.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "run": ["run1"],
            "lane": ["law"],
            "key": ["a"],
            "artifacts": [reference.model_dump_json()],
        }
    ).write_parquet(part)
    (root / "empty.txt").write_bytes(b"")
    (root / "noise.bin").write_bytes(os.urandom(70_000))
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_ingest_materialize_and_check_round_trip(
    mb, workspace, evidence, tmp_path_factory
) -> None:
    before = _snapshot(evidence)
    ran = mb("lake", "ingest", str(evidence), "--json")
    assert ran.code == 0, ran.said
    [kept] = json.loads(ran.out)
    assert kept == {"files": 4, "indexed": 4, "objects": 4, "size": sum(map(len, before.values()))}

    again = json.loads(mb("lake", "ingest", str(evidence), "--json").out)
    assert again == [{"files": 4, "indexed": 0, "objects": 0, "size": 0}]

    aside = tmp_path_factory.mktemp("aside") / "evidence"
    shutil.move(evidence, aside)
    ran = mb("lake", "materialize", f"{NODE}/evidence", "--json")
    assert ran.code == 0, ran.said
    assert len(json.loads(ran.out)) == 4
    assert _snapshot(evidence) == before

    ran = mb("lake", "check")
    assert ran.code == 0, ran.said
    ran = mb("query", "SELECT project, node, run FROM lake.evidence ORDER BY path", "--json")
    assert {tuple(row.values()) for row in json.loads(ran.out)} == {
        ("lab", "law", "run1"),
        ("lab", "law", ""),
    }
    imports = mb("query", "SELECT destination FROM lake.imports", "--json")
    assert json.loads(imports.out) == [{"destination": "evidence_log"}] * 2


def test_readers_find_evidence_that_left_the_tree(workspace, evidence, tmp_path_factory) -> None:
    project = workspace / "research" / "lab"
    Evidence(Lake.at(workspace)).ingest([evidence])
    store = Dataset(evidence / "receipts")
    [row] = store.rows()
    reference = Artifact.model_validate(row["artifacts"])
    pinned = reference.read(project)
    receipt = evidence / "receipts" / "run=run1" / "part-00000.parquet"
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()

    shutil.move(evidence, tmp_path_factory.mktemp("aside") / "evidence")
    assert reference.read(project) == pinned
    assert store.rows() == [row]
    assert [part.name for part in store.parts] == ["part-00000.parquet"]
    assert Dataset.holding(evidence) is not None
    assert hashlib.sha256(EvidenceTree(project).read(receipt, digest)).hexdigest() == digest
    assert evidence_of(project, "law") == "datasets/experiments/law/evidence"
    with pytest.raises(ValueError, match="pinned digest"):
        EvidenceTree(project).read(workspace / "mb.toml", "0" * 64)
    with pytest.raises(FileNotFoundError):
        EvidenceTree(project).recall("0" * 64)


def test_a_large_object_is_kept_in_ordered_chunks(workspace, monkeypatch) -> None:
    monkeypatch.setattr(blobs_module, "CHUNK_BYTES", 1000)
    large = workspace / "experiments" / "big" / "evidence" / "large.bin"
    large.parent.mkdir(parents=True)
    large.write_bytes(os.urandom(4500))
    digest = hashlib.sha256(large.read_bytes()).hexdigest()
    lake = Lake.at(workspace)
    Evidence(lake).ingest([large])
    [(chunks,)] = lake.query("SELECT count(*) FROM lake.blobs WHERE sha256 = ?", [digest])
    assert chunks == 5
    target = workspace / "restored.bin"
    with lake.open() as connection:
        assert Blobs(lake).write(connection, digest, target)
        assert Blobs(lake).read(connection, {digest}) == {digest: large.read_bytes()}
    assert target.read_bytes() == large.read_bytes()


def test_a_blob_from_before_chunking_still_reads(workspace) -> None:
    lake = Lake.at(workspace).ready()
    payload = b"source kept before ordinals"
    digest = hashlib.sha256(payload).hexdigest()
    with lake.open(write=True) as connection:
        lake.evolve(connection)
    lake.append(schema.blobs, [{"sha256": digest, "bytes": payload}])
    with lake.open() as connection:
        assert Blobs(lake).read(connection, {digest}) == {digest: payload}


def test_blob_membership_is_limited_to_requested_digests(workspace, monkeypatch) -> None:
    monkeypatch.setattr(blobs_module, "CHUNK_BYTES", 3)
    lake = Lake.at(workspace).ready()
    blobs = Blobs(lake)
    payloads = [b"chunked", b"legacy", b"unrequested"]
    chunked, legacy, unrelated = [hashlib.sha256(payload).hexdigest() for payload in payloads]
    lake.transact(
        lambda connection: blobs.stage(connection, {chunked: payloads[0], unrelated: payloads[2]})
    )
    lake.append(schema.blobs, [{"sha256": legacy, "bytes": payloads[1]}])
    with lake.open() as connection:
        assert blobs.held(connection, {chunked, legacy, "0" * 64}) == {chunked, legacy}
        assert blobs.read(connection, {chunked, legacy, "0" * 64}) == {
            chunked: payloads[0],
            legacy: payloads[1],
        }
        assert blobs.held(connection, {"0" * 64}) == set()
        assert blobs.read(connection, {"0" * 64}) == {}
        for invalid in ("' OR true --", "", "g" * 64, "0" * 63, "0" * 64 + "\n"):
            for select in (blobs.held, blobs.read):
                with pytest.raises(ValueError, match="not a SHA-256 digest"):
                    select(connection, {invalid})
    # An empty request does not touch even a closed connection.
    assert blobs.held(connection, set()) == set()
    assert blobs.read(connection, set()) == {}


def test_blob_membership_handles_large_requested_sets(workspace, monkeypatch) -> None:
    monkeypatch.setattr(blobs_module, "CHUNK_BYTES", 3)
    lake = Lake.at(workspace).ready()
    blobs = Blobs(lake)
    objects = {
        hashlib.sha256(payload).hexdigest(): payload
        for index in range(1601)
        for payload in (f"object-{index}".encode(),)
    }
    lake.transact(lambda connection: blobs.stage(connection, objects))
    # Duplicate chunks and duplicate requests must not multiply returned bytes.
    digest = next(iter(objects))
    lake.transact(lambda connection: blobs.stage(connection, {digest: objects[digest]}))
    unrequested = next(reversed(objects))
    expected = {key: value for key, value in objects.items() if key != unrequested}
    requested = [*expected, digest, "0" * 64]
    with lake.open() as connection:
        depth = connection.execute("SELECT current_setting('max_expression_depth')").fetchone()
        assert blobs.held(connection, requested) == set(expected)
        assert blobs.read(connection, requested) == expected
        assert (
            connection.execute("SELECT current_setting('max_expression_depth')").fetchone()
            == depth
        )


def test_ingest_deduplicates_within_and_across_windows(workspace, monkeypatch) -> None:
    monkeypatch.setattr(evidence_module, "STAGED_BYTES", 8)
    lake = Lake.at(workspace)
    keeper = Evidence(lake)
    previous = workspace / "previous.bin"
    previous.write_bytes(b"kept")
    keeper.ingest([previous])
    root = workspace / "evidence"
    root.mkdir()
    payloads = [b"same", b"same", b"same", b"diff", b"kept"]
    for index, payload in enumerate(payloads):
        (root / f"{index}.bin").write_bytes(payload)
    same, different, kept = [hashlib.sha256(payload).hexdigest() for payload in payloads[2:]]
    with patch.object(Blobs, "held", autospec=True, side_effect=Blobs.held) as membership:
        first = keeper.ingest([root])
        again = keeper.ingest([root])
    assert first.model_dump() == {"files": 5, "indexed": 5, "objects": 2, "size": 8}
    assert again.model_dump() == {"files": 5, "indexed": 0, "objects": 0, "size": 0}
    assert [call.args[2] for call in membership.call_args_list] == [
        {same},
        {same, different},
        {kept},
    ] * 2
    assert lake.query(
        "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.evidence_log)"
    ) == [(3, 6)]
    for path in root.iterdir():
        path.unlink()
    assert len(keeper.materialize([root])) == 5
    assert [path.read_bytes() for path in sorted(root.iterdir())] == payloads
    assert keeper.verify() == []


def _chunked(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Lake, Path, str]:
    """A lake keeping one 4500-byte evidence file in five 1000-byte chunks."""
    monkeypatch.setattr(blobs_module, "CHUNK_BYTES", 1000)
    large = workspace / "experiments" / "big" / "evidence" / "large.bin"
    large.parent.mkdir(parents=True)
    large.write_bytes(os.urandom(4500))
    lake = Lake.at(workspace)
    Evidence(lake).ingest([large])
    return lake, large, hashlib.sha256(large.read_bytes()).hexdigest()


def _damage(lake: Lake, sql: str, parameters: list[object]) -> None:
    """Change the lake's rows behind its writers' back, the way a failing disk would."""
    with lake.open(write=True) as connection:
        connection.execute(sql, parameters)


def test_ingest_records_each_chunks_checksum(workspace, monkeypatch) -> None:
    lake, large, digest = _chunked(workspace, monkeypatch)
    payload = large.read_bytes()
    recorded = lake.query(
        "SELECT ordinal, md5 FROM lake.checksums WHERE sha256 = ? ORDER BY ordinal", [digest]
    )
    assert recorded == [
        (ordinal, hashlib.md5(payload[start : start + 1000]).hexdigest())
        for ordinal, start in enumerate(range(0, 4500, 1000))
    ]
    assert Evidence(lake).verify() == []


def test_check_records_the_checksums_of_chunks_kept_before_them(workspace, evidence) -> None:
    lake = Lake.at(workspace)
    Evidence(lake).ingest([evidence])
    _damage(lake, "DELETE FROM lake.checksums", [])
    assert Evidence(lake).verify() == []
    [(chunks, sums)] = lake.query(
        "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.checksums)"
    )
    assert sums == chunks == 4
    assert Evidence(lake).verify() == []
    [(sums,)] = lake.query("SELECT count(*) FROM lake.checksums")
    assert sums == 4


def test_check_finds_a_chunk_whose_bytes_changed(workspace, monkeypatch) -> None:
    lake, _, digest = _chunked(workspace, monkeypatch)
    _damage(
        lake,
        "UPDATE lake.blobs SET bytes = repeat('\\x00'::BLOB, 1000) "
        "WHERE sha256 = ? AND ordinal = 3",
        [digest],
    )
    [finding] = Evidence(lake).verify()
    assert finding.kind == "evidence" and "fails its checksum" in finding.detail


def test_check_finds_a_chunk_gone_missing(workspace, monkeypatch) -> None:
    lake, _, digest = _chunked(workspace, monkeypatch)
    _damage(lake, "DELETE FROM lake.blobs WHERE sha256 = ? AND ordinal = 2", [digest])
    [finding] = Evidence(lake).verify()
    assert "1 of its 5 chunks are missing" in finding.detail
    _damage(lake, "DELETE FROM lake.blobs WHERE sha256 = ? AND ordinal = 4", [digest])
    [finding] = Evidence(lake).verify()
    assert "1 of its 4 chunks are missing" in finding.detail
    _damage(lake, "DELETE FROM lake.blobs WHERE sha256 = ?", [digest])
    [finding] = Evidence(lake).verify()
    assert "no copy held" in finding.detail


def test_check_finds_a_data_file_damaged_on_disk(mb, workspace, evidence) -> None:
    lake = Lake.at(workspace)
    Evidence(lake).ingest([evidence])
    [data] = [path for path in lake.data.rglob("*.parquet") if "blobs" in path.parts]
    damaged = bytearray(data.read_bytes())
    middle = len(damaged) // 2
    damaged[middle : middle + 64] = bytes(byte ^ 0xFF for byte in damaged[middle : middle + 64])
    data.write_bytes(bytes(damaged))
    ran = mb("lake", "check", "--json")
    assert ran.code == 1, ran.said
    assert {finding["kind"] for finding in json.loads(ran.out)} <= {"evidence", "unreadable"}


def test_a_group_of_small_files_lands_in_one_data_file(workspace) -> None:
    lake = Lake.at(workspace).ready()
    root = workspace / "experiments" / "many" / "evidence"
    root.mkdir(parents=True)
    for index in range(300):
        (root / f"{index:03}.bin").write_bytes(os.urandom(3000))
    Evidence(lake).ingest([root])
    for table in ("blobs", "checksums"):
        files = [path for path in lake.data.rglob("*.parquet") if table in path.parts]
        assert len(files) == 1, (table, files)
    [(chunks, sums)] = lake.query(
        "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.checksums)"
    )
    assert chunks == sums == 300
    assert Evidence(lake).verify() == []


def test_source_archives_restore_through_the_same_blobs(workspace, tmp_path_factory) -> None:
    source = workspace / "src" / "code.py"
    source.parent.mkdir()
    source.write_text("print('kept')\n", encoding="utf-8")
    lake = Lake.at(workspace)
    Evidence(lake).ingest([source])
    tree = SourceTree(workspace)
    identity, rows = tree.seal(["src/code.py"])
    digest = tree.archive(listing(rows))
    assert digest == identity.digest
    assert tree.archive(listing(rows)) == digest
    into = tmp_path_factory.mktemp("restored")
    [written] = tree.restore(digest, into)
    assert written.read_bytes() == source.read_bytes()
    copy = source.with_name("copy.py")
    copy.write_bytes(source.read_bytes())
    added = source.with_name("added.py")
    added.write_text("value = 1\n", encoding="utf-8")
    _, rows = tree.seal(["src/code.py", "src/copy.py", "src/added.py"])
    newer = tree.archive(listing(rows))
    assert tree.archive(listing(rows)) == newer
    assert lake.query(
        "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.closures)"
    ) == [(2, 4)]
    restored = tree.restore(newer, tmp_path_factory.mktemp("newer"))
    assert {path.name: path.read_bytes() for path in restored} == {
        path.name: path.read_bytes() for path in (source, copy, added)
    }


def test_a_directory_replica_keeps_each_object_once_and_the_index(
    workspace, evidence, tmp_path_factory
) -> None:
    kept = Evidence(Lake.at(workspace))
    kept.ingest([evidence])
    root = tmp_path_factory.mktemp("replica")
    assert kept.replicate(DirectoryReplica(root)) == 4
    assert kept.replicate(DirectoryReplica(root)) == 0
    for row in kept.indexed():
        copy = root / row.sha256[:2] / row.sha256
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == row.sha256
    index = (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index) == len(kept.indexed())
    assert not list(root.rglob("*.partial"))


def test_ingest_refuses_what_lies_outside_the_workspace(mb, tmp_path_factory) -> None:
    stranger = tmp_path_factory.mktemp("stranger") / "file.bin"
    stranger.write_bytes(b"x")
    ran = mb("lake", "ingest", str(stranger))
    assert ran.code == 1 and "outside the workspace" in ran.err


def test_the_lake_records_when_evidence_landed(workspace, evidence) -> None:
    lake = Lake.at(workspace)
    Evidence(lake).ingest([evidence])
    [(stamp,)] = lake.query("SELECT max(ts) FROM lake.evidence")
    assert stamp <= datetime.now(UTC)
