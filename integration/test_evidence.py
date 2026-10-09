"""Evidence kept in the lake: ingested byte for byte, read back once the files are gone from the
tree, written back on request and verified by `mb lake check`."""

import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZIP_ZSTANDARD, ZipFile

import pytest

from mainboard.core import MissionError, Project
from mainboard.dispatch.collection.collector import Collector
from mainboard.dispatch.collection.pack import HELD, pack
from mainboard.dispatch.provenance import SourceTree, listing
from mainboard.monitor import Monitor
from mainboard.nodes import evidence_of
from mainboard.state import DirectoryReplica, Evidence, EvidenceTree, Lake, schema
from mainboard.state import blobs as blobs_module
from mainboard.state import evidence as evidence_module
from mainboard.state.blobs import Blobs
from mainboard.state.relations import Relations, write_parquet
from mainboard.trials import Artifact, Dataset, Session
from mainboard.trials.artifacts import Artifacts, pinned

NODE = "research/lab/datasets/experiments/law"


@pytest.fixture
def evidence(workspace: Path) -> Path:
    """A node's evidence: a receipt partition, a content-addressed object the receipt pins, an
    empty file and a file of random bytes."""
    root = workspace / NODE / "evidence"
    payload = root / "artifacts" / "run1" / "objects" / "table"
    payload.parent.mkdir(parents=True)
    write_parquet(Relations().rows([{"bits": bits} for bits in (1, 2, 3)]), payload)
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
    receipt = {"run": "run1", "lane": "law", "key": "a", "artifacts": reference.model_dump_json()}
    write_parquet(Relations().rows([receipt]), part)
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


def test_evict_deletes_only_what_the_lake_keeps_intact(workspace, evidence) -> None:
    before = _snapshot(evidence)
    lake = Evidence(Lake.at(workspace))
    lake.ingest([evidence])
    (evidence / "noise.bin").write_bytes(b"changed after ingest")
    (evidence / "late.txt").write_bytes(b"never ingested")
    evicted = lake.evict([evidence])
    assert {path.name for path in evicted} == {
        "empty.txt",
        "part-00000.parquet",
        Path(next(name for name in before if name.startswith("artifacts"))).name,
    }
    assert sorted(_snapshot(evidence)) == ["late.txt", "noise.bin"]
    lake.materialize([evidence / "receipts"])
    assert (evidence / "receipts" / "run=run1" / "part-00000.parquet").is_file()


def _transfer(path: Path, files: Mapping[str, bytes], held: Mapping[str, str]) -> Path:
    """A host's transfer as `pack` writes it: zstd members, then what it left unsent."""
    with ZipFile(path, "w", ZIP_ZSTANDARD) as transfer:
        for name, data in files.items():
            transfer.writestr(name, data)
        transfer.writestr(HELD, json.dumps(held))
    return path


def test_collection_keeps_transfers_in_the_lake_never_the_tree(
    workspace, evidence, tmp_path
) -> None:
    lake = Evidence(Lake.at(workspace))
    lake.ingest([evidence])
    lake.evict([evidence])
    collector = Collector(workspace)
    indexed = {row.path: row.sha256 for row in lake.indexed(NODE)}
    assert collector._kept(PurePosixPath(NODE)) == indexed
    table = next(path for path in indexed if "/objects/" in path)
    copy = f"{NODE}/evidence/artifacts/run2/objects/{indexed[table]}"
    new = f"{NODE}/evidence/new.txt"
    staging = Project().out(workspace) / "tmp"
    live = staging / f"collect-{os.getpid()}-live"
    for left in (staging / "collect-999999999-dead", live):
        left.mkdir(parents=True)

    merged = collector.merge(
        _transfer(tmp_path / "t.zip", {new: b"fresh"}, {copy: indexed[table]}), path=NODE
    )
    assert merged == 2
    assert not (workspace / new).exists() and not (workspace / copy).exists()
    assert {row.path: row.sha256 for row in lake.indexed(NODE)} == {
        **indexed,
        new: hashlib.sha256(b"fresh").hexdigest(),
        copy: indexed[table],
    }
    assert EvidenceTree(workspace).read(workspace / copy, indexed[table])
    assert list(staging.iterdir()) == [live]
    noise = f"{NODE}/evidence/noise.bin"
    with pytest.raises(ValueError, match="the lake's copy kept"):
        collector.merge(_transfer(tmp_path / "c.zip", {noise: b"other bytes"}, {}), path=NODE)
    with pytest.raises(MissionError, match="keeps no object"):
        collector.merge(_transfer(tmp_path / "m.zip", {}, {new + "2": "0" * 64}), path=NODE)


def test_a_host_sends_only_what_the_lake_lacks(workspace, evidence, tmp_path, monkeypatch) -> None:
    host = tmp_path / "host"
    shutil.copytree(evidence, host / NODE / "evidence")
    [table] = (host / NODE / "evidence" / "artifacts" / "run1" / "objects").iterdir()
    again = host / NODE / "evidence" / "artifacts" / "run2" / "objects" / table.name
    again.parent.mkdir(parents=True)
    shutil.copy(table, again)
    live = host / NODE / "evidence" / "artifacts" / "run2" / "trial" / "events" / "live.ndjson"
    live.parent.mkdir(parents=True)
    live.write_bytes(b'{"offset": 0}\n')
    Evidence(Lake.at(workspace)).ingest([evidence])
    kept = Collector(workspace)._kept(PurePosixPath(NODE))

    def packed(known: Mapping[str, str]) -> ZipFile:
        sink = io.BytesIO()
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=sink))
        pack(str(host), relative=NODE, kept=kept, known=known, cache=".mb/run/digests.json")
        return ZipFile(io.BytesIO(sink.getvalue()))

    with packed({}) as archive:
        [snapshot] = [name for name in archive.namelist() if "/events/collected-" in name]
        assert sorted(archive.namelist()) == sorted([snapshot, HELD])
        assert {info.compress_type for info in archive.infolist()} == {ZIP_ZSTANDARD}
        assert json.loads(archive.read(HELD)) == {again.relative_to(host).as_posix(): table.name}
    with packed({snapshot: ""}) as archive:
        assert archive.namelist() == [HELD]


def test_a_pass_collects_each_folder_once() -> None:
    monitor = Monitor.__new__(Monitor)
    monitor.quiet, monitor.fetched = {}, {}
    jobs = [Mock(handle=Mock(host="miyabi-g", root="/work", fetch_path=NODE)) for _ in range(3)]
    assert [monitor.pull(job) for job in jobs] == [NODE] * 3
    assert sum(job.pull.call_count for job in jobs) == 1


def test_dedup_links_settled_duplicates_to_one_copy(mb, workspace) -> None:
    artifacts = workspace / NODE / "evidence" / "artifacts"
    data = os.urandom(5000)
    digest = hashlib.sha256(data).hexdigest()
    settled = time.time() - 7200
    for run, payload in [("run1", data), ("run2", data), ("run3", data), ("run4", data[::-1])]:
        copy = artifacts / run / "objects" / digest
        copy.parent.mkdir(parents=True)
        copy.write_bytes(payload)
        os.utime(copy, (settled, settled))
    flight = artifacts / "run5" / "objects" / digest
    flight.parent.mkdir(parents=True)
    flight.write_bytes(data)

    ran = mb("lake", "dedup", f"{NODE}/evidence", "--json")
    assert ran.code == 0, ran.said
    [done] = json.loads(ran.out)
    assert done == {
        "path": f"{NODE}/evidence",
        "files": 5,
        "hashed": 4,
        "linked": 2,
        "damaged": 1,
        "before": 25_000,
        "after": 15_000,
    }
    copies = [artifacts / f"run{run}" / "objects" / digest for run in "123"]
    assert len({copy.stat().st_ino for copy in copies}) == 1
    assert flight.stat().st_nlink == 1
    assert (artifacts / "run4" / "objects" / digest).read_bytes() == data[::-1]
    [again] = json.loads(mb("lake", "dedup", f"{NODE}/evidence", "--json").out)
    assert (again["linked"], again["before"], again["after"]) == (0, 15_000, 15_000)


def test_artifacts_are_held_once_per_node(tmp_path) -> None:
    store = Artifacts(tmp_path, tmp_path / "evidence")
    revision = "a" * 40
    source = f"hf://org/model@{revision}/tokenizer.json"
    first = store.write(b"tokenizer", media_type="application/json", source=source)
    digest = hashlib.sha256(b"tokenizer").hexdigest()
    assert first.path == f"evidence/objects/{digest[:2]}/{digest}" and first.source == source
    held = tmp_path / first.path
    held.write_bytes(b"damaged bytes")
    again = store.write(b"tokenizer", media_type="application/json")
    assert again.path == first.path and again.source == ""
    assert held.read_bytes() == b"tokenizer"
    assert [path.name for path in tmp_path.rglob("*") if path.is_file()] == [digest]
    assert store.written == {held}
    hub = tmp_path / "hub"
    assert pinned(hub / f"models--org--model/snapshots/{revision}/tokenizer.json") == source
    assert (
        pinned(hub / f"datasets--org--set/snapshots/{revision}/data/train.parquet")
        == f"hf://datasets/org/set@{revision}/data/train.parquet"
    )
    assert pinned(tmp_path / "tokenizer.json") == ""


@pytest.mark.parametrize("central", [True, False])
def test_a_closed_run_moves_into_the_center_lake(workspace, central) -> None:
    if central:
        (workspace / ".git").mkdir()
    lake = Lake.at(workspace).ready()
    store = Artifacts(workspace, workspace / NODE / "evidence")
    reference = store.write(b"table", media_type="application/octet-stream")
    session = Session.__new__(Session)
    session.staged, session.leased, session.writers = Mock(), None, {}
    session.stores, session.baseline = {"law": store}, {}
    session.declared = Mock(flags=(), universe=Mock(root=workspace / NODE))
    assert session.close() == ""
    held = workspace / reference.path
    assert held.exists() is not central
    assert [row.path for row in Evidence(lake).indexed(NODE)] == (
        [reference.path] if central else []
    )


def test_settling_verifies_what_only_the_lake_holds(workspace, evidence) -> None:
    lake = Evidence(Lake.at(workspace))
    lake.ingest([evidence])
    lake.evict([evidence])
    [row] = Dataset(evidence / "receipts").rows()
    reference = Artifact.model_validate(row["artifacts"])
    lost = reference.model_copy(update={"sha256": "1" * 64, "path": "datasets/lost"})

    def receipts(*references: Artifact) -> list[str]:
        listed = {str(at): value.model_dump() for at, value in enumerate(references)}
        return [json.dumps({"trial_receipt": {"artifacts": listed}})]

    Artifacts.verify(
        receipts(reference, reference), directory=workspace / NODE, boundary=workspace
    )
    with pytest.raises(FileNotFoundError):
        Artifacts.verify(receipts(lost), directory=workspace / NODE, boundary=workspace)


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
    asked = [call.kwargs.get("digests", call.args[-1]) for call in membership.call_args_list]
    assert (
        asked
        == [
            {same},
            {same, different},
            {kept},
        ]
        * 2
    )
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


def test_a_replica_reads_objects_in_windows_and_a_larger_one_alone(
    workspace, evidence, tmp_path_factory, monkeypatch
) -> None:
    kept = Evidence(Lake.at(workspace))
    kept.ingest([evidence])
    sizes = sorted({row.sha256: row.size for row in kept.indexed()}.items(), key=lambda kv: kv[1])
    # The second largest fills a window alone, so the rest take more than one, and the largest
    # exceeds every window.
    monkeypatch.setattr(evidence_module, "_WINDOW_BYTES", sizes[-2][1])
    root = tmp_path_factory.mktemp("replica")
    with (
        patch.object(Blobs, "read", autospec=True, side_effect=Blobs.read) as read,
        patch.object(Evidence, "_copy", autospec=True, side_effect=Evidence._copy) as alone,
    ):
        assert kept.replicate(DirectoryReplica(root)) == len(sizes)
    assert read.call_count >= 2
    assert [call.args[2] for call in alone.call_args_list] == [sizes[-1][0]]
    for digest, _ in sizes:
        assert hashlib.sha256((root / digest[:2] / digest).read_bytes()).hexdigest() == digest


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
