import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from mainboard.state import Evidence

_LANDED = (
    "SELECT DISTINCT project, node, run, ts <= now() AS landed FROM lake.evidence ORDER BY ALL"
)
_IMPORTS = "SELECT source, destination, rows FROM lake.imports ORDER BY ts"


@pytest.fixture
def verb(mb) -> Callable[..., list[dict[str, str | int | bool]]]:
    """Run an `mb` verb that must succeed, answering the rows it printed."""

    def run(*arguments: str) -> list[dict[str, str | int | bool]]:
        ran = mb(*arguments, "--json")
        assert ran.code == 0, ran.said
        return json.loads(ran.out)

    return run


@pytest.fixture(params=["absolute", "relative"])
def spelled(request: pytest.FixtureRequest, evidence: Path, node: str) -> str:
    """The evidence folder as a verb is given it: in full, or relative to where it runs."""
    return str(evidence) if request.param == "absolute" else f"{node}/evidence"


@pytest.fixture
def duplicates(workspace: Path, node: str) -> tuple[Path, bytes]:
    """An artifacts folder of an artifact's copies, answering it with the bytes they should hold.

    Runs one to three settled hours ago with the bytes the name digests, run four settled with
    other bytes under that name, and run five written just now.
    """
    artifacts = workspace / node / "evidence" / "artifacts"
    data = bytes(range(250)) * 20
    settled = time.time() - 7200
    for run, payload in [("run1", data), ("run2", data), ("run3", data), ("run4", data[::-1])]:
        copy = artifacts / run / "objects" / hashlib.sha256(data).hexdigest()
        copy.parent.mkdir(parents=True)
        copy.write_bytes(payload)
        os.utime(copy, (settled, settled))
    flight = artifacts / "run5" / "objects" / hashlib.sha256(data).hexdigest()
    flight.parent.mkdir(parents=True)
    flight.write_bytes(data)
    return artifacts, data


def test_evidence_survives_a_round_trip_through_the_lake_verbs(
    verb, evidence, node, spelled, files_of
) -> None:
    before = files_of(evidence)
    assert [verb("lake", "ingest", spelled), verb("lake", "ingest", spelled)] == [
        [{"files": 4, "indexed": 4, "objects": 4, "size": sum(map(len, before.values()))}],
        [{"files": 4, "indexed": 0, "objects": 0, "size": 0}],
    ]
    assert verb("lake", "dedup", spelled, "--dry-run")[0]["linked"] == 0
    assert verb("lake", "evict", spelled) == [{"evicted": 4}]
    assert len(verb("lake", "materialize", spelled)) == 4
    assert files_of(evidence) == before
    assert verb("lake", "check") == []
    assert verb("query", _LANDED) == [
        {"project": "lab", "node": "law", "run": "", "landed": True},
        {"project": "lab", "node": "law", "run": "run1", "landed": True},
    ]
    assert verb("query", _IMPORTS) == [
        {"source": f"{node}/evidence", "destination": "evidence_log", "rows": rows}
        for rows in (4, 0)
    ]


@pytest.mark.parametrize(
    ("command", "climbing"),
    [
        pytest.param("ingest", False, id="ingest"),
        pytest.param("evict", False, id="evict"),
        pytest.param("materialize", False, id="materialize"),
        pytest.param("dedup", False, id="dedup"),
        pytest.param("dedup", True, id="dedup-the-parent-folder"),
    ],
)
def test_a_lake_verb_refuses_a_path_outside_the_workspace(
    mb, command: str, climbing: bool, tmp_path_factory: pytest.TempPathFactory
) -> None:
    stranger = tmp_path_factory.mktemp("stranger") / "file.bin"
    stranger.write_bytes(b"x")
    ran = mb("lake", command, ".." if climbing else str(stranger))
    assert ran.code == 1 and "lies outside the workspace" in ran.said


def test_dedup_links_settled_duplicates_to_one_copy(
    verb, node: str, duplicates: tuple[Path, bytes]
) -> None:
    artifacts, data = duplicates
    name, folder = hashlib.sha256(data).hexdigest(), f"{node}/evidence"
    done = {"path": folder, "files": 5, "hashed": 4, "linked": 2, "damaged": 1}
    assert verb("lake", "dedup", folder) == [{**done, "before": 25_000, "after": 15_000}]
    assert len({(artifacts / f"run{run}" / "objects" / name).stat().st_ino for run in "123"}) == 1
    assert (artifacts / "run5" / "objects" / name).stat().st_nlink == 1
    assert (artifacts / "run4" / "objects" / name).read_bytes() == data[::-1]
    [again] = verb("lake", "dedup", folder)
    assert (again["linked"], again["before"], again["after"]) == (0, 15_000, 15_000)


def test_check_finds_a_data_file_damaged_on_disk(mb, keeper: Evidence, evidence: Path) -> None:
    keeper.ingest([evidence])
    [data] = [path for path in keeper.lake.data.rglob("*.parquet") if "blobs" in path.parts]
    damaged = bytearray(data.read_bytes())
    middle = len(damaged) // 2
    damaged[middle : middle + 64] = bytes(byte ^ 0xFF for byte in damaged[middle : middle + 64])
    data.write_bytes(bytes(damaged))
    ran = mb("lake", "check", "--json")
    assert ran.code == 1, ran.said
    assert {finding["kind"] for finding in json.loads(ran.out)} <= {"evidence", "unreadable"}
