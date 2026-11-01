import io
import itertools
import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from zipfile import ZIP_ZSTANDARD, ZipFile

import pytest
from pydantic import Field

from mainboard import MissionError, Project
from mainboard.dispatch import SshTransport
from mainboard.dispatch.collection.collector import Collector
from mainboard.dispatch.collection.pack import HELD, MORE, Center, pack
from mainboard.state import Evidence, EvidenceTree

_NO_DIGESTS: Mapping[str, str] = MappingProxyType({})
_CACHE = ".mb/run/digests.json"


def _index(keeper: Evidence, node: str) -> dict[str, str]:
    """The digest the lake indexes for each path under `node`."""
    return {row.path: row.sha256 for row in keeper.indexed(node)}


@pytest.fixture
def collector(keeper: Evidence, *, evidence: Path, workspace: Path) -> Collector:
    """A collector over a workspace whose lake keeps the node's evidence and whose tree lost it."""
    keeper.ingest([evidence])
    keeper.evict(keeper.finalize([evidence]))
    return Collector(workspace)


@pytest.fixture
def transfer(tmp_path: Path) -> Callable[[Mapping[str, bytes], Mapping[str, str]], Path]:
    """Write a host's transfer as `pack` does: zstd members, then what the host left unsent."""
    numbers = itertools.count()

    def write(files: Mapping[str, bytes], held: Mapping[str, str]) -> Path:
        path = tmp_path / f"transfer-{next(numbers)}.zip"
        with ZipFile(path, "w", ZIP_ZSTANDARD) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
            archive.writestr(HELD, json.dumps(held))
        return path

    return write


@pytest.fixture
def leftovers(workspace: Path) -> Path:
    """The staging folder, holding one folder of a pass that died and one of this live process."""
    staging = Project().out(workspace) / "tmp"
    for left in (staging / "collect-999999999-dead", staging / f"collect-{os.getpid()}-live"):
        left.mkdir(parents=True)
    return staging


@pytest.fixture
def host(tmp_path: Path, *, evidence: Path, node: str, pinned: str) -> Path:
    """A job host's workspace: the node's evidence, its pinned object again and a live event log.

    The object sits again under a second run, as a rerun of the same trial writes it.
    """
    root = tmp_path / "host"
    run = root / node / "evidence" / "artifacts" / "run2"
    shutil.copytree(evidence, root / node / "evidence")
    (run / "objects").mkdir(parents=True)
    shutil.copy(evidence / "artifacts" / "run1" / "objects" / pinned, run / "objects" / pinned)
    (run / "trial" / "events").mkdir(parents=True)
    (run / "trial" / "events" / "live.ndjson").write_bytes(b'{"offset": 0}\n')
    return root


@pytest.fixture
def packed(monkeypatch: pytest.MonkeyPatch, node: str) -> Callable[..., ZipFile]:
    """Run `pack` as a host does, answering the archive it streams."""

    def run(
        root: Path,
        *,
        kept: Mapping[str, str] = _NO_DIGESTS,
        known: Mapping[str, str] = _NO_DIGESTS,
        cache: str = "",
        allowance: int = sys.maxsize,
    ) -> ZipFile:
        sink = io.BytesIO()
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=sink))
        pack(str(root), relative=node, center=Center(kept, known, allowance), cache=cache)
        return ZipFile(io.BytesIO(sink.getvalue()))

    return run


def test_collection_keeps_transfers_in_the_lake_never_the_tree(
    collector: Collector,
    keeper: Evidence,
    transfer: Callable[[Mapping[str, bytes], Mapping[str, str]], Path],
    leftovers: Path,
    workspace: Path,
    node: str,
    pinned: str,
    digest_of: Callable[[bytes], str],
) -> None:
    copy, new = f"{node}/evidence/artifacts/run2/objects/{pinned}", f"{node}/evidence/new.txt"
    before = _index(keeper, node)
    assert collector.merge(transfer({new: b"fresh"}, {copy: pinned}), path=node) == 2
    assert not (workspace / new).exists() and not (workspace / copy).exists()
    assert _index(keeper, node) == {**before, new: digest_of(b"fresh"), copy: pinned}
    assert EvidenceTree(workspace).read(workspace / copy, pinned)
    assert list(leftovers.iterdir()) == [leftovers / f"collect-{os.getpid()}-live"]


def test_a_host_copy_with_other_bytes_goes_aside_and_the_centers_keeps_its_path(
    collector: Collector,
    keeper: Evidence,
    transfer: Callable[[Mapping[str, bytes], Mapping[str, str]], Path],
    node: str,
    digest_of: Callable[[bytes], str],
) -> None:
    noise, later = f"{node}/evidence/noise.bin", f"{node}/evidence/later.txt"
    other = digest_of(b"other bytes")
    before = _index(keeper, node)
    assert collector.merge(transfer({noise: b"other bytes", later: b"more"}, {}), path=node) == 2
    after = _index(keeper, node)
    assert after[noise] == before[noise]
    assert after[f"{node}/.conflicts/{other[:8]}/evidence/noise.bin"] == other
    assert after[later] == digest_of(b"more")


def test_a_transfer_naming_an_object_the_lake_lacks_is_refused(
    collector: Collector,
    transfer: Callable[[Mapping[str, bytes], Mapping[str, str]], Path],
    node: str,
) -> None:
    with pytest.raises(MissionError, match="keeps no object"):
        collector.merge(transfer({}, {f"{node}/evidence/missing.txt": "0" * 64}), path=node)


def test_a_host_sends_only_what_the_lake_lacks(
    host: Path,
    packed: Callable[..., ZipFile],
    keeper: Evidence,
    evidence: Path,
    node: str,
    pinned: str,
) -> None:
    keeper.ingest([evidence])
    kept = _index(keeper, node)
    again = f"{node}/evidence/artifacts/run2/objects/{pinned}"
    with packed(host, kept=kept, cache=_CACHE) as archive:
        [snapshot] = [name for name in archive.namelist() if "/events/collected-" in name]
        assert sorted(archive.namelist()) == sorted([snapshot, HELD])
        assert {info.compress_type for info in archive.infolist()} == {ZIP_ZSTANDARD}
        assert json.loads(archive.read(HELD)) == {again: pinned}
    with packed(host, kept=kept, known={snapshot: ""}, cache=_CACHE) as archive:
        assert archive.namelist() == [HELD]


def test_a_results_path_never_written_packs_nothing(
    packed: Callable[..., ZipFile], tmp_path: Path
) -> None:
    with packed(tmp_path) as archive:
        assert archive.namelist() == []


@pytest.fixture
def folder(tmp_path: Path, node: str) -> Path:
    """A host workspace whose node holds three files of ten bytes, a.bin to c.bin."""
    (tmp_path / node).mkdir(parents=True)
    for name in "abc":
        (tmp_path / node / f"{name}.bin").write_bytes(name.encode() * 10)
    return tmp_path


def test_a_pass_stops_before_the_file_that_passes_its_allowance_and_the_next_sends_the_rest(
    packed: Callable[..., ZipFile],
    folder: Path,
    node: str,
    digest_of: Callable[[bytes], str],
) -> None:
    with packed(folder, allowance=25) as archive:
        assert archive.namelist() == [f"{node}/a.bin", f"{node}/b.bin", MORE]
    sent = {f"{node}/{name}.bin": digest_of(name.encode() * 10) for name in "ab"}
    with packed(folder, known=sent, allowance=25) as archive:
        assert archive.namelist() == [f"{node}/c.bin"]


def test_a_file_past_the_whole_allowance_is_refused(
    packed: Callable[..., ZipFile], folder: Path
) -> None:
    with pytest.raises(ValueError, match="more than the 5 the center"):
        packed(folder, allowance=5)


def test_a_transfer_ended_at_the_hosts_allowance_is_kept_then_refused(
    collector: Collector,
    keeper: Evidence,
    transfer: Callable[[Mapping[str, bytes], Mapping[str, str]], Path],
    node: str,
    digest_of: Callable[[bytes], str],
) -> None:
    new = f"{node}/evidence/new.txt"
    archive = transfer({new: b"fresh"}, {})
    with ZipFile(archive, "a", ZIP_ZSTANDARD) as appended:
        appended.writestr(MORE, "{}")
    with pytest.raises(MissionError, match="more wait on the host"):
        collector.merge(archive, path=node)
    assert _index(keeper, node)[new] == digest_of(b"fresh")


class _Host(SshTransport):
    """An ssh transport whose host answers the transfer in `reply` and keeps the scripts it got."""

    reply: Path
    scripts: list[str] = Field(default_factory=list)

    def command(self, host: str) -> tuple[str, ...]:
        return ("ssh", host)

    def run(
        self,
        command: Sequence[str],
        host: str,
        *,
        operation: str,
        input_text: str | None = None,
        output: Path | None = None,
    ) -> str:
        self.scripts.append(input_text or "")
        shutil.copy(self.reply, output or "")
        return ""


@pytest.fixture
def ssh(
    collector: Collector, transfer: Callable[[Mapping[str, bytes], Mapping[str, str]], Path]
) -> _Host:
    """The host a pull reaches: it answers an empty transfer."""
    host = _Host(reply=transfer({}, {}))
    collector.transport = host
    return host


def test_a_pull_asks_for_a_third_of_the_disk_above_the_floor(
    collector: Collector, ssh: _Host, monkeypatch: pytest.MonkeyPatch, node: str
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=33 * 2**30))
    collector.pull("gold", root="/remote", path=node, python="python")
    assert "allowance=1073741824)" in ssh.scripts[0]


def test_a_pull_never_asks_for_more_than_one_pass_moves_in_its_deadline(
    collector: Collector, ssh: _Host, monkeypatch: pytest.MonkeyPatch, node: str
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=2**50))
    collector.pull("gold", root="/remote", path=node, python="python")
    assert f"allowance={16 * 2**30})" in ssh.scripts[0]


def test_a_pull_below_the_free_disk_floor_asks_the_host_for_nothing(
    collector: Collector, ssh: _Host, monkeypatch: pytest.MonkeyPatch, node: str
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=2**30))
    with pytest.raises(MissionError, match="free disk"):
        collector.pull("gold", root="/remote", path=node, python="python")
    assert ssh.scripts == []
