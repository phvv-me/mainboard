import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard.state import DirectoryReplica, Evidence
from mainboard.state.evidence import core as evidence_core
from mainboard.state.evidence.keeping import windows
from mainboard.state.storage.blobs import Blobs


@pytest.fixture
def ingested(keeper: Evidence, evidence: Path) -> Evidence:
    """The workspace's evidence with the node's four files kept in its lake."""
    keeper.ingest([evidence])
    return keeper


@settings(
    max_examples=12,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(window=st.one_of(st.integers(min_value=0, max_value=4000), st.integers(0, 80_000)))
@example(window=0)
@example(window=80_000)
def test_a_replica_holds_each_object_once_whatever_the_window_that_reads_it(
    ingested: Evidence, window: int
) -> None:
    sizes = {row.sha256: row.size for row in ingested.indexed()}
    with (
        TemporaryDirectory() as root,
        patch.object(evidence_core, "WINDOW_BYTES", window),
        patch.object(Blobs, "read", autospec=True, side_effect=Blobs.read) as read,
        patch.object(Blobs, "chunks", autospec=True, side_effect=Blobs.chunks) as alone,
    ):
        replica = DirectoryReplica(Path(root))
        assert [ingested.replicate(replica), ingested.replicate(replica)] == [len(sizes), 0]
        small = sorted(digest for digest, size in sizes.items() if size <= window)
        assert read.call_count == len(list(windows(small, sizes.__getitem__, window)))
        assert sorted(call.args[2] for call in alone.call_args_list) == sorted(
            digest for digest, size in sizes.items() if size > window
        )
        assert all(
            hashlib.sha256((Path(root) / digest[:2] / digest).read_bytes()).hexdigest() == digest
            for digest in sizes
        )
        index = (Path(root) / "index.jsonl").read_text("utf-8").splitlines()
        assert len(index) == len(ingested.indexed()) and not list(Path(root).rglob("*.partial"))
