import hashlib
import re
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import duckdb
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard.state import Lake, schema
from mainboard.state.storage import blobs as blobs_module
from mainboard.state.storage.blobs import Audit, Blobs

_CHUNKS = (
    "SELECT b.ordinal, b.bytes, c.md5 FROM lake.blobs b "
    "JOIN lake.checksums c ON c.sha256 = b.sha256 AND c.ordinal = b.ordinal "
    "WHERE b.sha256 = ? ORDER BY b.ordinal"
)
_DEPTH = "SELECT current_setting('max_expression_depth')"
_OBJECTS = 1600
_DIGEST = st.from_regex(r"[0-9a-f]{64}", fullmatch=True)
_MALFORMED = st.one_of(st.text(), st.from_regex(r"[0-9a-fA-F]{60,68}\n?", fullmatch=True)).filter(
    lambda text: re.fullmatch(r"[0-9a-f]{64}", text) is None
)
_PURE = settings(database=None, deadline=None, max_examples=100)
_STORED = settings(database=None, deadline=None, max_examples=15)
_LAKE = settings(
    database=None,
    deadline=None,
    max_examples=12,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
_AUDITS = st.integers(min_value=-1, max_value=40).flatmap(
    lambda last: st.builds(
        Audit,
        chunks=st.integers(min_value=0, max_value=last + 1),
        last=st.just(last),
        size=st.integers(min_value=0, max_value=500),
        intact=st.booleans(),
    )
)


@contextmanager
def _lake() -> Generator[tuple[Lake, Path]]:
    """A new lake in a workspace of its own, for one example of a property."""
    with TemporaryDirectory() as root:
        yield Lake.at(Path(root)).ready(), Path(root)


def _objects(payloads: Iterable[bytes]) -> dict[str, bytes]:
    """Each payload under its SHA-256."""
    return {hashlib.sha256(payload).hexdigest(): payload for payload in payloads}


@pytest.fixture
def pool(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Lake, dict[str, bytes]]:
    """A lake of 1,600 objects in three-byte chunks, one staged twice, and two kept unchunked."""
    monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", 3)
    lake = Lake.at(workspace).ready()
    blobs = Blobs(lake)
    chunked = _objects([f"object-{n}".encode() for n in range(_OBJECTS)])
    legacy = _objects([b"legacy", b"kept before ordinals"])
    lake.transact(lambda connection: blobs.stage(connection, chunked))
    lake.transact(lambda connection: blobs.stage(connection, dict([*chunked.items()][:1])))
    lake.append(schema.blobs, [{"sha256": digest, "bytes": legacy[digest]} for digest in legacy])
    return lake, chunked | legacy


@pytest.fixture
def lake(workspace: Path) -> Lake:
    """A lake that is never created, for what must be refused before it is read."""
    return Lake.at(workspace)


@_STORED
@given(
    payload=st.binary(max_size=300),
    size=st.integers(min_value=1, max_value=64),
    staging=st.sampled_from(["bytes", "file"]),
)
@example(payload=b"", size=1, staging="file")
@example(payload=b"abcdef", size=3, staging="bytes")
@example(payload=b"abc", size=64, staging="file")
@example(payload=b"abcdefgh", size=3, staging="file")
@example(payload=b"abcdef", size=3, staging="file")
def test_an_object_is_kept_in_ordered_chunks_with_a_checksum_each(
    payload: bytes, size: int, staging: str
) -> None:
    digest, pieces = (
        hashlib.sha256(payload).hexdigest(),
        [payload[at : at + size] for at in range(0, len(payload) or 1, size)],
    )
    with _lake() as (lake, root), patch.object(blobs_module, "_CHUNK_BYTES", size):
        (root / "source.bin").write_bytes(payload)
        source = root / "source.bin" if staging == "file" else payload
        lake.transact(lambda connection: Blobs(lake).stage(connection, {digest: source}))
        with lake.open() as connection:
            assert connection.execute(_CHUNKS, [digest]).fetchall() == [
                (at, piece, hashlib.md5(piece, usedforsecurity=False).hexdigest())
                for at, piece in enumerate(pieces)
            ]
            assert Blobs(lake).read(connection, {digest}) == {digest: payload}
            assert Blobs(lake).write(connection, digest, root / "out.bin") and (
                (root / "out.bin").read_bytes() == payload
            )
            assert not Blobs(lake).write(connection, "0" * 64, root / "absent.bin")


@_LAKE
@given(
    sampled=st.lists(st.integers(min_value=0, max_value=_OBJECTS + 1), max_size=30),
    unheld=st.lists(_DIGEST, max_size=3),
    bulk=st.integers(min_value=0, max_value=_OBJECTS + 100),
)
@example(sampled=[], unheld=[], bulk=_OBJECTS + 100)
@example(sampled=[0, 0], unheld=[], bulk=_OBJECTS + 1)
@example(sampled=[], unheld=["0" * 64], bulk=0)
@example(sampled=[0], unheld=["0" * 64], bulk=_OBJECTS + 100)
def test_membership_answers_the_requested_digests_the_lake_holds(
    pool: tuple[Lake, dict[str, bytes]], sampled: Sequence[int], unheld: Sequence[str], bulk: int
) -> None:
    lake, payloads = pool
    digests = list(payloads)
    requested = [
        *(digests[at % len(digests)] for at in range(bulk)),
        *(digests[at] for at in sampled),
        *unheld,
    ]
    expected = {digest: payloads[digest] for digest in requested if digest in payloads}
    with lake.open() as connection:
        depth = connection.execute(_DEPTH).fetchone()
        assert Blobs(lake).held(connection, requested) == set(expected)
        assert Blobs(lake).read(connection, requested) == expected
        assert connection.execute(_DEPTH).fetchone() == depth


@_PURE
@given(valid=st.sets(_DIGEST, max_size=3), malformed=_MALFORMED)
@example(valid=set(), malformed="' OR true --")
@example(valid=set(), malformed="")
@example(valid=set(), malformed="g" * 64)
@example(valid=set(), malformed="0" * 63)
@example(valid=set(), malformed="0" * 64 + "\n")
@pytest.mark.parametrize("answer", [Blobs.held, Blobs.read], ids=["held", "read"])
def test_a_malformed_digest_is_refused_before_any_query(
    lake: Lake,
    answer: Callable[[Blobs, duckdb.DuckDBPyConnection, set[str]], set[str] | dict[str, bytes]],
    valid: set[str],
    malformed: str,
) -> None:
    closed = duckdb.connect()
    closed.close()
    with pytest.raises(ValueError, match="not a SHA-256 digest"):
        answer(Blobs(lake), closed, {*valid, malformed})
    assert not answer(Blobs(lake), closed, set())


@_PURE
@given(audit=_AUDITS, expected=st.integers(min_value=0, max_value=500))
def test_an_audit_names_the_first_thing_keeping_an_object_from_being_whole(
    audit: Audit, expected: int
) -> None:
    held = audit.last + 1
    first = next(
        (
            text
            for failing, text in [
                (audit.chunks != held, f"{held - audit.chunks} of its {held} chunks are missing"),
                (audit.size != expected, f"the lake holds {audit.size} of its {expected} bytes"),
                (not audit.intact, "a chunk fails its checksum"),
            ]
            if failing
        ),
        "",
    )
    assert audit.fault(expected) == first
