import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from mainboard.state import Evidence, EvidenceTree, Lake, Relations, write_parquet
from mainboard.state.storage import blobs as blobs_module
from mainboard.trials import Artifact

_PARQUET = "application/vnd.apache.parquet"


@pytest.fixture
def node() -> str:
    """The workspace-relative folder of the node whose evidence the tests keep."""
    return "research/lab/datasets/experiments/law"


@pytest.fixture
def reference(workspace: Path, node: str) -> Artifact:
    """The reference to a Parquet object stored under its digest in the node's evidence."""
    payload = workspace / node / "evidence" / "artifacts" / "run1" / "objects" / "table"
    payload.parent.mkdir(parents=True)
    write_parquet(Relations().rows([{"bits": bits} for bits in (1, 2, 3)]), payload)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    pinned = Artifact(
        path=f"datasets/experiments/law/evidence/artifacts/run1/objects/{digest}",
        sha256=digest,
        size=payload.stat().st_size,
        media_type=_PARQUET,
    )
    payload.rename(payload.with_name(digest))
    return pinned


@pytest.fixture
def pinned(reference: Artifact) -> str:
    """The digest of the object the node's receipt pins, which names its file."""
    return reference.sha256


@pytest.fixture
def evidence(workspace: Path, node: str, reference: Artifact) -> Path:
    """A node's evidence: a receipt partition, an object it pins and two plain files.

    The object is content-addressed; the plain files are an empty one and one of 70,000 bytes.
    """
    root = workspace / node / "evidence"
    part = root / "receipts" / "run=run1" / "part-00000.parquet"
    part.parent.mkdir(parents=True)
    receipt = {"run": "run1", "lane": "law", "key": "a", "artifacts": reference.model_dump_json()}
    write_parquet(Relations().rows([receipt]), part)
    (root / "empty.txt").write_bytes(b"")
    (root / "noise.bin").write_bytes(b"noise" * 14_000)
    return root


@pytest.fixture
def keeper(workspace: Path) -> Evidence:
    """The workspace's evidence, in chunks of the default size."""
    return Evidence(Lake.at(workspace))


@pytest.fixture
def kept(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Evidence:
    """The center workspace's evidence, in chunks of four bytes."""
    (workspace / ".git").mkdir()
    monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", 4)
    center = Evidence(Lake.at(workspace))
    monkeypatch.setattr(EvidenceTree, "keepers", property(lambda tree: [center]))
    return center


@pytest.fixture
def stored(kept: Evidence) -> Callable[[str, bytes], Path]:
    """Write a file of the workspace holding a payload and ingest it, answering its path."""

    def store(name: str, payload: bytes) -> Path:
        path = kept.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        kept.ingest([path])
        return path

    return store


@pytest.fixture
def harm(kept: Evidence) -> Callable[..., None]:
    """Change the lake's rows behind its writers' back, the way a failing disk would."""

    def damage(*statements: str) -> None:
        with kept.lake.open(write=True) as connection:
            for statement in statements:
                connection.execute(statement)

    return damage


@pytest.fixture
def files_of() -> Callable[[Path], dict[str, bytes]]:
    """Every file under a folder with its bytes, by relative path."""

    def read(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    return read


@pytest.fixture
def digest_of() -> Callable[[bytes], str]:
    """The SHA-256 hex digest of bytes, which names an object in the lake."""
    return lambda payload: hashlib.sha256(payload).hexdigest()
