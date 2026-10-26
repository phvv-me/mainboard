import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from mainboard import MissionError
from mainboard.state import Evidence, Lake

_NAMES = st.sampled_from(["a.bin", "b.bin", "sub/c.bin", "sub/d.bin"])
_PAYLOADS = st.sampled_from([b"", b"one", b"two", b"three three"])


class EvidenceLifecycle(RuleBasedStateMachine):
    """One workspace's evidence folder under any order of writes, deletes, ingests, evictions
    and restores, checked against a model of the tree and of what the lake indexes."""

    @initialize()
    def start_with_kept_files(self) -> None:
        for name, payload in [("a.bin", b"one"), ("sub/c.bin", b"two")]:
            self.write(name, payload)
        self.ingest()

    @invariant()
    def the_lake_indexes_and_keeps_what_was_ingested(self) -> None:
        indexed = {
            row.path.removeprefix("evidence/"): row.sha256 for row in self.evidence.indexed()
        }
        digests = {name: hashlib.sha256(data).hexdigest() for name, data in self.index.items()}
        assert indexed == digests

    @invariant()
    def the_tree_is_what_the_model_says(self) -> None:
        found = {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        assert found == self.tree

    @precondition(lambda self: self.tree)
    @rule()
    def evict(self) -> None:
        evicted = self.evidence.evict(self.evidence.finalize([self.root]))
        gone = {name for name, payload in self.tree.items() if self.index.get(name) == payload}
        assert {path.relative_to(self.root).as_posix() for path in evicted} == gone
        self.tree = {name: payload for name, payload in self.tree.items() if name not in gone}

    @precondition(lambda self: self.tree)
    @rule()
    def ingest(self) -> None:
        self.evidence.ingest([self.root])
        self.index |= self.tree

    @precondition(lambda self: self.tree.items() != self.index.items())
    @rule()
    def restore(self) -> None:
        clash = {
            name for name, payload in self.tree.items() if self.index.get(name, payload) != payload
        }
        if clash:
            with pytest.raises(MissionError, match="refusing to overwrite"):
                self.evidence.materialize([self.root])
            return
        written = self.evidence.materialize([self.root])
        assert {
            path.relative_to(self.root).as_posix() for path in written
        } == self.index.keys() - self.tree.keys()
        self.tree |= self.index

    @rule(name=_NAMES)
    def delete(self, name: str) -> None:
        (self.root / name).unlink(missing_ok=True)
        self.tree.pop(name, None)

    @rule(name=_NAMES, payload=_PAYLOADS)
    def write(self, name: str, payload: bytes) -> None:
        (self.root / name).parent.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_bytes(payload)
        self.tree[name] = payload

    def __init__(self) -> None:
        super().__init__()
        self.directory = TemporaryDirectory()
        base = Path(self.directory.name)
        (base / "mb.toml").write_text('[workspace]\nname = "it"\n', encoding="utf-8")
        self.evidence = Evidence(Lake.at(base))
        self.root = base / "evidence"
        self.tree: dict[str, bytes] = {}
        self.index: dict[str, bytes] = {}

    def teardown(self) -> None:
        assert self.evidence.verify() == []
        self.directory.cleanup()


TestEvidenceLifecycle = EvidenceLifecycle.TestCase
TestEvidenceLifecycle.settings = settings(
    database=None, deadline=None, derandomize=True, max_examples=8, stateful_step_count=10
)
