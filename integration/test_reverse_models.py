import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mainboard.probe.reverse.models import (
    Artifact,
    ArtifactKind,
    Confidence,
    Evidence,
    Fact,
    JsonValue,
    Kernel,
    KernelInventory,
    Provenance,
)

from .reverse_fixtures import evidence

# Keys and strings stay short and recursion shallow: the property is about canonical encoding,
# not volume, and an unbounded recursive strategy trips Hypothesis's input-entropy health check.
_TEXT = st.text(max_size=16)
_JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | _TEXT,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(_TEXT, children, max_size=3),
    max_leaves=6,
)


@settings(database=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.dictionaries(_TEXT, _JSON, max_size=4))
def test_evidence_roundtrip_and_independent_cache_address(params: dict[str, JsonValue]) -> None:
    original = evidence(params)
    restored = Evidence[KernelInventory].model_validate_json(original.model_dump_json())
    assert restored == original
    independently_encoded = json.dumps(
        [original.artifact.sha256, "inventory", params, "fixture", "13.4.0"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert original.key() == hashlib.sha256(independently_encoded).hexdigest()
    assert original.key() == evidence(dict(reversed(tuple(params.items())))).key()


def test_nested_options_are_canonical_and_paths_do_not_identify_analyses() -> None:
    left = evidence({"options": {"a": 1, "b": 2}})
    right = evidence({"options": {"b": 2, "a": 1}})
    assert left.key() == right.key()
    moved = left.model_copy(
        update={"artifact": left.artifact.model_copy(update={"path": "/other/lib.so"})}
    )
    assert left.key() == moved.key()
    assert left.key() != evidence({"options": {"a": 1}}).key()


@pytest.mark.parametrize("field", ["artifact", "operation", "provider", "tool_version"])
def test_every_identity_coordinate_changes_the_key(field: str) -> None:
    original = evidence()
    changes = {
        "artifact": original.artifact.model_copy(update={"sha256": "b" * 64}),
        "operation": "resource",
        "provider": "other",
        "tool_version": "13.8",
    }
    assert original.model_copy(update={field: changes[field]}).key() != original.key()


def test_profile_is_required_and_must_match_the_evidence() -> None:
    serialized = evidence().model_dump()
    del serialized["analysis_profile"]
    with pytest.raises(ValidationError, match="analysis_profile"):
        Evidence[KernelInventory].model_validate(serialized)
    serialized = evidence().model_dump()
    serialized["params"] = {"changed": True}
    with pytest.raises(ValidationError, match="profile"):
        Evidence[KernelInventory].model_validate(serialized)
    with pytest.raises(ValueError):
        evidence({"bad": float("nan")})


def test_inventory_diff_reports_exact_metadata_changes() -> None:
    unchanged = Kernel(name="same", arch="sm_90a")
    older = Kernel(name="changed", arch="sm_90a")
    newer = older.model_copy(
        update={
            "registers": Fact[int | None](
                value=32,
                confidence=Confidence.observed,
                provenance=Provenance.IMPORTED,
            )
        }
    )
    removed, added = Kernel(name="old", arch="sm_90a"), Kernel(name="new", arch="sm_100")
    before = KernelInventory(kernels=(unchanged, older, removed), arches=("sm_90a",))
    after = KernelInventory(kernels=(added, newer, unchanged), arches=("sm_90a", "sm_100"))
    diff = before.diff(after)
    assert diff.added == (added,)
    assert diff.removed == (removed,)
    assert [(change.before, change.after) for change in diff.changed] == [(older, newer)]
    assert diff.added_arches == ("sm_100",)
    assert not before.diff(before).changed
    assert max(Provenance.IMPORTED, Provenance.ANALYSIS) == Provenance.ANALYSIS


@settings(database=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sets(st.text(min_size=1, max_size=16), max_size=15))
def test_inventory_diff_is_independent_of_order(names: set[str]) -> None:
    kernels = tuple(Kernel(name=name, arch="sm_90") for name in sorted(names))
    before = KernelInventory(kernels=kernels, arches=("sm_90",))
    diff = before.diff(KernelInventory(kernels=tuple(reversed(kernels)), arches=("sm_90",)))
    assert not diff.added and not diff.removed and not diff.changed


def test_artifact_hash_and_cuda_elf_detection(tmp_path: Path) -> None:
    library = tmp_path / "fixture.cubin"
    payload = b"\x7fELF\x02\x01" + bytes(12) + (190).to_bytes(2, "little") + bytes(44)
    library.write_bytes(payload)
    found = Artifact.at(library)
    assert found.kind == ArtifactKind.cubin
    assert found.size == len(payload)
    assert found.sha256 == hashlib.sha256(payload).hexdigest()
