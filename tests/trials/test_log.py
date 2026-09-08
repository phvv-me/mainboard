"""The injected facade preserves existing verdicts and durable, independently readable data."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from mainboard import span
from mainboard.profile import Feature, Profile
from mainboard.profile.profiler import Collection
from mainboard.trials import Artifact, Declaration, Log, Session
from mainboard.trials.artifacts import Artifacts

from .support import Item


@pytest.mark.parametrize("fault", ["missing", "changed", "escape", "other-project"])
def test_fetched_artifacts_require_the_declared_bytes_inside_the_selected_project(
    tmp_path: Path, fault: str
) -> None:
    directory = tmp_path / "research/one/datasets/node"
    writer = Artifacts(tmp_path / "research/one", directory)
    reference = writer.write(b"data", media_type="application/octet-stream")
    relative = reference.path
    if fault == "missing":
        (tmp_path / "research/one" / relative).unlink()
    elif fault == "changed":
        (tmp_path / "research/one" / relative).write_bytes(b"oops")
    elif fault == "escape":
        reference = reference.model_copy(update={"path": "../outside"})
    else:
        directory = tmp_path / "research/two/datasets/node"
    line = json.dumps({"trial_receipt": {"artifacts": {"value": reference.model_dump()}}})
    with pytest.raises((ValueError, OSError)):
        Artifacts.verify([line], directory=directory, boundary=tmp_path)


def test_log_derives_identity_preserves_verdict_and_records_bound_messages(
    session: Session, tmp_path: Path
) -> None:
    item = Item(
        "alpha/test_law.py::test_holds[qwen]", tmp_path / "alpha/test_law.py", {"model": "qwen"}
    )
    trial = session.trial(item)
    log = Log(trial)
    log.bind(phase="decode").info("measured {} tokens", 16)
    log.bind(phase="prefill").metrics(seconds=0.1)
    log.validated("the original gate held", ratio=1.5)
    frames = log.spool.frames_from(0)
    assert frames[0].payload["data"]["params"] == {"model": "qwen"}
    assert frames[1].payload["data"]["text"] == "measured 16 tokens"
    assert frames[1].payload["data"]["metadata"] == {"phase": "decode"}
    assert frames[2].payload["metadata"] == {"phase": "prefill"}
    assert trial.settled == "validated"
    with pytest.raises(ValueError, match="provenance"):
        log.bind(run="forged")
    log.close(passed=True)
    rows = session.declared.universe.dataset("alpha").rows(session.run)
    assert rows[0]["measured"] == {"ratio": 1.5}


def test_tables_are_parquet_and_refs_reject_changed_or_escaping_inputs(
    session: Session, tmp_path: Path
) -> None:
    log = Log(session.trial(Item("alpha/t.py::table", tmp_path / "alpha/t.py")))
    reference = log.table([{"ratio": 1.5}], schema_name="test.ratio.v1")
    assert pl.read_parquet(BytesIO(reference.read(tmp_path))).to_dicts() == [{"ratio": 1.5}]
    assert reference.schema_name == "test.ratio.v1"
    assert log.table([{"ratio": 1.5}]).sha256 == reference.sha256
    (tmp_path / reference.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        reference.read(tmp_path)
    with pytest.raises(ValueError):
        reference.model_copy(update={"path": "../escape"}).read(tmp_path)
    log.close(passed=False)


def test_declared_inputs_are_hash_checked_and_recorded(
    declared: Declaration, probed: None, tmp_path: Path
) -> None:
    content = b"frozen predecessor"
    (tmp_path / "input.bin").write_bytes(content)
    pinned = Artifact(
        path="input.bin", sha256=hashlib.sha256(content).hexdigest(), size=len(content)
    )
    session = Session(declared.model_copy(update={"inputs": {"predecessor": pinned}}))
    log = Log(session.trial(Item("alpha/t.py::read", tmp_path / "alpha/t.py")))
    assert log.read("predecessor") == content
    with pytest.raises(KeyError):
        log.read("latest")
    assert log.spool.frames_from(0)[-1].payload["data"]["sha256"] == pinned.sha256
    log.close(passed=True)


def test_profile_attaches_partial_evidence_without_swallowing_the_failure(
    session: Session, tmp_path: Path
) -> None:
    log = Log(session.trial(Item("alpha/t.py::profile", tmp_path / "alpha/t.py")))
    with (
        pytest.raises(RuntimeError, match="body failed"),
        log.profile(collection=Collection(features=Feature.SPANS)),
        span("partial"),
    ):
        raise RuntimeError("body failed")
    frames = log.spool.frames_from(0)
    artifact = Artifact.model_validate(
        {key: frames[1].payload["data"][key] for key in Artifact.model_fields}
    )
    profile = Profile.model_validate_json(artifact.read(tmp_path))
    assert profile.summaries[0].name == "partial"
    assert frames[2].payload["data"]["completed"] is False
    log.close(passed=False)


def test_artifact_publication_never_overwrites_corrupted_content(tmp_path: Path) -> None:
    artifacts = Artifacts(tmp_path, tmp_path / "out")
    first = artifacts.write(b"proof", media_type="text/plain")
    (tmp_path / first.path).write_bytes(b"bad")
    with pytest.raises(ValueError, match="collision"):
        artifacts.write(b"proof", media_type="text/plain")
    assert list((tmp_path / "out/objects").iterdir()) == [tmp_path / first.path]
