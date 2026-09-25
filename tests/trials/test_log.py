"""The injected facade preserves existing verdicts and durable, independently readable data."""

import hashlib
import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from mainboard import span
from mainboard.profile import Feature, Profile
from mainboard.profile.profiler import Collection
from mainboard.trials import Artifact, Declaration, Log, Session, digested
from mainboard.trials.artifacts import Artifacts

from .support import Item, declaration


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


def test_fetched_artifacts_skip_plain_entries_and_refuse_a_link_out_of_the_fetch(
    tmp_path: Path,
) -> None:
    """A path that names bytes inside the fetch is not enough when a link carries it elsewhere."""
    directory = tmp_path / "datasets/node"
    directory.mkdir(parents=True)
    (tmp_path / "secret").write_bytes(b"data")
    (directory / "escape").symlink_to(tmp_path / "secret")
    reference = Artifact(
        path="datasets/node/escape", sha256=hashlib.sha256(b"data").hexdigest(), size=4
    )
    artifacts = {"events": "datasets/node/events", "value": reference.model_dump()}
    line = json.dumps({"trial_receipt": {"artifacts": artifacts}})
    with pytest.raises(ValueError, match="link leaves the declared fetch"):
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


def test_log_reads_its_gate_through_the_trial_and_refuses_words_nobody_declared(
    session: Session, tmp_path: Path
) -> None:
    """The facade is not a second outcome system, so it settles only the workspace's own words."""
    log = Log(session.trial(Item("alpha/t.py::gate", tmp_path / "alpha/t.py")))
    registration = {"law_low": 0.9}
    assert log.gate(registration) is registration
    assert log.trial.gated == digested(registration)
    with pytest.raises(AttributeError, match="proved"):
        log.proved("a word this workspace never declared")
    with pytest.raises(ValueError, match="undeclared verdict"):
        log.settle("proved")
    assert not log.trial.settled
    log.close(passed=False)


def test_a_research_log_carries_its_claim_run_manifest_and_names_its_project(
    research: Path, tmp_path: Path
) -> None:
    session = Session(declaration(research, repo=tmp_path))
    log = Log(session.trial(Item("alpha/test_law.py::holds", research / "alpha/test_law.py")))
    assert log.trial.artifacts["run"] == session.manifests["alpha"].model_dump(mode="json")
    assert log.context["project"] == tmp_path.name
    log.close(passed=True)


@pytest.mark.parametrize(
    ("suffix", "media_type"),
    [(".PNG", "image/png"), (".jpeg", "image/jpeg"), (".svg", "image/svg+xml")],
)
def test_an_image_is_typed_by_its_suffix_and_named_when_left_unnamed(
    session: Session, tmp_path: Path, suffix: str, media_type: str
) -> None:
    rendered = tmp_path / f"figure{suffix}"
    rendered.write_bytes(b"rendered")
    log = Log(session.trial(Item("alpha/t.py::image", tmp_path / "alpha/t.py")))
    reference = log.image(rendered)
    assert reference.media_type == media_type
    assert reference.read(tmp_path) == b"rendered"
    assert log.trial.artifacts["image-1"] == reference.model_dump()
    log.close(passed=True)


def test_tables_gather_each_verified_parquet_artifact_beside_the_receipt_it_came_from(
    session: Session, tmp_path: Path
) -> None:
    """Every producer's rows come back together, each carrying the provenance of its receipt."""
    store = session.declared.universe.dataset("alpha")
    assert store.tables(tmp_path, schema_name="test.ratio.v1").is_empty()
    store.writer("run-0", {"node": "alpha"}).write({"lane": "old", "outcome": "passed"})
    log = Log(session.trial(Item("alpha/t.py::tables", tmp_path / "alpha/t.py")))
    log.table([{"ratio": 1.5}], name="ratios", schema_name="test.ratio.v1")
    log.table([{"other": 1}], schema_name="test.other.v1")
    log.validated("the law held", ratio=1.5)
    log.close(passed=True)

    frame = store.tables(tmp_path, schema_name="test.ratio.v1")
    assert frame["ratio"].to_list() == [1.5]
    provenance = json.loads(frame["_trial"][0])
    assert provenance["artifact_name"] == "ratios" and provenance["run"] == session.run
    assert "measured" not in provenance and "artifacts" not in provenance
    assert store.tables(tmp_path, schema_name="test.ratio.v1", run="run-0").is_empty()


@pytest.mark.parametrize(
    ("attach", "refusal"),
    [
        (lambda log: log.artifact(b"rows", schema_name="test.rows.v1"), "non-Parquet"),
        (
            lambda log: log.table(pl.DataFrame({"_trial": [0]}), schema_name="test.rows.v1"),
            "reserved _trial",
        ),
    ],
    ids=["bytes", "forged-provenance"],
)
def test_tables_refuse_an_artifact_that_is_not_a_plain_parquet_table(
    session: Session, tmp_path: Path, attach: Callable[[Log], Artifact], refusal: str
) -> None:
    log = Log(session.trial(Item("alpha/t.py::tables", tmp_path / "alpha/t.py")))
    attach(log)
    log.validated("the law held")
    log.close(passed=True)
    store = session.declared.universe.dataset("alpha")
    with pytest.raises(ValueError, match=refusal):
        store.tables(tmp_path, schema_name="test.rows.v1")


def test_declared_inputs_are_hash_checked_and_recorded(
    declared: Declaration, probed: None, tmp_path: Path
) -> None:
    content = b"frozen predecessor"
    (tmp_path / "input.bin").write_bytes(content)
    pinned = Artifact(
        path="input.bin", sha256=hashlib.sha256(content).hexdigest(), size=len(content)
    )
    table = BytesIO()
    pl.DataFrame({"ratio": [1.5]}).write_parquet(table)
    (tmp_path / "ratios.parquet").write_bytes(table.getvalue())
    ratios = Artifact(
        path="ratios.parquet",
        sha256=hashlib.sha256(table.getvalue()).hexdigest(),
        size=len(table.getvalue()),
        media_type="application/vnd.apache.parquet",
    )
    inputs = {"predecessor": pinned, "ratios": ratios}
    session = Session(declared.model_copy(update={"inputs": inputs}))
    log = Log(session.trial(Item("alpha/t.py::read", tmp_path / "alpha/t.py")))
    assert log.read("predecessor") == content
    with pytest.raises(KeyError):
        log.read("latest")
    assert log.spool.frames_from(0)[-1].payload["data"]["sha256"] == pinned.sha256
    assert log.read_table("ratios").to_dicts() == [{"ratio": 1.5}]
    log.close(passed=True)


@pytest.mark.parametrize(("name", "schema_name"), [("snapshot", "test.Profile.v1"), ("", "")])
def test_model_artifacts_preserve_exact_bytes_reference_and_defaults(
    session: Session, tmp_path: Path, name: str, schema_name: str, request: pytest.FixtureRequest
) -> None:
    module = pytest.Module.from_parent(request.session, path=tmp_path / "alpha/t.py")
    item = pytest.Function.from_parent(module, name="model", callobj=lambda: None)
    log = Log(session.trial(item))
    value = Profile(host="host-α", device="GPU")
    expected = value.model_dump_json().encode()
    reference = log.model(value, name=name, schema_name=schema_name)
    assert isinstance(reference, Artifact)
    assert reference.read(tmp_path) == expected
    assert reference.sha256 == hashlib.sha256(expected).hexdigest()
    assert reference.size == len(expected)
    assert reference.media_type == "application/json"
    assert reference.schema_name == schema_name
    assert log.trial.artifacts[name or "artifact-1"] == reference.model_dump()
    assert Profile.model_validate_json(reference.read(tmp_path)) == value
    log.close(passed=True)


def test_model_serialization_failure_is_not_published_or_swallowed(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    module = pytest.Module.from_parent(request.session, path=tmp_path / "alpha/t.py")
    item = pytest.Function.from_parent(module, name="model", callobj=lambda: None)
    log = Log(session.trial(item))
    before = dict(log.trial.artifacts)
    frame_count = len(log.spool.frames_from(0))
    failure = ValueError("model serializer failed")

    def fail(value: Profile) -> str:
        raise failure

    monkeypatch.setattr(Profile, "model_dump_json", fail)
    with pytest.raises(ValueError, match="model serializer failed") as raised:
        log.model(Profile())
    assert raised.value is failure
    assert log.trial.artifacts == before
    assert len(log.spool.frames_from(0)) == frame_count
    log.close(passed=False)


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


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "../outside",
        "/outside",
        "C:/outside",
        "C:outside",
        "//host/share/file",
        "data\\file",
        "data/../file",
        "data/./file",
        "data//file",
        "data/file:stream",
        "data/\0file",
    ],
)
def test_artifact_paths_are_portable_before_reading_or_transfer_verification(
    tmp_path: Path, path: str
) -> None:
    reference = Artifact(path=path, sha256=hashlib.sha256(b"").hexdigest(), size=0)
    with pytest.raises(ValueError, match="canonical and project-relative"):
        reference.read(tmp_path)
    receipt = json.dumps({"trial_receipt": {"artifacts": {"table": reference.model_dump()}}})
    with pytest.raises(ValueError, match="canonical and project-relative"):
        Artifacts.verify([receipt], directory=tmp_path, boundary=tmp_path)


def test_project_artifact_reads_preserve_logical_dataset_mounts(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "datasets").symlink_to(storage, target_is_directory=True)
    reference = Artifacts(project, project / "datasets/node").write(
        b"mounted bytes", media_type="text/plain"
    )
    assert reference.read(project) == b"mounted bytes"
    receipt = json.dumps({"trial_receipt": {"artifacts": {"data": reference.model_dump()}}})
    Artifacts.verify([receipt], directory=project / "datasets", boundary=project)
