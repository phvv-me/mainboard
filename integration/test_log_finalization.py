"""A verdict cannot silently omit evidence attached during later cleanup."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from mainboard.trials.artifacts import Artifacts
from mainboard.trials.log import Log
from mainboard.trials.provenance import Admissibility
from mainboard.trials.session import Trial
from mainboard.trials.vocabulary import Vocabulary


@pytest.mark.parametrize("order", ["before", "after", "resettled", "mutated"])
def test_last_receipt_must_include_cleanup_artifacts(tmp_path: Path, order: str) -> None:
    item = Mock(
        nodeid="test_fixture.py::test_fixture",
        path=tmp_path / "test_fixture.py",
        callspec=None,
        user_properties=[],
    )
    session = Mock()
    session.declared.flags = ()
    session.declared.universe.axes = ()
    session.declared.words = Vocabulary.of("known")
    session.baseline = {}
    session.cell.return_value.filters = {}
    session.taken.admits.return_value = Admissibility.ADMISSIBLE
    log = Log.__new__(Log)
    log.trial = Trial(item, session)
    log.identity, log.metadata, log.bound = "fixture", {}, {}
    log.spool = Mock()
    log.artifacts = Artifacts(tmp_path, tmp_path / "artifacts")
    log.artifact(b"output", name="output")
    if order == "before":
        log.artifact(b"cleanup", name="cleanup")
    log.known("fixture")
    if order in ("after", "resettled"):
        log.artifact(b"cleanup", name="cleanup")
    elif order == "mutated":
        artifact = log.trial.artifacts["output"]
        assert isinstance(artifact, dict)
        artifact["sha256"] = "changed"
    if order == "resettled":
        log.known("fixture with cleanup")
    if order in ("after", "mutated"):
        with pytest.raises(RuntimeError, match="Artifacts changed after the last receipt"):
            log.close(passed=True)
        recorded = session.writer.return_value.write.call_args.args[0]
        assert recorded["outcome"] == "failed" and recorded["verdict"] == ""
        assert not log.trial.settled
        assert session.writer.return_value.write.call_count == 2
    else:
        log.close(passed=True)
        assert (
            session.writer.return_value.write.call_args.args[0]["artifacts"] == log.trial.artifacts
        )
    log.spool.close.assert_called_once()
