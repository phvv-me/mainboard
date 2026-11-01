"""A verdict cannot silently omit evidence attached during later cleanup."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from mainboard.trials import Admissibility, Log, Trial, Vocabulary
from mainboard.trials.artifacts import Artifacts
from mainboard.trials.running.stream import TrialStream


def _log(tmp_path: Path) -> tuple[Log, Mock, Mock]:
    """A log over one fixture whose output artifact is attached, with its session and spool."""
    item = Mock(
        nodeid="test_fixture.py::test_fixture",
        path=tmp_path / "test_fixture.py",
        callspec=None,
        user_properties=[],
    )
    session = Mock(
        **{
            "declared.flags": (),
            "declared.universe.axes": (),
            "declared.words": Vocabulary.of("known"),
            "baseline": {},
            "cell.return_value.filters": {},
            "taken.admits.return_value": Admissibility.ADMISSIBLE,
        }
    )
    log, stream, spool = Log.__new__(Log), TrialStream.__new__(TrialStream), Mock()
    log.trial = stream.trial = Trial(item, session)
    log.metadata, log.stream = {}, stream
    stream.identity, stream.bound, stream.spool = "fixture", {}, spool
    log.artifacts = Artifacts(tmp_path, tmp_path / "artifacts")
    log.artifact(b"output", name="output")
    return log, session, spool


def _accepted(log: Log, *, session: Mock, spool: Mock) -> None:
    log.stream.close(passed=True)
    assert session.writer.return_value.write.call_args.args[0]["artifacts"] == log.trial.artifacts
    spool.close.assert_called_once()


def _refused(log: Log, *, session: Mock, spool: Mock) -> None:
    with pytest.raises(RuntimeError, match="Artifacts changed after the last receipt"):
        log.stream.close(passed=True)
    recorded = session.writer.return_value.write.call_args.args[0]
    assert recorded["outcome"] == "failed" and recorded["verdict"] == ""
    assert not log.trial.settled
    assert session.writer.return_value.write.call_count == 2
    spool.close.assert_called_once()


def test_a_cleanup_artifact_attached_before_the_receipt_is_kept(tmp_path: Path) -> None:
    log, session, spool = _log(tmp_path)
    log.artifact(b"cleanup", name="cleanup")
    log.known("fixture")
    _accepted(log, session=session, spool=spool)


def test_a_cleanup_artifact_attached_after_the_receipt_refuses_the_close(tmp_path: Path) -> None:
    log, session, spool = _log(tmp_path)
    log.known("fixture")
    log.artifact(b"cleanup", name="cleanup")
    _refused(log, session=session, spool=spool)


def test_a_resettled_receipt_keeps_the_cleanup_artifacts(tmp_path: Path) -> None:
    log, session, spool = _log(tmp_path)
    log.known("fixture")
    log.artifact(b"cleanup", name="cleanup")
    log.known("fixture with cleanup")
    _accepted(log, session=session, spool=spool)


def test_a_mutated_artifact_after_the_receipt_refuses_the_close(tmp_path: Path) -> None:
    log, session, spool = _log(tmp_path)
    log.known("fixture")
    artifact = log.trial.artifacts["output"]
    assert isinstance(artifact, dict)
    artifact["sha256"] = "changed"
    _refused(log, session=session, spool=spool)
