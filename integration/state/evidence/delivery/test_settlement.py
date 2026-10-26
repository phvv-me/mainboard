from contextlib import AbstractContextManager, nullcontext
from unittest.mock import Mock

import pytest

from mainboard import MissionError
from mainboard.settling.monitor import Settlement, Unverifiable, owes_receipt

type _Outcome = AbstractContextManager[pytest.ExceptionInfo[Exception] | None]

_RECEIPT = '{"trial_receipt": {"artifacts": {}}}'
_COVERED = "complete law on cell 3/3 from run1\n3 skipped in 0.12s\n"


@pytest.fixture
def settlement() -> Settlement:
    """A settlement with no board: the verbs under test never reach one."""
    bare = Settlement.__new__(Settlement)
    bare.quiet, bare.fetched = {}, {}
    return bare


@pytest.mark.parametrize(
    ("verdict", "transcript", "outcome"),
    [
        pytest.param(
            "ok",
            "",
            pytest.raises(Unverifiable, match="no captured receipt"),
            id="a-success-without-a-receipt",
        ),
        pytest.param("ok", _COVERED, nullcontext(), id="every-cell-was-already-complete"),
        pytest.param("failed", "", nullcontext(), id="a-crash-owes-none"),
    ],
)
def test_a_native_run_claiming_results_without_a_receipt_is_unverified(
    settlement: Settlement, verdict: str, transcript: str, outcome: _Outcome
) -> None:
    record = Mock(script="pytest tests/test_kernels.py::test_launch")
    job = Mock(handle=Mock(fetch_path="research/lab/experiments/law"))
    owed = owes_receipt(record, transcript=transcript, verdict=verdict)
    with outcome:
        settlement.verify(job, job.handle.fetch_path, (), owed=owed)


@pytest.mark.parametrize(
    ("read", "saved", "outcome", "receipts"),
    [
        pytest.param(
            None,
            None,
            pytest.raises(MissionError, match="could not be read"),
            (),
            id="unread-and-never-saved",
        ),
        pytest.param("", None, nullcontext(), (), id="read-and-empty"),
        pytest.param(None, _RECEIPT, nullcontext(), (_RECEIPT,), id="unread-falls-back-to-saved"),
        pytest.param("", _RECEIPT, nullcontext(), (_RECEIPT,), id="empty-falls-back-to-saved"),
    ],
)
def test_a_log_that_failed_to_transfer_is_no_empty_log(
    settlement: Settlement,
    read: str | None,
    saved: str | None,
    outcome: _Outcome,
    receipts: tuple[str, ...],
) -> None:
    settlement.captured = Mock(transcript=Mock(return_value=saved))
    record = Mock(handle="7")
    record.name = ""
    with outcome:
        assert settlement.capture(record, Mock(transcript=Mock(return_value=read))) == receipts
    settlement.captured.keep_transcript.assert_not_called()


def test_a_pass_collects_each_folder_once(settlement: Settlement) -> None:
    jobs = [Mock(handle=Mock(host="miyabi-g", root="/work", fetch_path="evidence")) for _ in "abc"]
    assert [settlement.pull(job) for job in jobs] == ["evidence"] * 3
    assert sum(job.pull.call_count for job in jobs) == 1
