from typing import TYPE_CHECKING

from mainboard.engines.compile.backend import CommandResult

if TYPE_CHECKING:
    import pytest


def test_succeeded_is_true_only_for_a_zero_returncode() -> None:
    assert CommandResult(0, "", "").succeeded
    assert not CommandResult(1, "", "").succeeded


def test_replay_writes_and_flushes_both_retained_streams(
    capsys: pytest.CaptureFixture[str],
) -> None:
    CommandResult(0, "out text", "err text").replay()
    captured = capsys.readouterr()
    assert captured.out == "out text"
    assert captured.err == "err text"
