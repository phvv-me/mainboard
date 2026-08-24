from typing import TYPE_CHECKING

from mainboard.engines.compile.backend import CommandResult

if TYPE_CHECKING:
    import pytest


def test_a_result_judges_its_own_exit_and_replays_what_it_retained(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert CommandResult(0, "", "").succeeded
    assert not CommandResult(1, "", "").succeeded

    CommandResult(0, "out text", "err text").replay()
    assert capsys.readouterr() == ("out text", "err text")
