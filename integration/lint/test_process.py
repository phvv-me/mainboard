"""`Invocation` runs one lint command and ends it with everything it started at its deadline."""

import os
from pathlib import Path

from mainboard.enums import Exit
from mainboard.lint.process import Invocation


def test_a_command_past_its_deadline_is_killed_with_its_workers_and_says_so(
    tmp_path: Path,
) -> None:
    invocation = Invocation(
        step="slow",
        owner=".",
        cwd=tmp_path,
        argv=("sh", "-c", "echo begun; sleep 60 & wait"),
        env="default",
        timeout=1.0,
    )

    outcome = invocation.run({"PATH": os.environ["PATH"]})

    assert (outcome.code, outcome.output, outcome.seconds < 30) == (
        Exit.TIMED_OUT,
        "begun\n\nslow exceeded its 1s deadline",
        True,
    )
