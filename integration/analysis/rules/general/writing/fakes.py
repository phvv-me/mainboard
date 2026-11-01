import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_SCRIPT = """\
#!{python}
import json
import sys

with open({log!r}, "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.stdin.read()) + "\\n")
sys.stdout.write(json.dumps({{"structured_output": {{"names": {names!r}}}}}))
sys.exit({status})
"""


class Claude:
    """The `claude` command the naming rules ask, as a script first on PATH.

    It answers `names` to every question (or exits with `status`, as a model that gives no
    answer does) and keeps the questions asked, so that no test reaches a real model.
    """

    @property
    def questions(self) -> list[str]:
        """What the rules asked so far, in order."""
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def __init__(
        self,
        folder: Path,
        *,
        monkeypatch: pytest.MonkeyPatch,
        names: Sequence[str],
        status: int = 0,
    ) -> None:
        script = folder / "bin" / "claude"
        script.parent.mkdir(parents=True)
        self.log = folder / "questions.jsonl"
        self.log.touch()
        script.write_text(
            _SCRIPT.format(
                python=sys.executable, log=str(self.log), names=list(names), status=status
            ),
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
