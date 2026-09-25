import json
from collections.abc import Mapping

from pydantic import model_validator

from ...core.base import Declared
from .spec import Json


class Gate(Declared):
    """One declared verification gate: a command whose exit status verdicts a `doctor` section.

    run: the command line, staged through the workspace's own environment.
    report: dotted path to the failure list in the command's JSON output; declaring one and
        printing none reads as a tool that never ran.
    install: the command putting the tool on the machine, the repair offered for no report.
    timeout: seconds before the gate counts as hung.
    """

    run: str
    report: str = ""
    install: str = ""
    timeout: float = 90.0

    @model_validator(mode="before")
    @classmethod
    def from_bare_string(cls, value: Json) -> Json:
        """Accept `lint = "ruff check ."` as shorthand for `{ run = "ruff check ." }`."""
        if isinstance(value, str):
            return {"run": value}
        return value

    def breakages(self, output: str) -> list[str]:
        """The failures the declared report names in `output`, empty when it carries none."""
        if not self.report or (start := output.find("{")) < 0:
            return []
        try:
            found: Json = json.loads(output[start:])
        except json.JSONDecodeError:
            return []
        for step in self.report.split("."):
            if not isinstance(found, Mapping):
                return []
            found = found.get(step)
        return [str(breakage) for breakage in found] if isinstance(found, list) else []
