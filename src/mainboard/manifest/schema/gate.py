import json
from collections.abc import Mapping

from pydantic import model_validator

from ...core.base import Declared
from .spec import Json


class Gate(Declared):
    """One declared verification gate: a command whose exit status verdicts a `doctor` section.

    This is how a workspace teaches the report to ask a question no table of its own could
    answer. A gate names a command and nothing else about the tool behind it, so a proof
    workbench, a schema checker and a house linter all join the report the same way, by exiting
    zero when they are happy, and adding one is an edit to the manifest rather than to this
    package.

    run: the command line the section runs, staged through the workspace's own environment.
    report: dotted path to the failure list inside the command's JSON output, so a broken gate
        names what broke rather than only that something did. A gate that declares one and then
        prints none is read as a tool that never ran, which is a different finding from a
        workspace that is broken.
    install: the command that puts this gate's tool on the machine, offered as the repair when
        the gate answers with no report at all.
    timeout: how many seconds the gate may take before it counts as hung.
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
        """The failures this gate's declared report names, empty when `output` carries none.

        output: everything the gate's command printed.
        """
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
