import shlex
import sys
from json import dumps
from pathlib import Path

import pytest

from mainboard.context.plan import ExecutionPlan
from mainboard.dispatch.transport import SshTransport
from mainboard.manifest.schema.host import HostProfile, Sync

# A step that says what it was asked to and exits with the code after the word, as a Python the
# test already runs so every platform can execute it.
PYTHON = shlex.quote(sys.executable)


def say(word: str, code: int = 0) -> str:
    """A step command that prints `word` and exits with `code`."""
    script = f"import sys; print({word!r}); sys.exit({code})"
    return f"{PYTHON} -c {shlex.quote(script)}"


def declare(root: Path, gate: str) -> Path:
    """A package at `root` whose pyproject.toml carries `gate` as its CI table body."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "p"\n\n[tool.mainboard.ci]\n{gate}', encoding="utf-8"
    )
    return root


def step(name: str, run: str, *, only: str = "") -> str:
    """One `[[tool.mainboard.ci.steps]]` entry."""
    extra = f"only = {only}\n" if only else ""
    return f"[[tool.mainboard.ci.steps]]\nname = {dumps(name)}\nrun = {dumps(run)}\n{extra}\n"


def plan(host: str, platform: str) -> ExecutionPlan:
    """A bare plan for `host` on `platform`, its mirror at `/m` and its own excludes declared."""
    sync = Sync(include=["everything"], exclude=["data/raw"], protect=["results/***"])
    profile = HostProfile(kind="ssh", platform=platform, root="/m", sync=sync)
    return ExecutionPlan(host=host, profile=profile, env="default")


class Ssh:
    """An ssh policy whose processes are answered from a script instead of started.

    answers: the `(code, stdout, stderr)` each call returns in turn, or the error it raises.
    calls: every argv handed over, with the deadline it was held to.
    """

    def __init__(self) -> None:
        self.answers: list[tuple[int, str, str] | Exception] = []
        self.calls: list[tuple[tuple[str, ...], float | None]] = []
        self.options = SshTransport().options

    def destination(self, host: str) -> str:
        return host

    def invoke(
        self, command: tuple[str, ...], host: str, *, operation: str, timeout: float | None = None
    ) -> tuple[int, str, str]:
        self.calls.append((command, timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def ssh() -> Ssh:
    """A scripted ssh policy with no answers queued yet."""
    return Ssh()
