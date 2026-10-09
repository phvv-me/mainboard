"""Mainboard runs nothing on a host's own Python, and nothing here lets that come back.

The owner's rule (2026-10-09): only mainboard's own Pythons run anything on a host, the
workspace environment's, or before that exists the uv-managed CPython onboarding puts there
first; never a system `python3`, never a pip into a user site. These tests run the launcher under
the login shells a host may have, build every command dispatch sends to a fake host, and read the
source for a system interpreter that could creep back.
"""

import ast
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from plumbum import local
from pydantic import ValidationError

import mainboard
from mainboard.board import Board
from mainboard.context import ExecutionPlan
from mainboard.context.plan import environment_prefix
from mainboard.core import MissionError
from mainboard.dispatch import HostUnreachable, SshTransport
from mainboard.dispatch.agent import SshLink, program, runner
from mainboard.dispatch.onboard import Bootstrap, installers
from mainboard.dispatch.shells import Dialect, HostShell
from mainboard.dispatch.wrapping import wrap
from mainboard.manifest import HostProfile

_HOST = "fakehost"
_ROOT = "/srv/jobs"
_SHELLS = [
    "sh",
    "bash",
    pytest.param("zsh", marks=pytest.mark.skipif(not shutil.which("zsh"), reason="no zsh")),
]
_SYSTEM = re.compile(r"\bpython3\b|-m pip\b|\bpip install\b|--break-system-packages")
_PLAN = ExecutionPlan(host=_HOST, profile=HostProfile(kind="ssh", root=_ROOT), env="default")
# The launcher of a host whose workspace sits at `~/.mb-jobs`.
_LAUNCHER = Dialect().python(environment_prefix("~/.mb-jobs", "default"), host=_HOST)


def _survey(root: str) -> program.Request:
    """The first thing a mirror asks: describe `root`, here with nothing in scope."""
    return {"survey": {"root": root, "state": ".state", "scopes": [], "named": []}}


def _stub(path: Path, body: str) -> Path:
    """An executable `sh` script at `path` running `body`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A login home whose PATH answers `python3` and `python` with stubs that leave a trace."""
    for name in ("python3", "python"):
        _stub(tmp_path / "system-bin" / name, f'echo "$0" >> "{tmp_path}/system"; exit 3')
    return tmp_path


def _login(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "PATH": f"{home}/system-bin:/usr/bin:/bin"}


def _environment(home: Path) -> Path:
    """Where the workspace environment's interpreter sits under `home`."""
    return Path(environment_prefix(str(home / ".mb-jobs"), "default")) / "bin" / "python"


def _sshd(home: Path, line: str, *, shell: str) -> subprocess.CompletedProcess[str]:
    """Run `line` the way sshd does, through the login `shell -c`, in `home`."""
    return subprocess.run(
        [shell, "-c", line],
        env=_login(home),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("shell", _SHELLS)
def test_the_workspace_environment_runs_first(home: Path, shell: str) -> None:
    _stub(_environment(home), 'echo env "$@"')
    _stub(home / ".local/bin/uv", f'echo uv >> "{home}/asked"; exit 1')
    ran = _sshd(home, f"{_LAUNCHER} -c 'print(1)'", shell=shell)
    assert (ran.returncode, ran.stdout) == (0, "env -c print(1)\n"), ran.stderr
    assert not (home / "system").exists()
    assert not (home / "asked").exists()


@pytest.mark.parametrize("shell", _SHELLS)
def test_uv_managed_cpython_runs_before_the_environment_exists(home: Path, shell: str) -> None:
    managed = _stub(home / "uv-python/bin/python3.14", 'echo managed "$@"')
    _stub(home / ".local/bin/uv", f'echo "$@" > "{home}/asked"; echo {managed}')
    ran = _sshd(home, f"{_LAUNCHER} -", shell=shell)
    assert (ran.returncode, ran.stdout) == (0, "managed -\n"), ran.stderr
    asked = f"python find --managed-python {Dialect.requires_python}\n"
    assert (home / "asked").read_text(encoding="utf-8") == asked
    assert not (home / "system").exists()


def test_a_host_with_neither_is_refused_by_name(home: Path) -> None:
    _stub(home / ".local/bin/uv", "exit 2")
    ran = _sshd(home, f"{_LAUNCHER} -c pass", shell="sh")
    assert ran.returncode == 127
    assert f"mainboard: {_HOST} has neither" in ran.stderr
    assert f"host setup {_HOST}" in ran.stderr
    assert not (home / "system").exists()


def test_activation_refuses_where_a_bare_path_would_find_a_system_python(home: Path) -> None:
    root = home / ".mb-jobs"
    root.mkdir()
    line = wrap(_PLAN, str(root), command="python -c pass")
    refused = _sshd(home, line, shell="bash")
    assert refused.returncode == 1
    assert f"{_HOST} has no default workspace environment" in refused.stderr
    assert refused.stderr.rstrip().endswith(f"run mb host setup {_HOST}")
    _stub(_environment(home), 'echo env "$@"')
    assert _sshd(home, line, shell="bash").stdout == "env -c pass\n"
    assert not (home / "system").exists()


@pytest.fixture
def board(workspace: Path) -> Board:
    """A workspace declaring the fake host."""
    manifest = f'[workspace]\nname = "it"\n[hosts.{_HOST}]\nkind = "ssh"\nroot = "{_ROOT}"\n'
    (workspace / "mb.toml").write_text(manifest, encoding="utf-8", newline="\n")
    return Board(workspace)


@pytest.mark.parametrize("shell", _SHELLS)
def test_the_agent_answers_through_the_launcher(
    board: Board, home: Path, shell: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher's own agent, its ssh swapped for the login shell sshd would start."""
    _environment(home).parent.mkdir(parents=True)
    _environment(home).symlink_to(sys.executable)
    monkeypatch.setattr(SshTransport, "command", lambda self, host: (shell, "-c"))
    for key, value in _login(home).items():
        monkeypatch.setenv(key, value)
    plan = _PLAN.model_copy(update={"profile": HostProfile(kind="ssh", root="~/.mb-jobs")})
    answer = board.dispatcher.agent(plan).ask(_survey(str(home / "mirror")))
    assert answer in ([{"fold": True}], [{"fold": False}])
    assert not (home / "system").exists()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every remote command dispatch hands ssh, each refused before it leaves."""
    lines: list[str] = []

    def spawn(self: SshLink, command: str) -> runner.Process:
        lines.append(command)
        raise HostUnreachable("recorded")

    def run(self: SshTransport, argv: Sequence[str], host: str, **_: str | Path) -> None:
        lines.append(argv[-1])
        raise HostUnreachable("recorded")

    monkeypatch.setattr(SshLink, "spawn", spawn)
    monkeypatch.setattr(SshTransport, "run", run)
    return lines


def test_dispatch_starts_every_remote_python_through_the_launcher(
    board: Board, sent: list[str]
) -> None:
    with pytest.raises(HostUnreachable):
        board.dispatcher.agent(_PLAN).ask(_survey(_ROOT))
    with pytest.raises(HostUnreachable):
        board.dispatcher.fetch_path(_HOST, root=_ROOT, path="results")
    launcher = Dialect().python(environment_prefix(_ROOT, "default"), host=_HOST)
    assert sent == [f'{launcher} -c "{runner.BOOTSTRAP}"', f"{launcher} -"]
    assert not any(_SYSTEM.search(command) for command in sent)


class _Recorded(HostShell):
    """A host shell that reaches no host: it keeps each command and answers from `absent`."""

    def __init__(self, *absent: str) -> None:
        super().__init__(local, _PLAN, _ROOT)
        self.absent = absent
        self.commands: list[str] = []

    def execute(self, line: str) -> tuple[int, str, str]:
        command = line.removeprefix(self.stage("", activate=False))
        self.commands.append(command)
        return int(command in self.absent), "", ""


def test_onboarding_puts_uv_and_its_cpython_there_first_or_refuses() -> None:
    shell = _Recorded("command -v uv")
    Bootstrap(shell).python()
    curl, installer = Dialect.uv_bootstrap
    wanted = shlex.quote(Dialect.requires_python)
    assert shell.commands == [
        f"mkdir -p {_ROOT}",
        "command -v uv",
        curl,
        installer,
        f"uv python install --no-bin {wanted}",
    ]
    with pytest.raises(MissionError, match="neither uv nor curl"):
        Bootstrap(_Recorded("command -v uv", "command -v curl")).python()


@pytest.mark.parametrize("source", ["vendored", "indexed"])
def test_the_tool_installs_through_uv_alone(source: str) -> None:
    routes = installers(_Recorded(), vendored=source == "vendored", floor="0.5.4")
    commands = [routes.select(name).command for name in routes.names]
    installs = [command for command in commands if command != Dialect.noop]
    assert installs
    assert all(
        command.startswith("uv tool install") and "--managed-python" in command
        for command in installs
    )
    assert not any(_SYSTEM.search(command) for command in commands)


def test_a_host_names_no_interpreter_of_its_own() -> None:
    with pytest.raises(ValidationError):
        HostProfile.model_validate({"python": "python3"})


def _literals(path: Path) -> Iterator[tuple[int, str]]:
    """Every string constant in `path` with its line, docstrings left out."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    documented = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
    }
    yield from (
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in documented
    )


@pytest.mark.parametrize("package", ["dispatch", "manifest", "center", "engines"])
def test_no_remote_path_names_a_system_python(package: str) -> None:
    sources = Path(mainboard.__file__).parent
    found = [
        f"{path.relative_to(sources)}:{line}: {text!r}"
        for path in sorted((sources / package).rglob("*.py"))
        for line, text in _literals(path)
        if _SYSTEM.search(text)
    ]
    assert not found, found
