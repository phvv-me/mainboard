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
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import mainboard
import pytest
from mainboard.board import Board
from mainboard.center.carrier import Carrier
from mainboard.context import ExecutionPlan
from mainboard.context.plan import environment_prefix
from mainboard.core import MissionError, Project
from mainboard.dispatch import HostUnreachable, SshTransport, landing, onboard
from mainboard.dispatch.agent import SshLink, program, runner
from mainboard.dispatch.landing import Landing, Renter
from mainboard.dispatch.onboard import Bootstrap, Onboarding, installers
from mainboard.dispatch.provenance import Source
from mainboard.dispatch.rentals import Rental
from mainboard.dispatch.shells import Dialect, HostShell
from mainboard.dispatch.shipment import Shipment
from mainboard.dispatch.targets import Facts
from mainboard.dispatch.transport import Endpoint, Machine
from mainboard.dispatch.vocabulary import Resources
from mainboard.dispatch.wrapping import wrap
from mainboard.engines.compile.backend import PIXI_VERSION
from mainboard.engines.compile.lockfile import Lockfile, Solved
from mainboard.manifest import HostProfile
from mainboard.probe.snapshot import HostFacts
from plumbum import local
from pydantic import ValidationError

_HOST = "fakehost"
_ROOT = "/srv/jobs"
_SHELLS = [
    "sh",
    "bash",
    pytest.param("zsh", marks=pytest.mark.skipif(not shutil.which("zsh"), reason="no zsh")),
]
_SYSTEM = re.compile(r"\bpython3\b|-m pip\b|\bpip install\b|--break-system-packages")
_PLAN = ExecutionPlan(
    host=_HOST, profile=HostProfile(kind="ssh", root=_ROOT, platform="linux-64"), env="default"
)
_LOCK = """version: 7
platforms:
- name: linux-64-system
  subdir: linux-64
- name: linux-aarch64
- name: osx-arm64
environments:
  default:
    packages:
      linux-64-system:
      - conda: https://example.invalid/linux-64/python-3.14.8-build_cp314.conda
      - conda: https://example.invalid/linux-64/python_abi-3.14-8_cp314.conda
      linux-aarch64:
      - conda: https://example.invalid/linux-aarch64/python-3.15.1-build_cp315.tar.bz2
      osx-arm64:
      - conda: https://example.invalid/osx-arm64/python-3.16.0rc1-build_cp316.conda
  training:
    packages:
      linux-64-system:
      - conda: https://example.invalid/linux-64/python-3.17.0-build_cp317.conda
"""


@pytest.fixture
def locked(workspace: Path) -> Path:
    lock = Lockfile(workspace)
    lock.path.write_text(
        lock.render({"default": Solved(solved_from="fake", solved_by="fake", lock=_LOCK)}),
        encoding="utf-8",
    )
    return workspace


@pytest.fixture
def launcher() -> str:
    """Resolve the test workspace's state directory after the fixture has entered it."""
    return Dialect().python(environment_prefix("~/.mb-jobs", "default"), host=_HOST, python="3.14")


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
def test_the_workspace_environment_runs_first(home: Path, shell: str, launcher: str) -> None:
    _stub(_environment(home), 'echo env "$@"')
    _stub(home / ".local/bin/uv", f'echo uv >> "{home}/asked"; exit 1')
    ran = _sshd(home, f"{launcher} -c 'print(1)'", shell=shell)
    assert (ran.returncode, ran.stdout) == (0, "env -c print(1)\n"), ran.stderr
    assert not (home / "system").exists()
    assert not (home / "asked").exists()


@pytest.mark.parametrize("shell", _SHELLS)
def test_uv_managed_cpython_runs_before_the_environment_exists(
    home: Path, shell: str, launcher: str
) -> None:
    managed = _stub(home / "uv-python/bin/python3.14", 'echo managed "$@"')
    _stub(home / ".local/bin/uv", f'echo "$@" > "{home}/asked"; echo {managed}')
    ran = _sshd(home, f"{launcher} -", shell=shell)
    assert (ran.returncode, ran.stdout) == (0, "managed -\n"), ran.stderr
    asked = "python find --managed-python 3.14\n"
    assert (home / "asked").read_text(encoding="utf-8") == asked
    assert not (home / "system").exists()


def test_a_host_with_neither_is_refused_by_name(home: Path, launcher: str) -> None:
    _stub(home / ".local/bin/uv", "exit 2")
    ran = _sshd(home, f"{launcher} -c pass", shell="sh")
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
def board(locked: Path) -> Board:
    """A workspace declaring the fake host."""
    manifest = (
        f'[workspace]\nname = "it"\n[hosts.{_HOST}]\nkind = "ssh"\nroot = "{_ROOT}"\n'
        'platform = "linux-64"\n'
    )
    (locked / "mb.toml").write_text(manifest, encoding="utf-8", newline="\n")
    return Board(locked)


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
    plan = _PLAN.model_copy(
        update={"profile": _PLAN.profile.model_copy(update={"root": "~/.mb-jobs"})}
    )
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
    launcher = Dialect().python(environment_prefix(_ROOT, "default"), host=_HOST, python="3.14")
    assert sent == [f'{launcher} -c "{runner.BOOTSTRAP}"', f"{launcher} -"]
    assert not any(_SYSTEM.search(command) for command in sent)


def test_collection_from_an_unrecorded_rental_uses_its_platform(
    board: Board, sent: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mainboard.dispatch.dispatcher.probe_capabilities",
        lambda host, **kwargs: Facts(name=host, platform="Linux aarch64"),
    )
    with pytest.raises(HostUnreachable):
        board.dispatcher.fetch_path("rental", root=_ROOT, path="results")
    assert len(sent) == 1
    assert '--managed-python "3.15"' in sent[0]


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


def test_onboarding_puts_uv_and_its_cpython_there_first_or_refuses(locked: Path) -> None:
    shell = _Recorded("command -v uv")
    Bootstrap(shell, workspace=locked).python()
    curl, installer = Dialect.uv_bootstrap
    wanted = "3.14"
    assert shell.commands == [
        f"mkdir -p {_ROOT}",
        "command -v uv",
        curl,
        installer,
        f"uv python install --no-bin {wanted}",
    ]
    with pytest.raises(MissionError, match="neither uv nor curl"):
        Bootstrap(_Recorded("command -v uv", "command -v curl"), workspace=locked).python()


@pytest.mark.parametrize("source", ["vendored", "indexed"])
def test_the_tool_installs_through_uv_alone(source: str) -> None:
    routes = installers(_Recorded(), python="3.14", vendored=source == "vendored", floor="0.5.4")
    commands = [routes.select(name).command for name in routes.names]
    installs = [command for command in commands if command != Dialect.noop]
    assert installs
    assert all(
        command.startswith("uv tool install") and "--managed-python" in command
        for command in installs
    )
    assert not any(_SYSTEM.search(command) for command in commands)


@pytest.mark.parametrize("existing", [False, True], ids=["fresh", "reuse"])
@pytest.mark.parametrize("vendored", [False, True], ids=["index", "editable"])
def test_bootstrap_uses_the_solved_minor(
    locked: Path, home: Path, existing: bool, vendored: bool
) -> None:
    """A fake uv would pick 3.15 from an open requirement, or reuse an installed 3.14."""
    shell = _Login(home)
    if vendored:
        (Path(shell.root) / "packages/mainboard").mkdir(parents=True)
    installed = home / "installed-python"
    if existing:
        installed.write_text("3.14\n", encoding="utf-8")
    _stub(
        home / ".local/bin/uv",
        """echo "$@" >> "$HOME/asked"
case "$1 $2" in
  "python install")
    wanted="$4"
    [ "$wanted" != '>=3.14' ] || wanted=3.15
    if [ ! -f "$HOME/installed-python" ]; then
      echo "$wanted" > "$HOME/installed-python"
      echo downloaded >> "$HOME/downloads"
    fi
    ;;
  "tool install")
    while [ "$1" != --python ]; do shift; done
    [ "$2" = 3.14 ] || { echo 'error: no cp315 wheel' >&2; exit 1; }
    ;;
  *) exit 2 ;;
esac""",
    )
    bootstrap = Bootstrap(shell, workspace=locked)
    bootstrap.python()
    assert bootstrap.tool().winner == ("uv" if vendored else "uv-index")
    commands = (home / "asked").read_text(encoding="utf-8").splitlines()
    assert commands[0] == "python install --no-bin 3.14"
    assert "--python 3.14" in commands[1]
    assert ("--editable" in commands[1]) == vendored
    assert installed.read_text(encoding="utf-8") == "3.14\n"
    assert (home / "downloads").exists() is not existing


class _Login(HostShell):
    """Run host commands under an isolated login home with only fake user tools."""

    def __init__(self, home: Path) -> None:
        super().__init__(local, _PLAN, str(home / ".mb-jobs"))
        self.home = home

    def execute(self, line: str) -> tuple[int, str, str]:
        ran = _sshd(self.home, line, shell="sh")
        return ran.returncode, ran.stdout, ran.stderr


@pytest.mark.parametrize("matching", [False, True], ids=["replace-wrong-minor", "reuse-tool"])
def test_an_existing_indexed_tool_must_use_the_solved_minor(
    locked: Path, home: Path, matching: bool
) -> None:
    shell = _Login(home)
    Path(shell.root).mkdir()
    _stub(home / ".local/bin/mainboard", "echo mainboard 0.5.4")
    _stub(
        home / ".local/bin/uv",
        """case "$1 $2" in
  "tool dir") echo "$HOME/tools" ;;
  "tool install") echo "$@" > "$HOME/reinstalled" ;;
  *) exit 1 ;;
esac""",
    )
    _stub(home / "tools/mainboard/bin/python", f"exit {0 if matching else 1}")
    result = Bootstrap(shell, workspace=locked, floor="0.5.4").tool()
    assert result.winner == ("present" if matching else "uv-index")
    assert (home / "reinstalled").exists() is not matching


def test_setup_and_sync_follow_a_changed_python_pin(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SetupShell(_Recorded):
        def execute(self, line: str) -> tuple[int, str, str]:
            code, out, err = super().execute(line)
            if line.endswith("pixi --version"):
                out = f"pixi {PIXI_VERSION}"
            if line.endswith(onboard.facts_command()):
                out = HostFacts().model_dump_json()
            return code, out, err

    shell = SetupShell()
    monkeypatch.setattr(onboard, "open_shell", lambda *args, **kwargs: shell)
    monkeypatch.setattr(
        onboard, "probe_capabilities", lambda host: Facts(name=host, platform="Linux x86_64")
    )
    monkeypatch.setattr(board.dispatcher, "mirror", lambda *args, **kwargs: [])
    onboarding = Onboarding(board.dispatcher, _PLAN, watch=lambda text: None)
    setup = onboarding.run()
    assert "uv python install --no-bin 3.14" in shell.commands
    assert any("uv tool install" in line and "--python 3.14" in line for line in shell.commands)
    shell.commands.clear()
    assert onboarding.run(sync_only=True).requires == setup.requires
    assert "uv python install --no-bin 3.14" in shell.commands
    assert not any("uv tool install" in line for line in shell.commands)
    lock = Lockfile(board.root)
    lock.path.write_text(
        lock.path.read_text(encoding="utf-8").replace("3.14.8", "3.15.0"), encoding="utf-8"
    )
    shell.commands.clear()
    assert onboarding.run(sync_only=True).requires != setup.requires
    assert "uv python install --no-bin 3.15" in shell.commands
    assert any("uv tool install" in line and "--python 3.15" in line for line in shell.commands)


def test_a_rental_uses_its_probed_platform_pin(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = _Recorded()

    def host_shell(remote: Machine, plan: ExecutionPlan, root: str) -> HostShell:
        shell.plan, shell.root = plan, root
        return shell

    def stop_at_environment(self: Bootstrap) -> None:
        raise MissionError("environment reached")

    monkeypatch.setattr(landing, "connection", lambda *args: nullcontext(local))
    monkeypatch.setattr(
        landing,
        "probe_capabilities",
        lambda *args, **kwargs: Facts(
            name="rental", home="/home/rental", platform="Linux aarch64"
        ),
    )
    monkeypatch.setattr(landing, "HostShell", host_shell)
    monkeypatch.setattr(Bootstrap, "environment", stop_at_environment)
    monkeypatch.setattr(board.dispatcher, "mirror", lambda *args, **kwargs: [])
    rental = Rental(handle="fake", endpoint=Endpoint(address="example.invalid"))
    plan = _PLAN.model_copy(update={"profile": HostProfile(kind="vast")})
    job = Landing(board.dispatcher, Mock(spec=Renter), plan, resources=Resources())
    shipment = Shipment.of_command("true", source=Source(identity="fake", key="fake"), imports=())
    with pytest.raises(MissionError, match="environment reached"):
        job.equip(rental, shipment=shipment)
    assert job.plan.profile.platform == "linux-aarch64"
    assert "uv python install --no-bin 3.15" in shell.commands
    assert any("uv tool install" in line and "--python 3.15" in line for line in shell.commands)


@pytest.mark.parametrize("step", ["python", "tool"])
def test_failed_uv_preserves_its_complete_diagnostic(locked: Path, home: Path, step: str) -> None:
    diagnostic = (
        "error: No solution found when resolving dependencies\n"
        "  cause: cuda-bindings has no wheel for cp315\n"
        "    " + "resolver context " * 30 + "\n"
        "  hint: use the interpreter solved by the workspace\n"
    )
    _stub(home / ".local/bin/uv", f"printf %s {shlex.quote(diagnostic)} >&2; exit 1")
    shell = _Login(home)
    Path(shell.root).mkdir()
    bootstrap = Bootstrap(shell, workspace=locked)
    with pytest.raises(MissionError) as refused:
        getattr(bootstrap, step)()
    assert diagnostic.strip() in str(refused.value)
    assert f"failed on '{_HOST}'" in str(refused.value)


def test_workspace_python_follows_the_default_platform_pin(locked: Path) -> None:
    lock = Lockfile(locked)
    generated = Project().out(locked) / "envs/default/pixi.lock"
    generated.parent.mkdir(parents=True)
    generated.write_text(_LOCK.replace("3.14.8", "3.18.0"), encoding="utf-8")
    assert lock.python_minor("linux-64") == "3.14"  # Committed beats a stale generated copy.
    assert lock.python_minor("linux-aarch64") == "3.15"
    assert lock.python_minor("osx-arm64") == "3.16"
    with pytest.raises(MissionError, match="lock default"):
        lock.python_minor("win-64")
    # The job environment does not choose the tool's interpreter.
    shell = _Recorded()
    shell.plan = _PLAN.model_copy(update={"env": "training"})
    assert Bootstrap(shell, workspace=locked).python_minor == "3.14"
    carrier = Carrier(_HOST, "uv", python=lock.python_minor("linux-aarch64"))
    assert "--python 3.15 " in carrier.argv[-1]
    lock.path.unlink()
    assert lock.python_minor("linux-64") == "3.18"  # Legacy generated-only workspaces.
    generated.unlink()
    with pytest.raises(MissionError, match="lock default"):
        Bootstrap(shell, workspace=locked)
    assert shell.commands == []  # Refuse before any host mutation.


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
