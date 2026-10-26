import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

import mainboard
from mainboard import Board
from mainboard.context.plan import environment_prefix
from mainboard.dispatch import Dialect, Facts, HostUnreachable, SshTransport
from mainboard.dispatch.agent import runner
from mainboard.dispatch.wrapping import wrap
from mainboard.manifest import HostProfile

from .builders import environment, login_environment, sshd, stub, survey
from .constants import HOMED, HOST, PLAN, ROOT, SHELLS, SYSTEM


def _furnish(home: Path, *, world: str) -> None:
    """Fill `home` as a host in `world` looks, its uv writing what it was asked to `asked`.

    world: `environment` has the workspace environment installed, `managed` a uv that finds a
        managed CPython, `bare` a uv that finds none.
    """
    uv, asked = home / ".local/bin/uv", f'echo "$@" >> "{home}/asked"'
    match world:
        case "environment":
            stub(environment(home), 'echo env "$@"')
            stub(uv, f"{asked}; exit 1")
        case "managed":
            managed = stub(home / "uv-python/bin/python3.14", 'echo managed "$@"')
            stub(uv, f"{asked}; echo {managed}")
        case "bare":
            stub(uv, f"{asked}; exit 2")


def _asked(home: Path) -> str | None:
    """What the host's uv was asked, None when nothing ran it."""
    path = home / "asked"
    return path.read_text(encoding="utf-8") if path.exists() else None


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("world", "args", "code", "out", "said", "asked"),
    [
        pytest.param(
            "environment",
            "-c 'print(1)'",
            0,
            "env -c print(1)\n",
            "",
            None,
            id="the-workspace-environment-runs-first",
        ),
        pytest.param(
            "managed",
            "-",
            0,
            "managed -\n",
            "",
            "python find --managed-python 3.14\n",
            id="uv-managed-cpython-runs-before-the-environment-exists",
        ),
        pytest.param(
            "bare",
            "-c pass",
            127,
            "",
            rf"mainboard: {HOST} has neither.*host setup {HOST}",
            "python find --managed-python 3.14\n",
            id="a-host-with-neither-is-refused-by-name",
        ),
    ],
)
def test_a_remote_python_starts_in_the_environment_else_in_uvs_cpython_else_the_host_is_refused(
    home: Path,
    launcher: str,
    shell: str,
    world: str,
    args: str,
    code: int,
    out: str,
    said: str,
    asked: str | None,
) -> None:
    _furnish(home, world=world)

    ran = sshd(home, f"{launcher} {args}", program=shell)

    assert (ran.returncode, ran.stdout) == (code, out), ran.stderr
    assert re.search(said, ran.stderr, re.DOTALL) and _asked(home) == asked
    assert not (home / "system").exists()


@pytest.mark.parametrize(
    ("world", "code", "out", "said"),
    [
        pytest.param(
            "bare",
            1,
            "",
            rf"{HOST} has no default workspace environment.*run mb host setup {HOST}\Z",
            id="a-bare-path-is-refused-by-name",
        ),
        pytest.param(
            "environment", 0, "env -c pass\n", "", id="the-environment-runs-once-it-exists"
        ),
    ],
)
def test_activation_refuses_where_a_bare_path_would_find_a_system_python(
    home: Path, world: str, code: int, out: str, said: str
) -> None:
    root = home / ".mb-jobs"
    root.mkdir()
    _furnish(home, world=world)

    ran = sshd(home, wrap(PLAN, str(root), command="python -c pass"), program="bash")

    assert (ran.returncode, ran.stdout) == (code, out), ran.stderr
    assert re.search(said, ran.stderr.rstrip(), re.DOTALL) and _asked(home) is None
    assert not (home / "system").exists()


@pytest.mark.parametrize("shell", SHELLS)
def test_the_agent_answers_through_the_launcher(
    board: Board, home: Path, shell: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher's own agent, its ssh swapped for the login shell sshd would start."""
    environment(home).parent.mkdir(parents=True)
    environment(home).symlink_to(sys.executable)
    monkeypatch.setattr(SshTransport, "command", lambda self, host: (shell, "-c"))
    for key, value in login_environment(home).items():
        monkeypatch.setenv(key, value)

    answer = board.dispatcher.agent(HOMED).ask(survey(str(home / "mirror")))

    assert answer in ([{"fold": True}], [{"fold": False}])
    assert not (home / "system").exists()


def _reach(board: Board, *, action: str, host: str) -> None:
    """Make dispatch start a remote Python on `host`: ask its agent, or fetch a path from it."""
    if action == "agent":
        board.dispatcher.agent(PLAN).ask(survey(ROOT))
    else:
        board.dispatcher.fetch_path(host, root=ROOT, path="results")


@pytest.mark.parametrize(
    ("action", "host", "python", "arguments"),
    [
        pytest.param(
            "agent",
            HOST,
            ">=3.14.0,<3.15",
            f' -c "{runner.Agent.BOOTSTRAP}"',
            id="the-agent-bootstrap",
        ),
        pytest.param("fetch", HOST, ">=3.14.0,<3.15", " -", id="a-fetch-from-a-declared-host"),
        pytest.param(
            "fetch", "rental", ">=3.15.0,<3.16", " -", id="a-fetch-from-an-unrecorded-rental"
        ),
    ],
)
def test_dispatch_starts_every_remote_python_through_the_launcher(
    board: Board,
    sent: list[str],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    host: str,
    python: str,
    arguments: str,
) -> None:
    """A rental's platform is the one it was probed as, a declared host's the one it declares."""
    probed = Facts(name="rental", platform="Linux aarch64")
    monkeypatch.setattr(
        "mainboard.dispatch.dispatcher.core.probe_capabilities", lambda *a, **k: probed
    )

    with pytest.raises(HostUnreachable):
        _reach(board, action=action, host=host)

    launcher = Dialect().python(environment_prefix(ROOT, env="default"), host=host, python=python)
    assert sent == [launcher + arguments]
    assert not any(SYSTEM.search(command) for command in sent)


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
        if SYSTEM.search(text)
    ]

    assert not found, found


def test_a_host_names_no_interpreter_of_its_own() -> None:
    with pytest.raises(ValidationError):
        HostProfile.model_validate({"python": "python3"})
