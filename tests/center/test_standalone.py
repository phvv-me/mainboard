import json
import os
import subprocess
from pathlib import Path

import pytest
from plumbum import local
from plumbum.commands.base import BaseCommand
from plumbum.commands.processes import ProcessTimedOut

from mainboard import MissionError, Project
from mainboard.center.standalone import Standalone
from mainboard.cli import build
from mainboard.engines.compile.backend.result import CommandResult
from mainboard.git.process import Git
from mainboard.manifest.loading import composition

_MANIFEST = Project().manifest

# The root's `PYTHONPATH`: a directory holding `common`, a member's own `src/`, and a directory
# outside the workspace, which no clone could be missing.
_PATHS = ["{{ config_root }}/research", "{{ config_root }}/research/head/src", "/opt/elsewhere"]

_ROOT = f"""
[workspace]
name = "life"
members = ["packages/*", "research/*"]

[env]
PYTHONPATH = "{os.pathsep.join(_PATHS)}"

[tasks]
head-figures = {{ run = "python plot.py", dir = "research/head/papers" }}
shared = "echo shared"

[envs.gpu.tasks]
head-kernel = "python research/head/kernel.py"

[papers.head]
dir = "research/head/papers/latex"
"""

_HEAD = """
[workspace]
name = "head"

[hosts.gold]
kind = "ssh"

[python.deps]
lib = { path = "../../packages/lib" }

[tasks]
gate = { run = "pytest", depends = ["shared", "figures"] }
figures = "python plot.py"
cache = { run = "echo", env = { DIR = "{{ config_root }}/../cache" } }

[envs.cuda.tasks]
kernel = { run = "python k.py", depends = ["gate", "warm", "shared"] }
warm = "echo"
"""

_HEAD_PROJECT = """
[project]
name = "head"
requires-python = ">=3.14"
dependencies = ["tool", "numpy"]

[tool.lengths]
line = 99
"""

_SEARCH = """
import common.bases
import lib
import numpy
from tool import thing
from .near import other
"""


def _repository(path: Path, files: dict[str, str | bytes]) -> Path:
    """A git repository at `path` holding `files`, all committed."""
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            (path / name).write_bytes(content)
        else:
            (path / name).write_text(content, encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "--allow-empty", "-m", "seed")):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    return path


@pytest.fixture
def life(tmp_path: Path) -> Path:
    """A monorepo whose members lean on it in every way the check knows, one way each.

    `packages/lib` is flat and `packages/tool` a `src/` layout, both installable repositories.
    `research/head` requires `tool`, imports `lib` unrequired, `common` from the root's
    `PYTHONPATH` and `experiments` from beside its `src/`, climbs out through two paths,
    declares `[hosts]`, depends on a root task, and is reached by root tasks, a paper and the
    `PYTHONPATH`. `research/loose` is a manifest alone in no repository of its own.
    """
    root = tmp_path / "life"
    _repository(
        root,
        {
            _MANIFEST: _ROOT,
            "research/common/__init__.py": "",
            "research/common/bases.py": "",
            "research/loose/mainboard.toml": '[tasks]\nlook = "echo"\n',
        },
    )
    _repository(
        root / "packages" / "lib", {"pyproject.toml": _project("lib"), "lib/__init__.py": ""}
    )
    _repository(
        root / "packages" / "tool",
        {"pyproject.toml": _project("tool"), "src/tool/__init__.py": ""},
    )
    head = _repository(
        root / "research" / "head",
        {
            "pyproject.toml": _HEAD_PROJECT,
            _MANIFEST: _HEAD,
            "src/head/__init__.py": "import experiments.search\nfrom . import near\n",
            "experiments/__init__.py": "",
            "experiments/search.py": _SEARCH,
            "experiments/broken.py": "def (:\n",
            "experiments/zeroed.py": b"\0",
            "experiments/gone.py": "import lib\n",
        },
    )
    (head / "experiments" / "gone.py").unlink()
    return root


def _project(name: str) -> str:
    return f'[project]\nname = "{name}"\n'


def _uv(answers: dict[str, CommandResult]) -> object:
    """A `Process.capture` answering each package's install from `answers`, by its last word."""

    def capture(command: BaseCommand, *, timeout: float | None = None) -> CommandResult:
        assert timeout
        return answers[command.formulate()[-1]]

    return staticmethod(capture)


def test_every_member_is_judged_for_whoever_clones_it_alone(
    life: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mainboard.center.standalone.Process.capture",
        _uv(
            {
                "head": CommandResult(0, "head\n", ""),
                "lib": CommandResult(0, "\n", ""),
                "tool": CommandResult(
                    1, "", "  x No solution found\n\n  because tool needs cuda\n"
                ),
            }
        ),
    )

    rows = [
        (row.section, row.verdict.value, row.detail)
        for row in Standalone(composition(life / _MANIFEST)).sections()
    ]

    assert rows == [
        ("packages/lib: install", "fail", "lib installs alone but provides no importable package"),
        (
            "packages/tool: install",
            "fail",
            "x No solution found because tool needs cuda",
        ),
        (
            "research/head: standalone",
            "warn",
            "declares [hosts], which apply only when it stands alone; the root's govern it here",
        ),
        (
            "research/head: escape",
            "fail",
            "mainboard.toml python.deps.lib.path reaches ../../packages/lib, outside the member",
        ),
        (
            "research/head: escape",
            "fail",
            "mainboard.toml tasks.cache.env.DIR reaches ../cache, outside the member",
        ),
        (
            "research/head: task",
            "fail",
            "gate depends on shared, which the member does not declare",
        ),
        (
            "research/head: task",
            "fail",
            "kernel depends on shared, which the member does not declare",
        ),
        (
            "research/head: import",
            "fail",
            "imports common from research, which only the root's PYTHONPATH provides; "
            "first of 1 files: experiments/search.py",
        ),
        (
            "research/head: import",
            "fail",
            "imports experiments from beside src/, which the installed package does not "
            "carry; first of 1 files: src/head/__init__.py",
        ),
        (
            "research/head: import",
            "fail",
            "imports lib from packages/lib, whose lib its pyproject.toml does not require; "
            "first of 1 files: experiments/search.py",
        ),
        ("research/head: root", "warn", "root tasks head-figures, head-kernel reach into it"),
        ("research/head: root", "warn", "root papers head reach into it"),
        ("research/head: root", "warn", "root [env] PYTHONPATH reach into it"),
        ("research/head: install", "pass", "head installs alone and imports head"),
        (
            "research/loose: project",
            "fail",
            "has no pyproject.toml [project], so nobody can install it alone",
        ),
        (
            "research/loose: clone",
            "fail",
            "is not a repository of its own, so nobody can clone it alone",
        ),
    ]


def test_a_member_without_a_manifest_or_a_python_path_is_judged_on_what_it_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `PYTHONPATH`, no manifest, and a sibling with nothing to install: nothing to find."""
    root = tmp_path / "solo"
    _repository(root, {_MANIFEST: '[workspace]\nname = "solo"\nmembers = ["*"]\n'})
    _repository(
        root / "app", {"pyproject.toml": _project("app"), "app/__init__.py": "import notes\n"}
    )
    _repository(root / "notes", {_MANIFEST: "", "src/notes/__init__.py": ""})
    monkeypatch.setattr(
        "mainboard.center.standalone.Process.capture",
        _uv({"app": CommandResult(0, "app\n", "")}),
    )

    rows = Standalone(composition(root / _MANIFEST)).sections(["app", "notes"])

    assert [(row.section, row.detail) for row in rows] == [
        (
            "app: import",
            "imports notes from notes, which has no installable project; first of 1 files: "
            "app/__init__.py",
        ),
        ("app: install", "app installs alone and imports app"),
        ("notes: project", "has no pyproject.toml [project], so nobody can install it alone"),
    ]


def test_an_install_that_cannot_start_or_never_ends_is_a_failure(
    life: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standalone = Standalone(composition(life / _MANIFEST))

    def timed_out(command: BaseCommand, *, timeout: float | None = None) -> CommandResult:
        raise ProcessTimedOut("slow", [])

    monkeypatch.setattr("mainboard.center.standalone.Process.capture", staticmethod(timed_out))
    assert standalone.sections(["lib"])[-1].detail == "uv gave no answer in 1800 s"
    with local.env(PATH=""):
        assert standalone.sections(["packages/lib"])[-1].detail == "uv is not on PATH"


def test_a_clone_git_refuses_is_the_install_row(
    life: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Git.run

    def refusing(self: Git, *args: str, stdin: str = "", network: bool = False) -> CommandResult:
        if args[0] == "clone":
            return CommandResult(128, "", "fatal: repository not found\n")
        return real(self, *args, stdin=stdin, network=network)

    monkeypatch.setattr(Git, "run", refusing)
    rows = Standalone(composition(life / _MANIFEST)).sections(["tool"])
    assert rows[-1].detail == "fatal: repository not found"


@pytest.mark.parametrize(
    ("root", "names", "match"),
    [
        ('[workspace]\nname = "bare"\n', (), "declares no"),
        ('[workspace]\nname = "w"\nmembers = ["packages/*"]\n', ("ghost",), "no member 'ghost'"),
    ],
)
def test_the_check_refuses_a_workspace_without_members_or_an_unknown_one(
    life: Path, root: str, names: tuple[str, ...], match: str
) -> None:
    (life / _MANIFEST).write_text(root, encoding="utf-8")
    with pytest.raises(MissionError, match=match):
        Standalone(composition(life / _MANIFEST)).sections(names)


def test_the_verb_prints_the_rows_and_exits_on_a_failure(
    life: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "mainboard.center.standalone.Process.capture",
        _uv({"head": CommandResult(0, "head\n", "")}),
    )
    with pytest.raises(SystemExit) as exited:
        build(life)(["center", "members", "head", "--json"])
    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().out)[-1]["verdict"] == "pass"
