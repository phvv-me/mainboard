from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from mainboard import Manifest, Project
from mainboard.engines.compile import LockPolicy, Provisioner, SyncState
from mainboard.engines.compile.backend import Pixi, PixiEngine
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.upkeep.doctor import Doctor

_WORKSPACE: JsonValue = {"name": "recompile", "platforms": ["linux-64"]}
_DOCTORED: dict[str, JsonValue] = {
    "workspace": _WORKSPACE,
    "python": {"deps": {"example": ">=1"}},
    "tasks": {"hello": "echo one"},
}
_CONDA: dict[str, JsonValue] = {
    "deps": {"pandas": "*", "numpy": "*"},
    "on": {"linux-64": {"python": {"deps": {"Python_Docx": "*"}}}},
}
_MOVED: dict[str, JsonValue] = {
    "deps": {"numpy": "*", "python-docx": "*"},
    "python": {"deps": {"pandas": "*", "httpx": "*"}},
}
_AGAIN: dict[str, JsonValue] = {
    "deps": {"numpy": "*", "python-docx": "*", "httpx": "*"},
    "python": {"deps": {"pandas": "*"}},
}
_OWED = ("httpx", "pandas", "python-docx")
_TASKED = Manifest.model_validate(
    {"workspace": _WORKSPACE, "tasks": {"ruff": "ruff check .", "lint": "ruff format ."}}
)


@pytest.mark.parametrize(
    ("edit", "lags"),
    [
        pytest.param({}, False, id="an-unedited-manifest"),
        pytest.param({"tasks": {"hello": "echo two"}}, False, id="an-edited-task"),
        pytest.param({"env": {"EXAMPLE": "1"}}, False, id="an-edited-activation-variable"),
        pytest.param({"python": {"deps": {"example": ">=2"}}}, True, id="an-edited-python-pin"),
        pytest.param({"deps": {"git": ">=2"}}, True, id="a-new-conda-requirement"),
        pytest.param(
            {"nodejs": {"deps": {"left-pad": "1.3.0"}}}, True, id="a-new-nodejs-requirement"
        ),
        pytest.param({"rust": {"deps": {"ripgrep": "14.0.0"}}}, True, id="a-new-rust-requirement"),
        pytest.param(
            {"workspace": {"name": "recompile", "platforms": ["linux-64", "osx-arm64"]}},
            True,
            id="an-added-platform",
        ),
    ],
)
def test_a_compile_that_lags_the_manifest_is_a_finding_only_when_an_install_carries_it_out(
    tmp_path: Path, edit: dict[str, JsonValue], lags: bool
) -> None:
    Provisioner(tmp_path, Manifest.model_validate(_DOCTORED)).recompiled()
    current = Provisioner(tmp_path, Manifest.model_validate({**_DOCTORED, **edit}))
    state = SyncState.load(current.environment_dir())

    assert Doctor._compiled_before(current.shard().compiler, state) is lags


def _owed(root: Path, tables: dict[str, JsonValue]) -> tuple[str, ...]:
    """What recompiling `tables` over the last compile leaves owed to a solve."""
    provisioner = Provisioner(root, Manifest.model_validate({"workspace": _WORKSPACE, **tables}))
    provisioner.recompiled()
    return SyncState.load(provisioner.environment_dir()).moved


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    """Every `pixi update` a solve asks for, each answered by writing a lock."""
    updates: list[tuple[str, list[str]]] = []

    def update(self: Pixi, env: str, names: Sequence[str] = ()) -> None:
        updates.append((env, list(names)))
        self.lock.write_text("version: 7\n", encoding="utf-8")

    monkeypatch.setattr(Pixi, "update", update)
    monkeypatch.setattr(PixiEngine, "version", lambda self: "0.0.0")
    return updates


def test_a_requirement_changing_ecosystem_stays_owed_to_a_solve_until_one_frees_it(
    tmp_path: Path, asked: list[tuple[str, list[str]]]
) -> None:
    assert _owed(tmp_path, _CONDA) == ()
    assert _owed(tmp_path, _MOVED) == ("pandas", "python-docx")
    assert _owed(tmp_path, _AGAIN) == _OWED

    compiler = (
        Provisioner(tmp_path, Manifest.model_validate({"workspace": _WORKSPACE, **_AGAIN}))
        .shard()
        .compiler
    )
    with GeneratedFiles(directory=Project().out(tmp_path)).locked() as files:
        assert compiler.locked(files, policy=LockPolicy.UPDATE)

    assert asked == [("default", list(_OWED))]
    assert SyncState.load(compiler.out).moved == ()


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, list[str]]]:
    """Every command pixi is asked to run, with the directory it was asked from."""
    asked: list[tuple[Path, list[str]]] = []

    def run(self: Pixi, command: Sequence[str], env: str = "default") -> int:
        asked.append((Path.cwd(), list(command)))
        return 0

    monkeypatch.setattr(Pixi, "run", run)
    return asked


@pytest.mark.parametrize(
    ("folder", "command", "said"),
    [
        pytest.param("src", ["python", "x.py"], True, id="a-bare-command-from-a-subfolder"),
        pytest.param("src", ["lint"], False, id="a-declared-task-from-a-subfolder"),
        pytest.param("", ["python", "x.py"], False, id="a-bare-command-from-the-root"),
    ],
)
def test_a_command_runs_at_the_workspace_root_and_says_so_when_that_is_not_where_it_was_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ran: list[tuple[Path, list[str]]],
    folder: str,
    command: list[str],
    said: bool,
) -> None:
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / folder)

    Provisioner(tmp_path, _TASKED).run(command, policy=LockPolicy.FROZEN, install=False)

    heard = capsys.readouterr()
    assert (f"running in {tmp_path}, the workspace root, not in " in heard.out + heard.err) is said
    assert ran == [(tmp_path.resolve(), command)]


def _generated_twice(root: Path) -> None:
    """Compile the default environment under two provisioners, as two commands of a process do."""
    for _ in range(2):
        provisioner = Provisioner(root, _TASKED)
        with GeneratedFiles(directory=provisioner.out).locked() as files:
            provisioner.shard().generate(files)


def test_a_task_named_like_an_installed_program_warns_once_naming_the_program(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binaries = Provisioner(tmp_path, _TASKED).shard().pixi.prefix("default").path / "bin"
    binaries.mkdir(parents=True)
    (binaries / "ruff").write_text("", encoding="utf-8")

    _generated_twice(tmp_path)

    said = capsys.readouterr()
    assert (said.out + said.err).count(f"task ruff shadows {binaries / 'ruff'}") == 1
    assert "task lint shadows" not in said.out + said.err


def test_a_task_named_like_a_program_the_environment_lacks_does_not_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generated_twice(tmp_path)

    said = capsys.readouterr()
    assert "shadows" not in said.out + said.err
