from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from mainboard import Manifest, Project
from mainboard.engines.compile import LockPolicy, Provisioner, SyncState
from mainboard.engines.compile.backend import Pixi
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

    assert Doctor._compiled_before(current.compiler, state) is lags


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
    monkeypatch.setattr(Pixi, "version", lambda self: "0.0.0")
    return updates


def test_a_requirement_changing_ecosystem_stays_owed_to_a_solve_until_one_frees_it(
    tmp_path: Path, asked: list[tuple[str, list[str]]]
) -> None:
    assert _owed(tmp_path, _CONDA) == ()
    assert _owed(tmp_path, _MOVED) == ("pandas", "python-docx")
    assert _owed(tmp_path, _AGAIN) == _OWED

    compiler = Provisioner(
        tmp_path, Manifest.model_validate({"workspace": _WORKSPACE, **_AGAIN})
    ).compiler
    with GeneratedFiles(directory=Project().out(tmp_path)).locked() as files:
        assert compiler.locked(files, policy=LockPolicy.UPDATE)

    assert asked == [("default", list(_OWED))]
    assert SyncState.load(compiler.out).moved == ()
