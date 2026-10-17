"""A compile that lags an edit to tasks needs no install; one that lags a dependency does."""

from pathlib import Path

from mainboard.doctor import Doctor
from mainboard.engines.compile.provisioner import Provisioner
from mainboard.engines.compile.state import SyncState
from mainboard.manifest import Manifest


def _manifest(*, task: str, dependency: str) -> Manifest:
    return Manifest.model_validate(
        {
            "workspace": {"name": "doctor-test", "platforms": ["linux-64"]},
            "python": {"deps": {"example": dependency}},
            "tasks": {"hello": task},
        }
    )


def _lags(root: Path, compiled: Manifest, current: Manifest) -> bool:
    Provisioner(root, compiled).recompiled()
    provisioner = Provisioner(root, current)
    return Doctor._compiled_before(
        provisioner.compiler, SyncState.load(provisioner.environment_dir())
    )


def test_an_edited_task_leaves_nothing_to_install(tmp_path: Path) -> None:
    compiled = _manifest(task="echo one", dependency=">=1")

    assert not _lags(tmp_path, compiled, compiled)
    assert not _lags(tmp_path, compiled, _manifest(task="echo two", dependency=">=1"))


def test_an_edited_dependency_is_a_compile_to_install(tmp_path: Path) -> None:
    compiled = _manifest(task="echo one", dependency=">=1")

    assert _lags(tmp_path, compiled, _manifest(task="echo one", dependency=">=2"))
