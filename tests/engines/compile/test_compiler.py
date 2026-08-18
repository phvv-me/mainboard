from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.compiler import Compiler
from mainboard.engines.compile.ecosystems import SecondStage
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.manifest import Manifest


def _compiler(tmp_path: Path, pixi: Pixi, manifest: Manifest) -> Compiler:
    out = pixi.manifest.parent
    return Compiler(tmp_path, manifest, out, pixi, SecondStage(tmp_path, manifest, out, pixi))


def test_digest_is_stable_for_equal_manifests_and_differs_for_different_ones(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    one = manifest_from('[workspace]\nname = "w"\n')
    same = manifest_from('[workspace]\nname = "w"\n')
    different = manifest_from('[workspace]\nname = "w"\n[deps]\nripgrep = "*"\n')
    assert _compiler(tmp_path, pixi, one).digest() == _compiler(tmp_path, pixi, same).digest()
    assert _compiler(tmp_path, pixi, one).digest() != _compiler(tmp_path, pixi, different).digest()


def test_stale_is_false_when_nothing_has_been_compiled_yet(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    assert _compiler(tmp_path, pixi, manifest).stale() is False


def test_stale_is_true_right_after_a_compile_exists_with_no_state(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    pixi.manifest.write_text("placeholder")
    # A compiled manifest with no matching state digest reads as stale.
    assert _compiler(tmp_path, pixi, manifest).stale() is True


def test_write_then_stale_round_trips_to_fresh(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    compiler = _compiler(tmp_path, pixi, manifest)
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        compiler.write(files, "default")
    assert pixi.manifest.exists()
    assert compiler.stale("default") is False
    # A different, never-compiled env is still stale against the same fresh manifest content.
    assert compiler.stale("serving") is True


def test_write_generates_the_dotenv_loader_when_declared(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, manifest).write(files, "default")
    assert (pixi.manifest.parent / "dotenv.sh").exists()


def test_write_omits_the_dotenv_loader_when_declined(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\ndotenv = false\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, manifest).write(files, "default")
    assert not (pixi.manifest.parent / "dotenv.sh").exists()


def test_first_write_never_marks_the_lock_stale_since_no_lock_exists_yet(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, manifest).write(files, "default")
    assert SyncState.load(pixi.manifest.parent).resolution_stale is False


def test_a_changed_local_python_projects_pyproject_marks_an_existing_lock_stale(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """Editable source metadata can drift the lock without changing a single manifest byte."""
    (tmp_path / "packages" / "lab-core").mkdir(parents=True)
    project_file = tmp_path / "packages" / "lab-core" / "pyproject.toml"
    project_file.write_text('[project]\nname = "lab-core"\ndependencies = ["numpy"]\n')
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [python.deps]
        lab-core = { path = "packages/lab-core", editable = true }
        """
    )
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, manifest).write(files, "default")
    assert SyncState.load(pixi.manifest.parent).resolution_stale is False

    pixi.lock.write_text("version: 7\n")
    project_file.write_text('[project]\nname = "lab-core"\ndependencies = ["numpy", "scipy"]\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, manifest).write(files, "default")
    assert SyncState.load(pixi.manifest.parent).resolution_stale is True


def test_local_python_projects_are_gathered_from_declared_envs_too(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """A path dependency declared only inside `[envs.*]` still feeds the resolution digest."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [envs.serving.python.deps]
        lab-core = { path = "packages/lab-core", editable = true }
        """
    )
    compiler = _compiler(tmp_path, pixi, manifest)
    assert compiler._local_python_projects() == ["packages/lab-core"]  # ruff:ignore[private-member-access]  reason=unit-tests the resolution-digest helper since=2026-08-17


def test_a_missing_local_project_pyproject_still_produces_a_stable_digest(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """A dangling path dependency (no `pyproject.toml` on disk yet) never raises."""
    manifest = manifest_from(
        """
        [workspace]
        name = "w"
        [python.deps]
        lab-core = { path = "packages/lab-core", editable = true }
        """
    )
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, manifest).write(files, "default")
    assert SyncState.load(pixi.manifest.parent).resolution_stale is False


def test_a_changed_compiled_manifest_marks_an_existing_lock_stale(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """A dependency edit is drift too, detected straight from the generated pixi.toml, not disk."""
    bare = manifest_from('[workspace]\nname = "w"\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, bare).write(files, "default")

    pixi.lock.write_text("version: 7\n")
    grown = manifest_from('[workspace]\nname = "w"\n[deps]\nripgrep = "*"\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, grown).write(files, "default")
    assert SyncState.load(pixi.manifest.parent).resolution_stale is True


def test_a_task_only_edit_never_marks_the_lock_stale(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    """`[tasks]` never affects resolution, so editing only a task must never demand a re-solve."""
    bare = manifest_from('[workspace]\nname = "w"\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, bare).write(files, "default")

    pixi.lock.write_text("version: 7\n")
    with_task = manifest_from('[workspace]\nname = "w"\n[tasks]\nbuild = "make"\n')
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        _compiler(tmp_path, pixi, with_task).write(files, "default")
    assert SyncState.load(pixi.manifest.parent).resolution_stale is False


def test_install_locked_refuses_a_stale_lock_without_resolve(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, pixi: Pixi
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    compiler = _compiler(tmp_path, pixi, manifest)
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        files.write(
            SyncState.path(pixi.manifest.parent), SyncState(resolution_stale=True).render()
        )
        with pytest.raises(MissionError, match="resolve=True"):
            compiler.install_locked(files, "default", resolve=False)


def test_install_locked_clears_a_stale_flag_after_a_successful_resolve(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    pixi: Pixi,
    fp: FakeProcess,
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    # `resolve=True` recurses into a second, locked install to verify the freshly solved lock
    # (`Pixi.install`'s known double-install wart), so the lock must already exist by then.
    pixi.lock.write_text("version: 7\n")
    compiler = _compiler(tmp_path, pixi, manifest)
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        files.write(
            SyncState.path(pixi.manifest.parent), SyncState(resolution_stale=True).render()
        )
        compiler.install_locked(files, "default", resolve=True)
    assert SyncState.load(pixi.manifest.parent).resolution_stale is False


def test_install_locked_leaves_a_fresh_state_untouched(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    pixi: Pixi,
    fp: FakeProcess,
) -> None:
    manifest = manifest_from('[workspace]\nname = "w"\n')
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    pixi.lock.write_text("version: 7\n")
    compiler = _compiler(tmp_path, pixi, manifest)
    fp.register([fp.any()], stdout="environment ready\n")
    with GeneratedFiles(directory=pixi.manifest.parent).locked() as files:
        compiler.install_locked(files, "default", resolve=False)
    assert not SyncState.path(pixi.manifest.parent).exists()
