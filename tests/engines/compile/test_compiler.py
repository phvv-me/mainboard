import tomllib
from typing import TYPE_CHECKING

import pytest
import tomlkit

from mainboard import MissionError
from mainboard.engines.compile.pixi_manifest import PixiManifest
from mainboard.engines.compile.state import SyncState
from mainboard.manifest import Manifest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
    from mainboard.engines.compile.compiler import Compiler
    from mainboard.engines.compile.generated import Writer
    from mainboard.manifest.schema.spec import Json

_BARE = '[workspace]\nname = "w"\n'
_GROWN = '[workspace]\nname = "w"\n[deps]\nripgrep = "*"\n'
_PROJECT = "mainboard"

# One edit per declared manifest table, so which side of `Manifest.uncompiled` a table sits on
# is decided by the compiler rather than by whoever last remembered to update a set. Every
# table gets an edit, and the test below reads the compiled output to say whether that edit
# reached it.
_EDITS: dict[str, Json] = {
    "workspace": {"name": "w", "version": "9.9.9"},
    "vars": {"where": "elsewhere"},
    "system": {"cuda": "12.6"},
    "env": {"PYTHONPATH": "src"},
    "deps": {"ripgrep": "*"},
    "on": {"linux-64": {"deps": {"ripgrep": "*"}}},
    "dev": {"deps": {"ruff": "*"}},
    "envs": {"serving": {"deps": {"vllm": "*"}}},
    "tasks": {"build": "make"},
    "gates": {"lint": "ruff check ."},
    "templates": {"lib": "templates/lib"},
    "tracking": {"project": "lab", "interval": 30},
    "containers": {"cuda": {"image": "docker://nvidia/cuda"}},
    "hosts": {"miyabi-g": {"kind": "pbs", "defaults": {"interact-queue": "interact-g"}}},
}


def _edited(table: str) -> str:
    """The bare manifest with one table's declared edit applied.

    table: the manifest table being edited, keyed into `_EDITS`.
    """
    return tomlkit.dumps({**tomllib.loads(_BARE), table: _EDITS[table]})


def test_every_declared_manifest_table_is_classified_by_this_suite() -> None:
    """A new table has to arrive with the edit that decides whether a compile can read it."""
    assert set(_EDITS) == set(Manifest.model_fields)


@pytest.mark.parametrize("table", sorted(_EDITS))
def test_a_manifest_table_moves_the_digest_exactly_when_a_compile_reads_it(
    table: str, compiler_from: Callable[[str], Compiler]
) -> None:
    """What the compiled pixi manifest says, not a hand-kept list, is what makes an env stale.

    Editing a host profile, a container, a gate, a template or a var reaches no generated file,
    so every installed environment must stay fresh through it, and editing anything a compile
    translates must stale them all. The compiled text is the whole answer for the declared
    tables here: the second stage reads scopes (`[deps]`, `[dev]`, `[envs]`, `[on]`) and the
    workspace name, which are the same tables that already reach this text.
    """
    bare, edited = compiler_from(_BARE), compiler_from(_edited(table))
    compiled = [
        PixiManifest.from_manifest(compiler.manifest, project_name=_PROJECT).to_toml()
        for compiler in (bare, edited)
    ]

    reaches_the_compile = compiled[0] != compiled[1]
    assert reaches_the_compile is (table not in Manifest.uncompiled)
    assert (bare.digest() != edited.digest()) is reaches_the_compile


def test_staleness_starts_once_something_has_been_compiled_to_be_stale_against(
    compiler_from: Callable[[str], Compiler], files: Writer, pixi: Pixi
) -> None:
    """A workspace with nothing compiled yet is not stale, since first provisioning is
    `provision`'s job rather than `activated`'s."""
    compiler = compiler_from(_BARE)
    assert compiler.stale() is False

    pixi.manifest.write_text("placeholder")
    assert compiler.stale() is True

    compiler.write(files, "default")
    assert compiler.stale("default") is False
    # A different, never-compiled env is still stale against the same fresh manifest content.
    assert compiler.stale("serving") is True


@pytest.mark.parametrize(
    ("declared", "written"),
    [
        pytest.param("", True, id="the-workspace-asks-for-a-dotenv-loader-by-default"),
        pytest.param("dotenv = false\n", False, id="a-workspace-that-declines-gets-none"),
    ],
)
def test_the_dotenv_loader_is_generated_only_when_the_workspace_asks_for_it(
    declared: str, *, written: bool, compiler_from: Callable[[str], Compiler], files: Writer
) -> None:
    compiler = compiler_from(f'[workspace]\nname = "w"\n{declared}')
    compiler.write(files, "default")
    assert (compiler.out / "dotenv.sh").exists() is written


def test_write_never_blesses_a_lock_it_did_not_solve(
    compiler_from: Callable[[str], Compiler], files: Writer, pixi: Pixi
) -> None:
    """Compiling changes what a lock must answer to, never what it already answered to."""
    pixi.lock.write_text("version: 7\n")
    compiler = compiler_from(_BARE)
    compiler.write(files, "default")
    assert not SyncState.load(compiler.out).solved_from


@pytest.mark.parametrize(
    ("other", "same"),
    [
        pytest.param(_GROWN, False, id="a-dependency-edit-can-move-what-resolves"),
        pytest.param(
            '[workspace]\nname = "w"\n[tasks]\nbuild = "make"\n[env]\nPYTHONPATH = "/elsewhere"\n',
            True,
            id="neither-tasks-nor-activation-can-change-which-versions-resolve",
        ),
    ],
)
def test_the_resolution_digest_covers_what_a_solve_reads(
    other: str, *, same: bool, compiler_from: Callable[[str], Compiler], files: Writer
) -> None:
    """Leaving activation in would make the digest depend on where the workspace lives and so
    refuse every host whose root differs from the machine that solved."""
    bare = compiler_from(_BARE)
    bare.write(files, "default")
    before = bare.resolution_digest()

    edited = compiler_from(other)
    edited.write(files, "default")
    assert (edited.resolution_digest() == before) is same


def test_the_resolution_digest_follows_every_local_python_projects_own_metadata(
    compiler_from: Callable[[str], Compiler], files: Writer, tmp_path: Path
) -> None:
    """Editable source metadata drifts the lock without changing a manifest byte, a path
    declared only inside `[envs.*]` counts too, and one with nothing on disk yet never raises."""
    compiler = compiler_from(
        """
        [workspace]
        name = "w"
        [envs.serving.python.deps]
        lab-core = { path = "packages/lab-core", editable = true }
        """
    )
    compiler.write(files, "default")
    dangling = compiler.resolution_digest()

    project = tmp_path / "packages" / "lab-core" / "pyproject.toml"
    project.parent.mkdir(parents=True)
    project.write_text('[project]\nname = "lab-core"\n')
    declared = compiler.resolution_digest()
    project.write_text('[project]\nname = "lab-core"\ndependencies = ["numpy"]\n')

    assert len({dangling, declared, compiler.resolution_digest()}) == 3


@pytest.mark.parametrize(
    ("solved", "refusal"),
    [
        pytest.param(
            True, "was not solved from this manifest", id="a-lock-this-tree-never-solved"
        ),
        pytest.param(False, r"pixi\.lock is missing", id="no-lock-at-all-is-pixis-own-diagnosis"),
    ],
)
def test_install_locked_refuses_a_lock_nothing_on_disk_vouches_for(
    *,
    solved: bool,
    refusal: str,
    compiler_from: Callable[[str], Compiler],
    files: Writer,
    pixi: Pixi,
) -> None:
    """The refusal compares the lock's own recorded resolution against what is on disk now."""
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    if solved:
        pixi.lock.write_text("version: 7\n")
    with pytest.raises(MissionError, match=refusal):
        compiler_from(_BARE).install_locked(files, "default", resolve=False)


def test_install_locked_accepts_a_lock_solved_somewhere_else_from_this_very_tree(
    compiler_from: Callable[[str], Compiler], files: Writer, pixi: Pixi, fp: FakeProcess
) -> None:
    """The shipped-artifact case, where a host that never solved installs from a lock it was
    handed together with the manifest and the metadata that lock was solved from."""
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    pixi.lock.write_text("version: 7\n")
    compiler = compiler_from(_BARE)
    fp.register([fp.any()], stdout="environment ready\n")

    shipped = SyncState(solved_from=compiler.resolution_digest())
    files.write(SyncState.path(compiler.out), shipped.render())
    compiler.install_locked(files, "default", resolve=False)

    assert SyncState.load(compiler.out) == shipped


def test_install_locked_blesses_the_lock_after_a_successful_resolve(
    compiler_from: Callable[[str], Compiler], files: Writer, pixi: Pixi, fp: FakeProcess
) -> None:
    """Blessing happens only once a solve has returned without raising."""
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    # `resolve=True` recurses into a second, locked install to verify the freshly solved lock
    # (`Pixi.install`'s known double-install wart), so the lock must already exist by then.
    pixi.lock.write_text("version: 7\n")
    compiler = compiler_from(_BARE)
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")

    compiler.install_locked(files, "default", resolve=True)

    assert SyncState.load(compiler.out).solved_from == compiler.resolution_digest()
