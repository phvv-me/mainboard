import tomllib
from typing import TYPE_CHECKING

import pytest
import tomlkit

from mainboard import MissionError
from mainboard.engines.compile.pixi_manifest import PixiManifest
from mainboard.engines.compile.state import SyncState
from mainboard.manifest import Manifest

from .support import CompilerFrom

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.engines.compile.backend import Pixi
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
    "engines": {"vserve": {"command": "true"}},
    "hosts": {"miyabi-g": {"kind": "pbs", "defaults": {"interact-queue": "interact-g"}}},
}


def test_every_declared_manifest_table_is_classified_by_this_suite() -> None:
    """A new table has to arrive with the edit that decides whether a compile can read it."""
    assert set(_EDITS) == set(Manifest.model_fields)


@pytest.mark.parametrize("table", sorted(_EDITS))
def test_a_manifest_table_moves_the_digest_exactly_when_a_compile_reads_it(
    table: str, compiler_from: CompilerFrom
) -> None:
    """What the compiled pixi manifest says, not a hand-kept list, is what makes an env stale.

    Editing a host profile, a container, a gate, a template or a var reaches no generated file,
    so every installed environment must stay fresh through it, and editing anything a compile
    translates must stale them all. The compiled text is the whole answer for the declared
    tables here: the second stage reads scopes (`[deps]`, `[dev]`, `[envs]`, `[on]`) and the
    workspace name, which are the same tables that already reach this text.
    """
    bare = compiler_from(_BARE)
    edited = compiler_from(tomlkit.dumps({**tomllib.loads(_BARE), table: _EDITS[table]}))
    compiled = [
        PixiManifest.from_manifest(
            compiler.manifest, project_name=_PROJECT, environment=compiler.environment
        ).to_toml()
        for compiler in (bare, edited)
    ]

    reaches_the_compile = compiled[0] != compiled[1]
    assert reaches_the_compile is (table not in Manifest.uncompiled and table != "envs")
    assert (bare.digest() != edited.digest()) is reaches_the_compile


def test_selected_environment_digests_follow_only_the_scopes_they_inherit(
    compiler_from: CompilerFrom,
) -> None:
    """Root edits reach default and inherited shards, but not a no-default shard."""
    base = """
        [workspace]
        name = "w"
        [deps]
        root = "1"
        [envs.serving.deps]
        server = "1"
        [envs.isolated]
        no-default = true
        [envs.isolated.deps]
        kernel = "1"
        [envs.unrelated.deps]
        other = "1"
    """
    edited = base.replace('root = "1"', 'root = "2"').replace('other = "1"', 'other = "2"')

    assert compiler_from(base).digest() != compiler_from(edited).digest()
    assert (
        compiler_from(base, environment="serving").digest()
        != compiler_from(edited, environment="serving").digest()
    )
    assert (
        compiler_from(base, environment="isolated").digest()
        == compiler_from(edited, environment="isolated").digest()
    )
    unrelated = base.replace('other = "1"', 'other = "2"')
    for environment in ("default", "serving", "isolated"):
        assert (
            compiler_from(base, environment=environment).digest()
            == compiler_from(unrelated, environment=environment).digest()
        )


def test_staleness_starts_once_something_has_been_compiled_to_be_stale_against(
    compiler_from: CompilerFrom, files: Writer, pixi: Pixi
) -> None:
    """A workspace with nothing compiled yet is not stale.

    First provisioning is `provision`'s job rather than `activated`'s.
    """
    compiler = compiler_from(_BARE)
    assert compiler.stale() is False

    pixi.manifest.write_text("placeholder")
    assert compiler.stale() is True

    compiler.write(files)
    assert compiler.stale() is False


@pytest.mark.parametrize(
    ("declared", "written"),
    [
        pytest.param("", True, id="the-workspace-asks-for-a-dotenv-loader-by-default"),
        pytest.param("dotenv = false\n", False, id="a-workspace-that-declines-gets-none"),
    ],
)
def test_the_dotenv_loader_is_generated_only_when_the_workspace_asks_for_it(
    declared: str, *, written: bool, compiler_from: CompilerFrom, files: Writer
) -> None:
    compiler = compiler_from(f'[workspace]\nname = "w"\n{declared}')
    compiler.write(files)
    assert (compiler.out / "dotenv.sh").exists() is written
    assert (compiler.out / "dotenv.bat").exists() is written


def test_a_variable_declared_false_is_unset_rather_than_set_to_an_empty_string(
    compiler_from: CompilerFrom, files: Writer
) -> None:
    """An empty variable is still defined, which is a different thing from an absent one.

    `[env]` could only ever set, so declaring a clean arithmetic environment was structurally
    impossible from the manifest and needed an `env -u` workaround downstream. pixi's own
    activation table cannot say "not set", so the clear becomes shell in a generated script that
    every consumer of the activation already sources.
    """
    declared = '[workspace]\nname = "w"\n[env]\nKEEP = "1"\n'
    compiler = compiler_from(f"{declared}OMP_NUM_THREADS = false\nMKL_NUM_THREADS = false\n")
    compiler.write(files)
    script = (compiler.out / "unset.sh").read_text(encoding="utf-8")
    windows_script = (compiler.out / "unset.bat").read_text(encoding="utf-8")
    assert "unset -v OMP_NUM_THREADS" in script and "unset -v MKL_NUM_THREADS" in script
    assert "set OMP_NUM_THREADS=" in windows_script and "set MKL_NUM_THREADS=" in windows_script
    assert "KEEP" not in script
    # Sourced after the dotenv loader, so an explicit clear beats a value `.env` filled in.
    compiled = tomllib.loads(compiler.pixi.manifest.read_text(encoding="utf-8"))
    assert compiled["activation"]["scripts"] == ["dotenv.sh", "unset.sh"]
    assert compiled["activation"]["env"] == {"KEEP": "1"}


def test_a_workspace_that_clears_nothing_carries_no_unset_script(
    compiler_from: CompilerFrom, files: Writer
) -> None:
    """And one that stops clearing has the script taken away rather than left behind stale."""
    clearing = compiler_from('[workspace]\nname = "w"\n[env]\nOMP_NUM_THREADS = false\n')
    clearing.write(files)
    assert (clearing.out / "unset.sh").exists()
    plain = compiler_from('[workspace]\nname = "w"\n[env]\nKEEP = "1"\n')
    plain.write(files)
    assert not (plain.out / "unset.sh").exists()
    assert not (plain.out / "unset.bat").exists()


def test_env_refuses_the_one_boolean_that_says_nothing() -> None:
    """A variable is set or taken away, so `true` is a typo rather than a third meaning."""
    with pytest.raises(ValueError, match="which says nothing"):
        Manifest.model_validate({"workspace": {"name": "w"}, "env": {"FOO": True}})


def test_write_never_blesses_a_lock_it_did_not_solve(
    compiler_from: CompilerFrom, files: Writer, pixi: Pixi
) -> None:
    """Compiling changes what a lock must answer to, never what it already answered to."""
    pixi.lock.write_text("version: 7\n")
    compiler = compiler_from(_BARE)
    compiler.write(files)
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
    other: str, *, same: bool, compiler_from: CompilerFrom, files: Writer
) -> None:
    """The digest leaves activation out.

    Leaving it in would make the digest depend on where the workspace lives and so refuse
    every host whose root differs from the machine that solved.
    """
    bare = compiler_from(_BARE)
    bare.write(files)
    before = bare.resolution_digest()

    edited = compiler_from(other)
    edited.write(files)
    assert (edited.resolution_digest() == before) is same


@pytest.mark.parametrize(
    ("edited", "same"),
    [
        pytest.param('serve = "vllm serve --port 8001"', True, id="the-command-it-runs"),
        pytest.param('serve = "vllm serve"\nwarm = "vllm bench"', True, id="one-more-task"),
        pytest.param(
            'serve = "vllm serve"\n[envs.serving.deps]\nvllm = "*"',
            False,
            id="a-dependency-declared-beside-them",
        ),
    ],
)
def test_editing_an_environments_own_tasks_leaves_the_lock_it_was_solved_against_alone(
    edited: str, *, same: bool, compiler_from: CompilerFrom, files: Writer
) -> None:
    """A per-environment task compiles into `[feature.<name>.tasks]`, which the pop never reached.

    So renaming a command moved this digest, refused the lock sitting beside it, forced a full
    re-solve on a machine that wanted no such thing, and then invalidated that same lock on every
    host already holding it. What a command is called cannot change which versions resolve.
    """
    base = '[workspace]\nname = "w"\n[envs.serving.tasks]\nserve = "vllm serve"\n'
    before = compiler_from(base, environment="serving")
    before.write(files)
    solved = before.resolution_digest()

    after = compiler_from(
        f'[workspace]\nname = "w"\n[envs.serving.tasks]\n{edited}\n',
        environment="serving",
    )
    after.write(files)
    assert (after.resolution_digest() == solved) is same


def test_the_resolution_digest_follows_every_local_python_projects_own_metadata(
    compiler_from: CompilerFrom, files: Writer, tmp_path: Path
) -> None:
    """Editable metadata drift counts as staleness wherever it is declared.

    Editable source metadata drifts the lock without changing a manifest byte, a path
    declared only inside `[envs.*]` counts too, and one with nothing on disk yet never
    raises.
    """
    compiler = compiler_from(
        """
        [workspace]
        name = "w"
        [envs.serving.python.deps]
        lab-core = { path = "packages/lab-core", editable = true }
        """,
        environment="serving",
    )
    compiler.write(files)
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
    compiler_from: CompilerFrom,
    files: Writer,
    pixi: Pixi,
) -> None:
    """The refusal compares the lock's own recorded resolution against what is on disk now."""
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    if solved:
        pixi.lock.write_text("version: 7\n")
    with pytest.raises(MissionError, match=refusal):
        compiler_from(_BARE).install_locked(files, resolve=False)


def test_install_locked_accepts_a_lock_solved_somewhere_else_from_this_very_tree(
    compiler_from: CompilerFrom, files: Writer, pixi: Pixi, fp: FakeProcess
) -> None:
    """A shipped lock installs without a solve.

    A host that never solved installs from a lock it was handed together with the manifest
    and the metadata that lock was solved from.
    """
    pixi.manifest.write_text('[workspace]\nplatforms = ["linux-64"]\n')
    pixi.lock.write_text("version: 7\n")
    compiler = compiler_from(_BARE)
    fp.register([fp.any()], stdout="environment ready\n")

    shipped = SyncState(environment="default", solved_from=compiler.resolution_digest())
    files.write(SyncState.path(compiler.out), shipped.render())
    compiler.install_locked(files, resolve=False)

    assert SyncState.load(compiler.out) == shipped


def test_install_locked_blesses_the_lock_after_a_successful_resolve(
    compiler_from: CompilerFrom, files: Writer, pixi: Pixi, fp: FakeProcess
) -> None:
    """Blessing happens only once a solve has returned without raising."""
    # `resolve=True` recurses into a second, locked install to verify the freshly solved lock
    # (`Pixi.install`'s known double-install wart), so the lock must already exist by then.
    pixi.lock.write_text("version: 7\n")
    compiler = compiler_from(_BARE)
    compiler.write(files)
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")

    compiler.install_locked(files, resolve=True)

    state = SyncState.load(compiler.out)
    assert state.environment == "default"
    assert state.solved_from == compiler.resolution_digest()
