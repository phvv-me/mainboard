import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard.engines.compile import Provisioner, task_line
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.manifest import Manifest

_BARE = '[workspace]\nname = "w"\n'
_NODE = '[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n'
_PINNED = '[workspace]\nname = "w"\nplatforms = ["linux-64"]\n'
_WRAPPED = "pixi run --manifest-path .mainboard/pixi.toml --frozen"


def _solvable(provisioner: Provisioner) -> None:
    """Seed the lock a real `pixi install --resolve` would leave behind as it solves.

    The fake process writes nothing, so the recursive locked re-verify that follows a solve
    would find no lock to check.
    """
    provisioner.out.mkdir(exist_ok=True)
    provisioner.pixi.lock.write_text("version: 7\n")


def test_provision_compiles_and_installs_under_one_lock(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """A second provision recompiles nothing.

    The writer is a no-op once the generated file already matches.
    """
    provisioner = Provisioner(tmp_path, manifest_from(_PINNED))
    assert provisioner.out == tmp_path / ".mainboard"
    assert provisioner.pixi.manifest == provisioner.out / "pixi.toml"
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)
    compiled = provisioner.pixi.manifest.read_text()
    provisioner.provision()

    assert provisioner.pixi.manifest.read_text() == compiled
    assert SyncState.load(provisioner.out).envs.get("default") == provisioner.compiler.digest()


def test_activated_never_compiles_a_workspace_that_was_never_provisioned(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """First-time compilation is `provision`'s job, `activated` only catches up a stale one."""
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))

    with provisioner.activated():
        pass

    assert not provisioner.pixi.manifest.exists()


def test_activated_recompiles_a_provisioned_env_that_has_gone_stale(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """`activated` only compiles under its lock, it never touches `pixi install`."""
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    _solvable(provisioner)
    for _ in range(2):
        fp.register([fp.any()], stdout="environment ready\n")
    provisioner.provision(resolve=True)

    edited = Provisioner(tmp_path, manifest_from(f'{_BARE}[deps]\nripgrep = "*"\n'))
    with edited.activated():
        pass

    assert len(fp.calls) == 2  # no further pixi invocation happened during `activated`
    assert "ripgrep" in edited.pixi.manifest.read_text()


def test_activated_puts_a_second_stage_toolchains_binaries_ahead_of_the_env(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """A tool npm installed is reachable by name exactly like a conda one."""
    provisioner = Provisioner(tmp_path, manifest_from(_NODE))
    linked = provisioner.out / "node_modules" / ".bin"
    linked.mkdir(parents=True)
    env_bin = provisioner.pixi.env_prefix("default") / ("Scripts" if os.name == "nt" else "bin")
    env_bin.mkdir(parents=True)

    with provisioner.activated():
        assert local.env["PATH"].startswith(
            os.pathsep.join(
                (str(linked), str(provisioner.pixi.env_prefix("default")), str(env_bin))
            )
            if os.name == "nt"
            else os.pathsep.join((str(linked), str(env_bin), ""))
        )


def test_activated_leaves_out_a_directory_nothing_has_installed_into(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """An environment provisioned without a `[nodejs]` table exports no dead PATH entry."""
    provisioner = Provisioner(tmp_path, manifest_from(_BARE))
    before = local.env["PATH"]

    with provisioner.activated():
        assert local.env["PATH"] == before


@pytest.mark.parametrize(
    ("modules", "loaded"),
    [
        pytest.param({}, False, id="no-modules-leaves-the-surrounding-stack-alone"),
        pytest.param(
            {"singularity": "4.2.1"}, True, id="a-per-host-map-is-loaded-by-name-and-version"
        ),
    ],
)
def test_activate_writes_the_script_a_bare_shell_gets_the_whole_runtime_from(
    modules: dict[str, str],
    *,
    loaded: bool,
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
) -> None:
    """A remote job sourcing `activate.sh` gets the same PATH `activated` builds in process."""
    provisioner = Provisioner(tmp_path, manifest_from(_NODE))
    linked = provisioner.out / "node_modules" / ".bin"
    linked.mkdir(parents=True)
    fp.register([fp.any()], stdout="export PATH=/env/bin:$PATH\n")

    path = provisioner.activate(modules=modules)

    assert path == provisioner.out / "activate.sh"
    text = path.read_text()
    assert ("module load singularity/4.2.1" in text) is loaded
    assert "export PATH=/env/bin:$PATH" in text
    assert str(linked) in text


def test_activate_gives_a_named_environment_its_own_script(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """Installing one environment must not overwrite the activation another one is sourced by."""
    provisioner = Provisioner(tmp_path, manifest_from(f"{_BARE}[envs.serving]\n"))
    fp.register([fp.any()], stdout="export PATH=/env/serving/bin:$PATH\n")

    path = provisioner.activate("serving")

    assert path == provisioner.out / "activate-serving.sh"
    assert not (provisioner.out / "activate.sh").exists()


def test_provision_installs_the_second_stage_after_pixi(
    manifest_from: Callable[[str], Manifest],
    tmp_path: Path,
    fp: FakeProcess,
    stub_binary: Callable[[str], str],
) -> None:
    """Every second-stage manager ships as a conda package, so pixi has to land first."""
    npm = stub_binary("npm")
    provisioner = Provisioner(
        tmp_path, manifest_from(f'{_PINNED}[nodejs.deps]\nprettier = ">=3"\n')
    )
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)

    assert "prettier" in (provisioner.out / "package.json").read_text()
    assert [next(iter(call)) for call in fp.calls][-1] == npm


def test_a_refresh_asks_the_indexes_before_installing_and_blesses_the_result(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """Satisfying the manifest and being current differ, so `update` runs before the install."""
    provisioner = Provisioner(tmp_path, manifest_from(_PINNED))
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="lock updated\n")

    provisioner.provision(refresh=True)

    assert "update" in fp.calls[0]
    assert SyncState.load(provisioner.out).solved_from


@pytest.mark.parametrize(
    ("command", "env", "line"),
    [
        pytest.param(
            "lint --fix",
            "default",
            f"{_WRAPPED} -e default lint --fix",
            id="a-workspace-task-goes-to-pixi-with-its-arguments",
        ),
        pytest.param(
            "serve", "serving", f"{_WRAPPED} -e serving serve", id="an-env-declares-its-own-tasks"
        ),
        pytest.param("serve", "default", "serve", id="another-envs-task-is-not-a-task-here"),
        pytest.param(
            "python -c 'print(1)'",
            "default",
            "python -c 'print(1)'",
            id="a-command-line-is-not-a-task-name",
        ),
        pytest.param(
            "lint",
            "undeclared",
            f"{_WRAPPED} -e undeclared lint",
            id="an-env-nobody-declared-still-resolves-the-workspace-tasks",
        ),
    ],
)
def test_task_line_hands_only_a_declared_task_to_pixi(
    command: str, env: str, line: str, manifest_from: Callable[[str], Manifest]
) -> None:
    """The manifest path is relative because a wrapped command already changed into the root."""
    manifest = manifest_from(
        f'{_BARE}[tasks]\nlint = "ruff check"\n[envs.serving.tasks]\nserve = "vllm serve"\n'
    )
    assert task_line(manifest, command, env=env) == line
