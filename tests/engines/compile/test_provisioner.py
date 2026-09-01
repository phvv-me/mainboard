import json
import os
import tomllib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile import Provisioner, task_line
from mainboard.engines.compile.generated import GeneratedFiles
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_subprocess import FakeProcess

    from mainboard.manifest import Manifest

_BARE = '[workspace]\nname = "w"\n'
_NODE = '[workspace]\nname = "w"\n[nodejs.deps]\nprettier = ">=3"\n'
_PINNED = '[workspace]\nname = "w"\nplatforms = ["linux-64"]\n'
_WRAPPED = "pixi run --manifest-path .mainboard/envs/{env}/pixi.toml --frozen"


def _solvable(provisioner: Provisioner, environment: str = "default") -> None:
    """Seed the lock a real `pixi install --resolve` would leave behind as it solves.

    The fake process writes nothing, so the recursive locked re-verify that follows a solve
    would find no lock to check.
    """
    provisioner.environment_dir(environment).mkdir(parents=True, exist_ok=True)
    provisioner.pixi_for(environment).lock.write_text("version: 7\n")


def test_provision_compiles_and_installs_under_one_lock(
    manifest_from: Callable[[str], Manifest], tmp_path: Path, fp: FakeProcess
) -> None:
    """A second provision recompiles nothing.

    The writer is a no-op once the generated file already matches.
    """
    provisioner = Provisioner(tmp_path, manifest_from(_PINNED))
    assert provisioner.out == tmp_path / ".mainboard"
    assert provisioner.pixi.manifest == provisioner.out / "envs" / "default" / "pixi.toml"
    _solvable(provisioner)
    for _ in range(3):
        fp.register([fp.any()], stdout="environment ready\n")

    provisioner.provision(resolve=True)
    compiled = provisioner.pixi.manifest.read_text()
    provisioner.provision()

    assert provisioner.pixi.manifest.read_text() == compiled
    state = SyncState.load(provisioner.environment_dir())
    assert state.environment == "default"
    assert state.compiled_from == provisioner.compiler.digest()


def test_each_environment_compiles_into_an_independent_selected_manifest_shard(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """Default, inherited and isolated environments carry exactly their active scopes."""
    provisioner = Provisioner(
        tmp_path,
        manifest_from(
            """
            [workspace]
            name = "w"
            [deps]
            root = "*"
            [python.deps]
            local-root = { path = "packages/root", editable = true }
            [dev.deps]
            devtool = "*"
            [tasks]
            check = "python -m pytest"
            [envs.serving.deps]
            server = "*"
            [envs.isolated]
            no-default = true
            [envs.isolated.deps]
            kernel = "*"
            """
        ),
    )

    with GeneratedFiles(directory=provisioner.out).locked() as files:
        for environment in ("default", "serving", "isolated"):
            provisioner.compiler_for(environment).write(files)

    documents = {
        environment: tomllib.loads(
            provisioner.pixi_for(environment).manifest.read_text(encoding="utf-8")
        )
        for environment in ("default", "serving", "isolated")
    }
    assert set(documents["default"].get("dependencies", {})) == {"root"}
    assert set(documents["default"]["feature"]) == {"dev"}
    assert "serving" not in documents["default"].get("feature", {})
    assert set(documents["serving"].get("dependencies", {})) == {"root"}
    assert set(documents["serving"]["feature"]) == {"serving"}
    assert documents["serving"]["environments"] == {"serving": {"features": ["serving"]}}
    assert "dependencies" not in documents["isolated"]
    assert set(documents["isolated"]["feature"]) == {"isolated"}
    assert documents["isolated"]["environments"] == {
        "isolated": {"features": ["isolated"], "no-default-feature": True}
    }
    assert documents["default"]["pypi-dependencies"]["local-root"]["path"] == (
        "../../../packages/root"
    )
    assert documents["default"]["tasks"]["check"]["cwd"] == "../../.."
    assert provisioner.pixi_for("serving").env_prefix("serving") == (
        provisioner.out / "envs" / "serving" / ".pixi" / "envs" / "serving"
    )
    assert provisioner.artifact_for("serving") == (
        ".mainboard/envs/serving/pixi.toml",
        ".mainboard/envs/serving/pixi.lock",
        ".mainboard/envs/serving/state.toml",
    )


@pytest.mark.parametrize(
    "environment",
    [
        "../escape",
        "a/b",
        r"a\b",
        "a:b",
        ".hidden",
        "trailing.",
        "CON",
        "con.txt",
        "LPT9",
    ],
)
def test_an_environment_name_cannot_escape_or_alias_its_portable_shard_directory(
    environment: str, manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """One conservative path-segment contract is enforced before any path is constructed."""
    manifest = manifest_from(f'[workspace]\nname = "w"\n[envs.{json.dumps(environment)}]\n')
    with pytest.raises(MissionError, match="cannot name a generated directory"):
        Provisioner(tmp_path, manifest).environment_dir(environment)
    assert not (tmp_path / ".mainboard").exists()


def test_environment_names_that_are_portable_segments_keep_their_logical_spelling(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """Dots, underscores and hyphens remain available for ordinary logical names."""
    provisioner = Provisioner(
        tmp_path, manifest_from('[workspace]\nname = "w"\n[envs."py3.14_cuda-13"]\n')
    )
    assert provisioner.environment_dir("py3.14_cuda-13") == (
        tmp_path / ".mainboard" / "envs" / "py3.14_cuda-13"
    )


def test_task_wrapping_validates_the_environment_before_interpolating_its_manifest_path(
    manifest_from: Callable[[str], Manifest],
) -> None:
    """Remote command staging cannot manufacture an escaped shard path either."""
    manifest = manifest_from('[workspace]\nname = "w"\n[tasks]\ncheck = "pytest"\n')
    with pytest.raises(MissionError, match="cannot name a generated directory"):
        task_line(manifest, "check", env="../escape")


def test_environment_names_cannot_alias_on_a_case_insensitive_filesystem(
    manifest_from: Callable[[str], Manifest], tmp_path: Path
) -> None:
    """The same manifest keeps one shard per environment on all three operating systems."""
    manifest = manifest_from('[workspace]\nname = "w"\n[envs.Train]\n[envs.train]\n')
    with pytest.raises(MissionError, match="case-insensitive filesystem"):
        Provisioner(tmp_path, manifest)


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
    linked = provisioner.environment_dir() / "node_modules" / ".bin"
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
    linked = provisioner.environment_dir() / "node_modules" / ".bin"
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

    assert "prettier" in (provisioner.environment_dir() / "package.json").read_text()
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
    assert SyncState.load(provisioner.environment_dir()).solved_from


@pytest.mark.parametrize(
    ("command", "env", "line"),
    [
        pytest.param(
            "lint --fix",
            "default",
            f"{_WRAPPED.format(env='default')} -e default lint --fix",
            id="a-workspace-task-goes-to-pixi-with-its-arguments",
        ),
        pytest.param(
            "serve",
            "serving",
            f"{_WRAPPED.format(env='serving')} -e serving serve",
            id="an-env-declares-its-own-tasks",
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
            f"{_WRAPPED.format(env='undeclared')} -e undeclared lint",
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
